#!/usr/bin/env python
import os
import io
import sys
import errno
import argparse
import scipy.io
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from scipy import stats
from math import sqrt, degrees, radians, sin, cos, atan2, pi
from decorators import *
from datetime import date, datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

# Python 2
try:
    from urllib2 import urlopen, HTTPError
# Python 3
except ImportError:
    from urllib.request import urlopen, HTTPError


################################################################################
############################## Auxiliary Functions #############################
################################################################################


# Show progress bar for whatever task
def progressBar(count, total, status=''):
    barlen = 50
    filledlen = int(round(barlen * count / float(total)))

    percents = round(100. * count / float(total), 1)
    bar = '#' * filledlen + '-' * (barlen-filledlen)

    if percents < 100.0:
        sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
    else:
        sys.stdout.write('[%s] %s%s %s\n' % (bar, percents, '%',
                         'Done downloading data' + ' '*(len(status)-21)))
        sys.stdout.flush()


# Return list of lines in url; if HTTPError, repeat
@retry(HTTPError, tries=5, delay=0.5, backoff=1)
def accessURL(url):
    return [line.rstrip() for line in urlopen(url).readlines()]


# increase readability of given catalog times
def formatTime(time):
    return pd.Timestamp(' '.join(time.split('T'))[:-1])


# convert formatted time to epoch time
def toEpoch(time):
    fstr = '%Y-%m-%dT%H:%M:%S.%fZ'
    epoch = datetime(1970, 1, 1)
    finaltime = (datetime.strptime(time, fstr) - epoch).total_seconds()

    return finaltime


# trim catalogs so they span the same time window
def trimTimes(cat1, cat2, OTwin):
    mintime = max(cat1['time'].min(), cat2['time'].min())
    maxtime = min(cat1['time'].max(), cat2['time'].max())
    adjmin = mintime - OTwin
    adjmax = maxtime + OTwin

    cat1trim = cat1[cat1['time'].between(adjmin, adjmax, inclusive=True)].copy()
    cat2trim = cat2[cat2['time'].between(adjmin, adjmax, inclusive=True)].copy()
    cat1trim = cat1trim.reset_index(drop=True)
    cat2trim = cat2trim.reset_index(drop=True)

    return cat1trim, cat2trim


# calculate equirectangular distance between two points
def eqdist(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    x = (lon2 - lon1) * cos(0.5 * (lat2+lat1))
    y = lat2 - lat1
    eqd = 6371 * sqrt(x*x + y*y)

    return eqd


# calculate azimuth from point 1 to point 2
def getAZ(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    londiff = lon2 - lon1
    x = sin(londiff) * cos(lat2)
    y = cos(lat1)*sin(lat2) - (sin(lat1)*cos(lat2)*cos(londiff))
    az = (degrees(atan2(x, y)) + 360) % 360

    return az


# calculate azimuths for all matches between two catalogs
def getAZsandDists(cat1, cat2, cat1mIDs, cat2mIDs):

    azimuths = []
    dists = []

    for ix, ID in enumerate(cat1mIDs):
        mask1 = cat1['id'] == ID
        mask2 = cat2['id'] == cat2mIDs[ix]

        lon1, lat1 = cat1[mask1].longitude, cat1[mask1].latitude
        lon2, lat2 = cat2[mask2].longitude, cat2[mask2].latitude

        az = getAZ(lon1, lat1, lon2, lat2)
        dist = eqdist(lon1, lat1, lon2, lat2)
        azimuths.append(az)
        dists.append(dist)

    return azimuths, dists


# condition for event matching
def cond(row1, row2, OTwin, distwin):
    lon1, lat1 = row1.longitude, row1.latitude
    lon2, lat2 = row2.longitude, row2.latitude

    return (eqdist(lon1, lat1, lon2, lat2) < distwin) \
            & (abs(row1.time - row2.time) < OTwin)


# generate map bounds and gridsize
def getmapbounds(cat1, cat2):
    minlat1, maxlat1 = cat1['latitude'].min(), cat1['latitude'].max()
    minlon1, maxlon1 = cat1['longitude'].min(), cat1['longitude'].max()
    minlat2, maxlat2 = cat2['latitude'].min(), cat2['latitude'].max()
    minlon2, maxlon2 = cat2['longitude'].min(), cat2['longitude'].max()

    minlat, maxlat = min(minlat1, minlat2), max(maxlat1, maxlat2)
    minlon, maxlon = min(minlon1, minlon2), max(maxlon1, maxlon2)

    latdiff, londiff = (maxlat-minlat) / 5., (maxlon-minlon) / 5.
    lllat, lllon = max(minlat-latdiff, -90), max(minlon-londiff, -180)
    urlat, urlon = min(maxlat+latdiff, 90), min(maxlon+londiff, 180)

    if (lllon < 175) and (urlon > 175) \
        and (len(cat1[cat1['longitude'].between(-100, 100)]) == 0):

        lllon = cat1[cat1['longitude'] > 0].min()['longitude']
        urlon = 360 + cat1[cat1['longitude'] < 0].max()['longitude']
        clon = 180

    else:
        clon = 0

    gridsize = max(urlat-lllat, urlon-lllon) / 45.
    hgridsize, tgridsize = gridsize / 2., gridsize / 10.

    return lllat, lllon, urlat, urlon, gridsize, hgridsize, tgridsize, clon


# round number to nearest histogram bin edge (either "up" or "down")
def round2bin(number, binsize, direction):
    if direction == 'down':
        return number - (number%binsize)
    if direction == 'up':
        return number - (number%binsize) + binsize


################################################################################
################################# Main Functions ###############################
################################################################################


# download catalog data from earthquake.usgs.gov
def getData(catalog1, catalog2, startyear, endyear, minmag=0.1, maxmag=10,
            write=False):

    catalog1, catalog2 = catalog1.lower(), catalog2.lower()
    year = startyear
    alldata1, alldata2 = [], []
    cat1name = catalog1 if catalog1 else 'all'
    cat2name = catalog2 if catalog2 else 'all'

    if cat1name == cat2name:
        print('Catalogs cannot be the same. Exiting...')
        sys.quit()

    dirname = '%s-%s_%s-%s' % (cat1name, cat2name, startyear, endyear)
    f1name = '%s%s-%s.csv' % (cat1name, startyear, endyear)
    f2name = '%s%s-%s.csv' % (cat2name, startyear, endyear)
    bartotal = 12 * (endyear - startyear + 1)
    barcount = 1

    while (year <= endyear):
        
        month = 1
        yeardata1, yeardata2 = [], []

        while (month <= 12):

            if month in [4, 6, 9, 11]:
                endday = 30

            elif month == 2:
                checkLY = (year % 4)

                if (checkLY == 0):
                    endday = 29
                else:
                    endday = 28
            else:
                endday = 31

            startd = '-'.join([str(year), str(month)])
            endd = '-'.join([str(year), str(month), str(endday)])

            if not catalog1:
                url1 = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                       '.csv?starttime=' + startd + '-1%2000:00:00'
                       '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                       '&minmagnitude=' + str(minmag) + '&maxmagnitude='
                       + str(maxmag))
            else:
                url1 = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                       '.csv?starttime=' + startd + '-1%2000:00:00'
                       '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                       '&catalog=' + catalog1 + '&minmagnitude=' +
                       str(minmag) + '&maxmagnitude=' + str(maxmag))

            if not catalog2:
                url2 = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                       '.csv?starttime=' + startd + '-1%2000:00:00'
                       '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                       '&minmagnitude=' + str(minmag) + '&maxmagnitude='
                       + str(maxmag))
            else:
                url2 = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                       '.csv?starttime=' + startd + '-1%2000:00:00'
                       '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                       '&catalog=' + catalog2 + '&minmagnitude=' +
                       str(minmag) + '&maxmagnitude=' + str(maxmag))


            monthdata1 = accessURL(url1)
            monthdata2 = accessURL(url2)

            if (month != 1) or (year != startyear):
                del monthdata1[0]
                del monthdata2[0]

            yeardata1.append(monthdata1)
            yeardata2.append(monthdata2)

            progressBar(barcount, bartotal, 'Downloading data ... ')
            barcount += 1
            month += 1

        alldata1.append(yeardata1)
        alldata2.append(yeardata2)

        year += 1

    alldata1 = [item for sublist in alldata1 for item in sublist]
    alldata1 = [item for sublist in alldata1 for item in sublist]
    alldata2 = [item for sublist in alldata2 for item in sublist]
    alldata2 = [item for sublist in alldata2 for item in sublist]
    
    if write:
        with open('%s/%s' % (dirname, f1name), 'w') as openfile:
            for event in alldata1:
                openfile.write('%s\n' % event.decode())
        alldatadf1 = pd.read_csv('%s/%s' % (dirname, f1name))

        with open('%s/%s' % (dirname, f2name), 'w') as openfile:
            for event in alldata2:
                openfile.write('%s\n' % event.decode())
        alldatadf2 = pd.read_csv('%s/%s' % (dirname, f2name))

    else:
        with open('getDataTEMP1.csv', 'w') as openfile:
            for event in alldata1:
                openfile.write('%s\n' % event.decode())
        alldatadf1 = pd.read_csv('getDataTEMP1.csv')

        with open('getDataTEMP2.csv', 'w') as openfile:
            for event in alldata2:
                openfile.write('%s\n' % event.decode())
        alldatadf2 = pd.read_csv('getDataTEMP2.csv')
        os.remove('getDataTEMP1.csv')
        os.remove('getDataTEMP2.csv')

    if (len(alldatadf1) != 1) and (len(alldatadf2) != 1):
        return alldatadf1, alldatadf2
    else:
        print(('At least one catalog has no data available for that time'
               ' period. Quitting...'))
        sys.exit()


# gather basic catalog summary statistics
@printstatus('Creating basic catalog summary')
def basicCatSum(catalog, catname, dirname):

    lines = []

    lines.append('Catalog name: %s\n\n' % dirname[:-9].upper())

    lines.append('First date in catalog: %s\n' % catalog['time'].min())
    lines.append('Last date in catalog: %s\n\n' % catalog['time'].max())

    lines.append('Total number of events: %s\n\n' % len(catalog))

    lines.append('Minimum latitude: %s\n' % catalog['latitude'].min())
    lines.append('Maximum latitude: %s\n' % catalog['latitude'].max())
    lines.append('Minimum longitude: %s\n' % catalog['longitude'].min())
    lines.append('Maximum longitude: %s\n\n' % catalog['longitude'].max())

    lines.append('Minimum depth: %s\n' % catalog['depth'].min())
    lines.append('Maximum depth: %s\n' % catalog['depth'].max())
    lines.append('Number of 0 km depth events: %s\n'
                 % len(catalog[catalog['depth'] == 0]))
    lines.append('Number of NaN depth events: %s\n\n'
                 % len(catalog[pd.isnull(catalog['depth'])]))

    lines.append('Minimum magnitude: %s\n' % catalog['mag'].min())
    lines.append('Maximum magnitude: %s\n' % catalog['mag'].max())
    lines.append('Number of 0 magnitude events: %s\n'
                 % len(catalog[catalog['mag'] == 0]))
    lines.append('Number of NaN magnitude events: %s'
                 % len(catalog[pd.isnull(catalog['mag'])]))

    with open('%s_%ssummary.txt' % (dirname, catname), 'w') as sumfile:
        for line in lines:
            sumfile.write(line)


# trim catalogs and summarize comparison criteria
@printstatus('Creating summary of comparison criteria')
def compCriteria(cat1, cat2, dirname, reg, OTwin=16, distwin=100,
                 lowmag=3.5, magwin=0.5, depwin=np.nan):

    lines = []

    mintime = formatTime(max(cat1['time'].min(), cat2['time'].min()))
    maxtime = formatTime(min(cat1['time'].max(), cat2['time'].max()))

    lines.append('Overlapping time period: %s to %s\n' % (mintime, maxtime))
    lines.append('Region: %s\n' % reg)
    lines.append('Authoritative agency: %s\n' % reg)
    lines.append('Lower magnitude limit: %s\n\n' % lowmag)

    lines.append('Matching Criteria\n')
    lines.append('Time window: %s s\n' % OTwin)
    lines.append('Distance window: %s km\n\n' % distwin)

    lines.append('Problem Event Parameter Tolerance\n')
    lines.append('Magnitude tolerance: %s\n' % magwin)
    lines.append('Depth tolerance: %s km' % depwin)

    with open('%s_comparisoncriteria.txt' % dirname, 'w') as compfile:
        for line in lines:
            compfile.write(line)


# match events within two catalogs
@printstatus(status='Matching events')
def matchEvents(cat1, cat2, OTwin=16, distwin=100,
                lowmag=3.5, magwin=0.5, depwin=np.nan):

    cat1.loc[:, 'time'] = [toEpoch(x) for x in cat1['time']]
    cat2.loc[:, 'time'] = [toEpoch(x) for x in cat2['time']]

    cat1, cat2 = trimTimes(cat1, cat2, OTwin)

    cat1IDs, cat2IDs = [], []

    for i in range(len(cat1)):

        cat2ix = cat2[abs(cat2['time'] - cat1.ix[i]['time']) <= 16].index.values

        if len(cat2ix) != 0:

            C = np.array([eqdist(cat1.ix[i]['longitude'], cat1.ix[i]['latitude'],
                    cat2.ix[x]['longitude'], cat2.ix[x]['latitude'])
                    for x in cat2ix])

            inds = np.argwhere(C < 100)

            if len(inds) != 0:
                for ind in inds:
                    cat1ID = cat1.ix[i]['id']
                    cat2ID = cat2.ix[cat2ix[ind]]['id'].values[0]

                    cat1IDs.append(cat1ID)
                    cat2IDs.append(cat2ID)

    cat1matched = cat1[cat1['id'].isin(cat1IDs)].reset_index(drop=True)
    cat2matched = cat2[cat2['id'].isin(cat2IDs)].reset_index(drop=True)

    return cat1IDs, cat2IDs, cat1matched, cat2matched


# map catalog events only within the appropriate region
@printstatus('Mapping events from both catalogs')
def mapEvents(cat1, cat2, reg, dirname):
    
    lllat, lllon, urlat, urlon, _, _, _, clon = getmapbounds(cat1, cat2)

    regionmat = scipy.io.loadmat('../regions.mat')
    regdict = {regionmat['region'][i][0][0]: x[0]
               for i, x in enumerate(regionmat['coord'])}
    regzone = regdict[reg]
    reglons, reglats = [x[0] for x in regzone], [x[1] for x in regzone]

    cat1lons, cat1lats = cat1.longitude, cat1.latitude
    cat2lons, cat2lats = cat2.longitude, cat2.latitude

    minlon, maxlon = min(reglons)-2.5, max(reglons)+2.5
    minlat, maxlat = min(reglats)-2.5, max(reglats)+2.5

    plt.figure(figsize=(12,7))
    m = plt.axes(projection=ccrs.Robinson(central_longitude=clon))
    m.set_extent([lllon, urlon, lllat, urlat], ccrs.PlateCarree())
    m.coastlines('50m')

    m.scatter(cat1lons, cat1lats, color='b', s=2, zorder=4,
              transform=ccrs.PlateCarree())
    m.scatter(cat2lons, cat2lats, color='r', s=2, zorder=4,
              transform=ccrs.PlateCarree())
    m.plot(reglons, reglats, c='k', linestyle='--', zorder=5,
           transform=ccrs.PlateCarree())

    plt.savefig('%s_mapmatcheddetecs.png' % dirname, dpi=300)


# make polar scatter/histogram of azimuth vs. distance
@printstatus('Graphing polar histogram of azimuths and distances')
def makeAZDist(cat1, cat2, cat1mIDs, cat2mIDs, dirname,
               distwin=100, numbins=16):

    azimuths, distances = getAZsandDists(cat1, cat2, cat1mIDs, cat2mIDs)

    width = 2*pi / numbins
    razimuths = list(map(radians, azimuths))
    bins = np.linspace(0, 2*pi, numbins+1)
    h = np.histogram(razimuths, bins=bins)[0]
    hist = (float(distwin)/max(h)) * h
    bins = (bins + width/2)[:-1]

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(razimuths, distances, color='b', s=10)
    bars = ax.bar(bins, hist, width=width)
    ax.set_theta_zero_location('N')
    ax.set_rmax(distwin)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(112.5)

    for r, bar in list(zip(hist, bars)):
        bar.set_facecolor('b')
        bar.set_alpha(0.5)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.11)

    plt.savefig('%s_polarazimuth.png' % dirname, dpi=300)


# compare magnitudes of matched events
@printstatus('Comparing parameters of matched events')
def compareParams(cat1, cat2, cat1mIDs, cat2mIDs, param, dirname):
    cat1params, cat2params = [], []

    for ix, ID in enumerate(cat1mIDs):
        param1 = float(cat1[cat1['id'] == ID][param])
        param2 = float(cat2[cat2['id'] == cat2mIDs[ix]][param])

        cat1params.append(param1)
        cat2params.append(param2)

    #m, b = np.polyfit(cat1params, cat2params, 1)
    m, b, R, _, _ = stats.linregress(cat1params, cat2params)
    linegraph = [m*x + b for x in cat1params]
    R2 = R*R

    plt.scatter(cat1params, cat2params, edgecolor='b', facecolor=None)
    plt.plot(cat1params, linegraph, c='r', linewidth=1,
             label=r'$\mathregular{R^2}$ = %0.2f' % R2)
    plt.plot(cat1params, cat1params, c='k', linewidth=1, label='B = 1')
    plt.legend(loc='upper left')
    
    plt.savefig('%s_compare%s.png' % (dirname, param), dpi=300)


# make histogram of parameter differences between matched detections
@printstatus('Graphing parameter differences between matched events')
def makeDiffHist(cat1, cat2, cat1mIDs, cat2mIDs, param, binsize, dirname,
                 color='b', title='', xlabel='', ylabel='Count'):

    paramdiffs = []

    for ix, ID in enumerate(cat1mIDs):
        c1mask = cat1['id'] == ID
        c2mask = cat2['id'] == cat2mIDs[ix]
        cat1param = cat1[c1mask][param].values[0]
        cat2param = cat2[c2mask][param].values[0]
        pardiff = cat1param - cat2param
        paramdiffs.append(pardiff)

    minpardiff, maxpardiff = min(paramdiffs or [0]), max(paramdiffs or [0])
    pardiffdown = round2bin(minpardiff, binsize, 'down')
    pardiffup = round2bin(maxpardiff, binsize, 'up')
    numbins = int((pardiffup-pardiffdown) / binsize)
    pardiffbins = np.linspace(pardiffdown, pardiffup+binsize,
                              numbins+2) - binsize/2.
    labelbuff = float(pardiffup-pardiffdown) / numbins * 0.5

    plt.figure(figsize=(12,6))
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.hist(paramdiffs, pardiffbins, alpha=0.7, color=color, edgecolor='k')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.11)
    plt.tick_params(labelsize=12)
    plt.xlim(pardiffdown+binsize/2., pardiffup+binsize/2.)
    plt.ylim(0)

    plt.savefig('%s_%sdiffs.png' % (dirname, param), dpi=300)


################################################################################
################################################################################
################################################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('catalog1', nargs='?', default='',
                        help='pick first catalog to download data from')
    parser.add_argument('catalog2', nargs='?', default='',
                        help='pick second catalog to download data from')
    parser.add_argument('startyear', help='pick starting year')
    parser.add_argument('endyear', help='pick end year (to get a single year \
                        of data, enter same year as startyear)')

    parser.add_argument('-fd', '--forcedownload', action='store_true',
                        help='forces downloading of data even if .csv file\
                        exists')

    args = parser.parse_args()
    cat1, cat2 = args.catalog1.lower(), args.catalog2.lower()
    startyear, endyear = map(int, [args.startyear, args.endyear])
    download = args.forcedownload

    dirname = '%s-%s_%s-%s' % (cat1, cat2, startyear, endyear)

    if download:
        try:
            os.makedirs(dirname)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        datadf1, datadf2 = getData(cat1, cat2, startyear, endyear, write=True)
    else:
        # Python 2
        try:
            try:
                datadf1 = pd.read_csv('%s/%s%s-%s.csv' % 
                          (dirname, cat1, startyear, endyear))
                datadf2 = pd.read_csv('%s/%s%s-%s.csv' %
                          (dirname, cat2, startyear, endyear))
            except IOError:
                try:
                    os.makedirs(dirname)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise
                datadf1, datadf2 = getData(cat1, cat2, startyear, endyear,
                                           write=True)
        # Python 3
        except:
            try:
                datadf1 = pd.read_csv('%s/%s%s-%s.csv' % 
                          (dirname, cat1, startyear, endyear))
                datadf2 = pd.read_csv('%s/%s%s-%s.csv' %
                          (dirname, cat2, startyear, endyear))
            except FileNotFoundError:
                try:
                    os.makedirs(dirname)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise
                datadf1, datadf2 = getData(cat1, cat2, startyear, endyear,
                                           write=True)

    os.chdir(dirname)
    basicCatSum(datadf1, cat1, dirname)
    basicCatSum(datadf2, cat2, dirname)
    compCriteria(datadf1, datadf2, dirname, cat1.upper())
    cat1IDs, cat2IDs, newcat1, newcat2 = matchEvents(datadf1, datadf2)
    mapEvents(newcat1, newcat2, cat1.upper(), dirname)
    plt.close()
    makeAZDist(newcat1, newcat2, cat1IDs, cat2IDs, dirname)
    plt.close()
    compareParams(newcat1, newcat2, cat1IDs, cat2IDs, 'mag', dirname)
    plt.close()
    compareParams(newcat1, newcat2, cat1IDs, cat2IDs, 'depth', dirname)
    plt.close()
    makeDiffHist(newcat1, newcat2, cat1IDs, cat2IDs, 'time', 0.5, dirname,
                 xlabel='Time residuals (sec)')
    plt.close()
    makeDiffHist(newcat1, newcat2, cat1IDs, cat2IDs, 'mag', 0.1, dirname,
                 xlabel='Magnitude residuals')
    plt.close()
    makeDiffHist(newcat1, newcat2, cat1IDs, cat2IDs, 'depth', 2, dirname,
                 xlabel='Depth residuals (km)')


if __name__ == '__main__':

    try:
        main()
    except (KeyboardInterrupt, SystemError):
        sys.stdout.write('\nProgram canceled. Exiting...\n')
        sys.exit()
