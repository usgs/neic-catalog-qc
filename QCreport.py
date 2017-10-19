#!/usr/bin/env python
import os
import io
import sys
import errno
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt, radians, cos
from decorators import *
from datetime import date, datetime
from mpl_toolkits.basemap import Basemap
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


# generate map bounds and gridsize
def getmapbounds(catalog):
    minlat, maxlat = catalog['latitude'].min(), catalog['latitude'].max()
    minlon, maxlon = catalog['longitude'].min(), catalog['longitude'].max()
    latdiff, londiff = (maxlat-minlat) / 5., (maxlon-minlon) / 5.
    lllat, lllon = max(minlat-latdiff, -90), max(minlon-londiff, -180)
    urlat, urlon = min(maxlat+latdiff, 90), min(maxlon+londiff, 180)

    gridsize = max(urlat-lllat, urlon-lllon) / 45.
    hgridsize, tgridsize = gridsize / 2., gridsize / 10.

    return lllat, lllon, urlat, urlon, gridsize, hgridsize, tgridsize


# round number to nearest grid-square center
def round2center(num, gridsize):
    hgridsize = gridsize / 2

    return num - (num%gridsize) + hgridsize


# round number to nearest timezone longitude
def round2lon(num):
    return 15 * round(num / 15.)


# add corresponding centers to catalog
def addcenters(catalog, gridsize):
    zippedlatlon = list(zip(round2center(catalog['latitude'], gridsize),
                            round2center(catalog['longitude'], gridsize)))
    catalog = catalog.reset_index()
    catalog.loc[:,'center'] = pd.Series(zippedlatlon)
    catalog = catalog.set_index('index')
    catalog.index.names = ['']

    return catalog


# group detections by nearest grid-square center, and return min/max of counts
def grouplatlons(catalog, minmag=0):

    if not catalog['mag'].isnull().all():
        magmask = catalog['mag'] >= minmag
        groupedlatlons = catalog[magmask].groupby('center')
        groupedlatlons = groupedlatlons.count().sort_index()
    elif catalog['Mag'].isnull().all() and (minmag != 0):
        groupedlatlons = catalog.groupby('center').count().sort_index()
        print("No magnitude data in catalog - plotting all events")
    else:
        groupedlatlons = catalog.groupby('center').count().sort_index()
    groupedlatlons = groupedlatlons[['id']]
    groupedlatlons.columns = ['count']
    cmin = min(list(groupedlatlons['count']) or [0])
    cmax = max(list(groupedlatlons['count']) or [0])

    return groupedlatlons, cmin, cmax


# create a list of red RGB values using colmin and colmax with numcolors colors
def range2rgb(rmin, rmax, numcolors):
    colors = np.linspace(rmax/255., rmin/255., numcolors)
    colors = [(min(1, x), max(0, x-1), max(0, x-1)) for x in colors]

    return colors


# draw rectangle with vertices given in degrees
def draw_grid(lats, lons, m, col, alpha=1):
    x, y = m(lons, lats)
    xy = list(zip(x,y))
    poly = Polygon(xy, facecolor=col, alpha=alpha, edgecolor='k', zorder=11)
    plt.gca().add_patch(poly)


# round number to nearest histogram bin edge (either "up" or "down")
def round2bin(number, binsize, direction):
    if direction == 'down':
        return number - (number%binsize)
    if direction == 'up':
        return number - (number%binsize) + binsize


# Wiemer and Wyss (2000) method for determing a and b values
def WW2000(Mc, mags, binsize):

    mags = mags[~np.isnan(mags)]
    mags = np.around(mags, 1)
    Mc_vec = np.arange(Mc-1.5, Mc+1.5+binsize/2., binsize)
    max_mag = max(mags)
    Corr = binsize / 2.
    bvalue = np.zeros(len(Mc_vec))
    std_dev = np.zeros(len(Mc_vec))
    avalue = np.zeros(len(Mc_vec))
    R = np.zeros(len(Mc_vec))

    for ii in range(len(Mc_vec)):
        M = mags[mags >= Mc_vec[ii]]
        Mag_bins_edges = np.arange(Mc_vec[ii]-binsize/2., max_mag+binsize,
                                   binsize)
        Mag_bins_centers = np.arange(Mc_vec[ii], max_mag+binsize/2., binsize)

        cdf = np.zeros(len(Mag_bins_centers))

        for jj in range(len(cdf)):
            cdf[jj] = np.count_nonzero(~np.isnan(mags[
                                       mags >= Mag_bins_centers[jj]]))


        bvalue[ii] = np.log10(np.exp(1))/(np.average(M) - (Mc_vec[ii]-Corr))
        std_dev[ii] = bvalue[ii]/sqrt(cdf[0])

        avalue[ii] = np.log10(len(M)) + bvalue[ii]*Mc_vec[ii]
        log_L = avalue[ii] - bvalue[ii]*Mag_bins_centers
        L = 10.**log_L

        B, _ = np.histogram(M, Mag_bins_edges)
        S = abs(np.diff(L))
        R[ii] = (sum(abs(B[:-1] - S))/len(M))*100

    ind = np.where(R <= 10)

    if not ind:
        ii = ind[0]
    else:
        ii = list(R).index(min(R))

    Mc = Mc_vec[ii]
    bvalue = bvalue[ii]
    avalue = avalue[ii]
    std_dev = std_dev[ii]
    Mag_bins = np.arange(0, max_mag+binsize/2., binsize)
    L = 10.**(avalue-bvalue*Mag_bins)

    return Mc, bvalue, avalue, L, Mag_bins, std_dev


# convert from ComCat time format to Unix/epoch time
def toEpoch(ogtime):

    fstr = '%Y-%m-%dT%H:%M:%S.%fZ'
    epoch = datetime(1970, 1, 1)
    epochtime = (datetime.strptime(ogtime, fstr) - epoch).total_seconds()

    return epochtime


# calculate equirectangular distance between two points
def eqdist(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    x = (lon2 - lon1) * cos(0.5 * (lat2+lat1))
    y = lat2 - lat1
    eqd = 6371 * sqrt(x*x + y*y)

    return eqd


################################################################################
################################# Main Functions ###############################
################################################################################


# download catalog data from earthquake.usgs.gov
def getData(catalog, startyear, endyear, minmag=0.1, maxmag=10, system=0,
            write=False):

    systems = ['prod01', 'prod02', 'dev', 'dev01', 'dev02']
    SY = systems[system]
    catalog = catalog.lower()
    year = startyear
    alldata = []
    catname = catalog if catalog else 'all'
    dirname = '%s%s-%s' % (catname, startyear, endyear)
    fname = '%s.csv' % dirname
    bartotal = 12 * (endyear - startyear + 1)
    barcount = 1

    while (year <= endyear):
        
        month = 1
        yeardata = []

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

            if not catalog:
                if (system == 0):
                    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                           '.csv?starttime=' + startd + '-1%2000:00:00'
                           '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                           '&minmagnitude=' + str(minmag) + '&maxmagnitude='
                           + str(maxmag))
                else:
                    url = ('https://' + SY + '-earthquake.cr.usgs.gov/fdsnws/'
                           'event/1/query.csv?starttime=' + startd + '-1%2000:'
                           '00:00&endtime=' + endd + '%2023:59:59&orderby='
                           'time-asc&minmagnitude=' + str(minmag) +
                           '&maxmagnitude=' + str(maxmag))
            else:
                if (system == 0):
                    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                           '.csv?starttime=' + startd + '-1%2000:00:00'
                           '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                           '&catalog=' + catalog + '&minmagnitude=' +
                           str(minmag) + '&maxmagnitude=' + str(maxmag))
                else:
                    url = ('https://' + SY + '-earthquake.cr.usgs.gov/fdsnws/'
                           'event/1/query.csv?starttime=' + startd + '-1%2000:'
                           '00:00&endtime=' + endd + '%2023:59:59&orderby='
                           '&catalog=' + catalog + 'time-asc&minmagnitude=' +
                           str(minmag) + '&maxmagnitude=' + str(maxmag))

            monthdata = accessURL(url)

            if (month != 1) or (year != startyear):
                del monthdata[0]

            yeardata.append(monthdata)

            progressBar(barcount, bartotal, ('Downloading data from '
                                             'earthquake.usgs.gov ...'))
            barcount += 1
            month += 1

        alldata.append(yeardata)

        year += 1

    alldata = [item for sublist in alldata for item in sublist]
    alldata = [item for sublist in alldata for item in sublist]
    
    if write:
        with open('%s/%s' % (dirname, fname), 'w') as openfile:
            for event in alldata:
                openfile.write('%s\n' % event.decode())
        alldatadf = pd.read_csv('%s/%s' % (dirname, fname))
    else:
        with open('getDataTEMP.csv', 'w') as openfile:
            for event in alldata:
                openfile.write('%s\n' % event.decode())
        alldatadf = pd.read_csv('getDataTEMP.csv')
        os.remove('getDataTEMP.csv')

    if len(alldatadf) != 1:
        return alldatadf
    else:
        print('Catalog has no data available for that time period. Quitting...')
        sys.exit()


# gather basic catalog summary statistics
@printstatus('Creating basic catalog summary')
def basicCatSum(catalog, dirname):

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

    with open('%s_catalogsummary.txt' % dirname, 'w') as sumfile:
        for line in lines:
            sumfile.write(line)


# make scatter plot of detections with magnitudes (if applicable)
@printstatus('Mapping earthquake locations')
def mapDetecs(catalog, dirname, minmag=0, mindep=-50, title='', proj='cyl',
              color='r', res='c', mark='x', marksize=15):

    catalog = catalog[(catalog['mag'] >= minmag)
                      & (catalog['depth'] >= mindep)].copy()

    # define map bounds
    lllat, lllon, urlat, urlon = getmapbounds(catalog)[0:4]

    plt.figure(figsize=(12,7))
    m = Basemap(projection=proj, llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon, resolution=res)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='wheat', lake_color='lightblue')
    m.drawcoastlines()

    # if catalog has magnitude data
    if not catalog['mag'].isnull().all():
        bins = [0, 5, 6, 7, 8, 15]
        binnames = ['< 5', '5-6', '6-7', '7-8', r'$\geq$8']
        binsizes = [10, 25, 50, 100, 400]
        bincolors = ['g', 'b', 'y', 'r', 'r']
        binmarks = ['o', 'o', 'o', 'o', '*']
        catalog.loc[:,'maggroup'] = pd.cut(catalog['mag'], bins, 
                                           labels=binnames)

        for i, label in enumerate(binnames):
            mgmask = catalog['maggroup'] == label
            rcat = catalog[mgmask]
            lons, lats = list(rcat['longitude']), list(rcat['latitude'])
            x, y = m(lons, lats)
            m.scatter(x, y, s=binsizes[i], marker=binmarks[i], c=bincolors[i],
                      label=binnames[i], alpha=0.8, zorder=10)

        plt.legend(loc='lower left')

    # if catalog does not have magnitude data
    else:
        lons, lats = list(catalog['longitude']), list(catalog['latitude'])
        x, y = m(lons, lats)
        m.scatter(x, y, s=marksize, marker=mark, c=color, zorder=10)

    plt.title(title)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    plt.savefig('%s_mapdetecs.png' % dirname, dpi=300)


# map detections and a grid of detection density 
# (rmax=510 is white, rmin=0 is black)
@printstatus('Mapping earthquake density')
def mapDetecNums(catalog, dirname, title='', proj='cyl', lon0=0, res='c',
                 numcolors=20, rmin=77, rmax=490, minmag=0, pltevents=True):

    # generate bounds for map
    mask = catalog['mag'] >= minmag

    lllat, lllon, urlat, urlon, gridsize, hgridsize = \
        getmapbounds(catalog[mask])[0:6]

    catalog = addcenters(catalog, gridsize)
    groupedlatlons, cmin, cmax = grouplatlons(catalog, minmag=minmag)

    # print message if there are no detections with magnitudes above minmag
    if cmax == 0:
        print("No detections over magnitude %s" % minmag)

    # create color gradient from light red to dark red
    colors = range2rgb(rmin, rmax, numcolors)

    # put each center into its corresponding color group
    colorgroups = list(np.linspace(0, cmax, numcolors))
    groupedlatlons.loc[:,'group'] = np.digitize(groupedlatlons['count'], 
                                                colorgroups)

    # create map
    plt.figure(figsize=(12,6))
    m = Basemap(projection=proj, llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon, resolution=res)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='wheat', lake_color='lightblue')
    m.drawcoastlines()
    plt.title(title, fontsize=20)
    plt.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.01)

    # create color map based on rmin and rmax
    cmap = LinearSegmentedColormap.from_list('CM', colors)._resample(numcolors)

    # make dummy plot for setting color bar
    xx, yy = list(catalog['longitude']), list(catalog['latitude'])
    colormesh = m.pcolormesh(colors, colors, colors, cmap=cmap, alpha=1,
                             vmin=0, vmax=cmax)

    # format color bar
    cbticks = [x for x in np.linspace(0, cmax, numcolors+1)]
    cbar = m.colorbar(colormesh, ticks=cbticks)
    cbar.ax.set_yticklabels([('%.0f' % x) for x in cbticks])
    cbar.set_label('# of detections', rotation=270, labelpad=15)

    # plot rectangles with color corresponding to number of detections
    for center, count, cgroup in groupedlatlons.itertuples():
        minlat, maxlat = center[0]-hgridsize, center[0]+hgridsize
        minlon, maxlon = center[1]-hgridsize, center[1]+hgridsize
        glats = [minlat, maxlat, maxlat, minlat]
        glons = [minlon, minlon, maxlon, maxlon]

        color = colors[cgroup-1]

        draw_grid(glats, glons, m, col=color, alpha=0.8)

    # if provided, plot detection epicenters
    if pltevents and not catalog['mag'].isnull().all():
        magmask = catalog['mag'] >= minmag
        x = list(catalog['longitude'][magmask])
        y = list(catalog['latitude'][magmask])
        m.scatter(x, y, c='k', s=7, marker='x', zorder=5)
    elif catalog['mag'].isnull().all():
        x = list(catalog['longitude'])
        y = list(catalog['latitude'])
        m.scatter(x, y, c='k', s=7, marker='x', zorder=5)

    plt.savefig('%s_eqdensity.png' % dirname, dpi=300)


# plot histogram grouped by some parameter
@printstatus('Making histogram of given parameter')
def makeHist(catalog, param, binsize, dirname, color='b', title='', xlabel='', 
             ylabel='Count', countlabel=False):

    paramlist = catalog[pd.notnull(catalog[param])][param].tolist()
    minparam, maxparam = min(paramlist), max(paramlist)
    paramdown = round2bin(minparam, binsize, 'down')
    paramup = round2bin(maxparam, binsize, 'up')
    numbins = int((paramup-paramdown) / binsize)
    labelbuff = float(paramup-paramdown) / numbins * 0.5

    diffs = [abs(paramlist[i+1]-paramlist[i]) for i in range(len(paramlist))
             if i+1 < len(paramlist)]
    diffs = [round(x, 1) for x in diffs if x>0]

    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    if min(diffs) == binsize:
        parambins = np.linspace(paramdown, paramup+binsize,
                                numbins+2) - binsize/2.
        plt.xlim(paramdown+binsize/2., paramup+binsize/2.)
    else:
        parambins = np.linspace(paramdown, paramup, numbins+1)
        plt.xlim(paramdown, paramup)

    h = plt.hist(paramlist, parambins, alpha=0.7, color=color, edgecolor='k')
    maxbarheight = max([h[0][x] for x in range(numbins)] or [0])
    labely = maxbarheight / 50.

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.11)

    plt.ylim(0, maxbarheight*1.1+0.1)

    # put count numbers above the bars if countlabel=True
    if countlabel:
        for i in range(numbins):
            plt.text(h[1][i]+labelbuff, h[0][i]+labely, '%0.f' % h[0][i], 
                     size=12, ha='center')

    plt.savefig('%s_%shistogram.png' % (dirname, param), dpi=300)


# make histogram either by hour of the day or by date
@printstatus('Making histogram of given time duration')
def makeTimeHist(catalog, timelength, dirname, title='', xlabel='',
                 ylabel='Count'):

    timelist = catalog['time']
    mintime, maxtime = timelist.min(), timelist.max()

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.ylabel(ylabel, fontsize=14)

    if (timelength == 'hour'):
        lons = np.linspace(-180, 180, 25).tolist()
        hours = np.linspace(-12, 12, 25).tolist()

        tlonlist = catalog.loc[:, ['longitude', 'time']]
        tlonlist.loc[:, 'rLon'] = round2lon(tlonlist['longitude'])

        tlonlist.loc[:, 'hour'] = [int(x.split('T')[1].split(':')[0])
                                   for x in tlonlist['time']]
        tlonlist.loc[:, 'rhour'] = [x.hour + hours[lons.index(x.rLon)]
                                    for x in tlonlist.itertuples()]

        tlonlist.loc[:, 'rhour'] = [x+24 if x<0 else x-24 if x>23 else x
                                    for x in tlonlist['rhour']]

        hourlist = tlonlist.rhour.tolist()
        hourbins = np.linspace(-0.5, 23.5, 25)

        plt.hist(hourlist, hourbins, alpha=0.7, color='b', edgecolor='k')
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.xlim(-0.5, 23.5)

    elif (timelength == 'day'):
        eqdates, counts = [], []
        daylist = [x.split('T')[0] for x in timelist]
        daydf = pd.DataFrame({'date': daylist})
        daydf['date'] = daydf['date'].astype('datetime64[ns]')
        daydf = daydf.groupby([daydf['date'].dt.year,
                               daydf['date'].dt.month,
                               daydf['date'].dt.day]).count()

        for eqdate in daydf.itertuples():
            eqdates.append(eqdate.Index)
            counts.append(eqdate.date)

        eqdates = [date(x[0], x[1], x[2]) for x in eqdates]
        minday, maxday = min(eqdates), max(eqdates)

        plt.bar(eqdates, counts, alpha=1, color='b', width=1)
        plt.xlabel('Date', fontsize=14)
        plt.xlim(minday, maxday)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.11)

    plt.savefig('%s_%shistogram.png' % (dirname, timelength), dpi=300)


# make bar graph of mean time separation between events by date
@printstatus('Graphing mean time separation')
def graphTimeSep(catalog, dirname):

    catalog['convtime'] = [' '.join(x.split('T')).split('.')[0]
                           for x in catalog['time'].tolist()]
    catalog['convtime'] = catalog['convtime'].astype('datetime64[ns]')
    catalog['dt'] = catalog.convtime.diff().astype('timedelta64[ns]')
    catalog['dtmin'] = catalog['dt'] / pd.Timedelta(minutes=1)

    mindate = catalog['convtime'].min() - pd.Timedelta(days=15)
    maxdate = catalog['convtime'].max() - pd.Timedelta(days=15)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Time separation (min)')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off',
                   left='off', right='off')

    ax1 = fig.add_subplot(311)
    plt.plot(catalog['convtime'], catalog['dtmin'], alpha=1, color='b')
    plt.xlabel('Date')
    plt.title('Time separation between events')
    plt.xlim(mindate+pd.Timedelta(days=15), maxdate)
    plt.ylim(0)

    ax2 = fig.add_subplot(312)
    month_max = catalog.resample('1M', on='convtime').max()['dtmin']
    months = month_max.index.map(lambda x: x.strftime('%Y-%m')).tolist()
    months = [date(int(x[:4]), int(x[-2:]), 1)  for x in months]
    plt.bar(months, month_max.tolist(), color='b', alpha=1, width=31)
    plt.xlabel('Month')
    plt.title('Maximum event separation by month')
    plt.xlim(mindate, maxdate)

    ax3 = fig.add_subplot(313)
    month_med = catalog.resample('1M', on='convtime').median()['dtmin']
    plt.bar(months, month_med.tolist(), color='b', alpha=1, width=31)
    plt.xlabel('Month')
    plt.title('Median event separation by month')
    plt.tight_layout()
    plt.xlim(mindate, maxdate)

    plt.savefig('%s_timeseparation.png' % dirname, dpi=300)


# plot catalog magnitude completeness (NEEDS FIXING; Mc_est VALUES WRONG)
@printstatus('Graphing magnitude completeness')
def catMagComp(catalog, dirname, magbin=0.1):

    catalog = catalog[pd.notnull(catalog['mag'])]
    EQEvents = catalog[catalog['type'] == 'earthquake']
    mags = np.array(catalog[catalog['mag'] > 0]['mag'])
    mags = np.around(mags, 1)

    minmag, maxmag = 0, max(mags)

    mag_centers = np.arange(minmag, maxmag + 2*magbin, magbin)
    cdf = np.zeros(len(mag_centers))

    for ii in range(len(cdf)):
        cdf[ii] = np.count_nonzero(~np.isnan(mags[mags >= mag_centers[ii]]))

    mag_edges = np.arange(minmag - magbin/2., maxmag+magbin, magbin)
    g_r, _ = np.histogram(mags, mag_edges)
    ii = list(g_r).index(max(g_r))

    Mc_est = mag_centers[ii]
    #Mc_est = 5.2

    try:
        Mc_est, bvalue, avalue, L, Mc_bins, std_dev = WW2000(Mc_est, mags,
                                                             magbin)
    except:
        Mc_est = Mc_est + 0.3
        Mc_bins = np.arange(0, maxmag + magbin/2., magbin)
        bvalue = np.log10(np.exp(1))/(np.average(mags[mags >= Mc_est])
                          - (Mc_est-magbin/2.))
        avalue = np.log10(len(mags[mags >= Mc_est])) + bvalue*Mc_est
        log_L = avalue-bvalue*Mc_bins
        L = 10.**log_L
        std_dev = bvalue/sqrt(len(mags[mags >= Mc_est]))


    maxincremcomp = mag_centers[ii]
    pm = u'\u00B1'

    plt.figure(figsize=(8,6))
    plt.scatter(mag_centers[:-1], g_r, edgecolor='r', marker='o',
                facecolor='none', label='Incremental')
    plt.scatter(mag_centers, cdf, c='k', marker='+', label='Cumulative')
    plt.axvline(Mc_est, c='r', linestyle='--', label='Mc = %2.1f' % Mc_est)
    plt.plot(Mc_bins, L, c='k', linestyle='--', label='B = %1.3f%s%1.3f'
                                                % (bvalue, pm, std_dev))

    ax = plt.gca()
    ax.set_yscale('log')
    max_count = np.amax(cdf) + 100000
    ax.set_xlim([0, maxmag])
    ax.set_ylim([1, max_count])
    plt.title('Frequency-Magnitude Distribution', fontsize=18)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Log10 Count', fontsize=14)
    plt.legend(numpoints=1)

    plt.savefig('%s_catmagcomp.png' % dirname, dpi=300)


# plot magnitudes vs. origin time
@printstatus('Graphing magnitude versus time for each earthquake')
def graphMagTime(catalog, dirname):

    catalog = catalog[pd.notnull(catalog['mag']) & (catalog['mag'] > 0)]
    catalog['convtime'] = [' '.join(x.split('T')).split('.')[0]
                           for x in catalog['time'].tolist()]
    catalog['convtime'] = catalog['convtime'].astype('datetime64[ns]')

    times = catalog['time']
    mags = catalog['mag']

    plt.figure(figsize=(10,6))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.plot_date(times, mags, alpha=0.7, markersize=2, c='b')
    plt.xlim(min(times), max(times))
    plt.ylim(0)

    plt.savefig('%s_magvtime.png' % dirname, dpi=300)


# graph cumulative moment release
@printstatus('Graphing cumulative moment release')
def cumulMomentRelease(catalog, dirname):

    catalog = catalog[pd.notnull(catalog['mag']) & (catalog['mag'] > 0)]
    catalog['convtime'] = [' '.join(x.split('T')).split('.')[0]
                           for x in catalog['time'].tolist()]
    catalog['convtime'] = catalog['convtime'].astype('datetime64[ns]')
    times = catalog['convtime']

    minday, maxday = min(times), max(times)

    M0 = 10.**((3/2.)*(catalog['mag']+10.7))
    M0 = M0 * 10.**(-7)
    cumulM0 = np.cumsum(M0)

    plt.figure(figsize=(10,6))
    plt.plot(times, cumulM0, 'k-')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(r'Cumulative Moment Release (N$\times$m)', fontsize=14)
    plt.xlim(minday, maxday)
    plt.ylim(0)

    colors = ['r', 'm', 'c', 'y', 'g']
    largestnum = 5
    largesteqs = catalog.sort_values('mag').tail(5)
    for i, eq in enumerate(largesteqs.itertuples()):
        plt.axvline(x=eq.time, color=colors[i], linestyle='--')

    plt.savefig('%s_cumulmomentrelease.png' % dirname, dpi=300)


# graph possible number of duplicate events given various distances
# and time differences
@printstatus('Graphing possible number of duplicate events')
def catDupSearch(catalog, dirname):
    
    nquakes = len(catalog)
    epochtimes = [toEpoch(row.time) for row in catalog.itertuples()]
    tdifsec = np.asarray(abs(np.diff(epochtimes)))

    lat1 = np.asarray(catalog.latitude[:-1]) 
    lon1 = np.asarray(catalog.longitude[:-1])
    lat2 = np.asarray(catalog.latitude[1:])
    lon2 = np.asarray(catalog.longitude[1:])
    ddelkm = [eqdist(lat1[i], lon1[i], lat2[i], lon2[i])
              for i in range(len(lat1))]

    df = pd.DataFrame({'tdifsec': tdifsec, 'ddelkm': ddelkm})

    kmlimits = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tmax = 16
    dt = 0.05
    timebins = np.arange(0, tmax+dt/2, dt)

    nn = np.empty([len(kmlimits), len(timebins)-1])

    for jj in range(len(kmlimits)):

        cat_subset = df[df.ddelkm <= kmlimits[jj]]

        for ii in range(len(timebins)-1):

            nn[jj][ii] = cat_subset[cat_subset.tdifsec.between(timebins[ii],
                                    timebins[ii+1])].count()[0]

    totmatch = np.transpose(np.cumsum(np.transpose(nn),axis=0))

    plt.figure(figsize=(10,6))
    for ii in range(len(kmlimits)):
        x = timebins[1:]
        y = totmatch[ii]
        lab = str(kmlimits[ii]) + ' km'
        plt.plot(x,y,label=lab)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Total Number of Events', fontsize=14)
    plt.xlim(0, tmax)
    plt.ylim(0, np.amax(totmatch)+0.5)
    plt.legend(loc=2, numpoints=1)

    plt.savefig('%s_catdupsearch.png' % dirname, dpi=300)


################################################################################
################################################################################
################################################################################


#@suppresskbinterrupt()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('catalog', nargs='?', type=str, default='',
                        help='pick which catalog to download data from')
    parser.add_argument('startyear', nargs='?', type=int, default=2000,
                        help='pick starting year')
    parser.add_argument('endyear', nargs='?', type=int, default=2000,
                        help='pick end year (to get a single year of data, \
                        enter same year as startyear)')

    parser.add_argument('-sf', '--specifyfile', type=str,
                        help='specify existing .csv file to use')
    parser.add_argument('-fd', '--forcedownload', action='store_true',
                        help='forces downloading of data even if .csv file\
                        exists')

    args = parser.parse_args()

    if args.specifyfile is None:

        catalog = args.catalog.lower()
        startyear, endyear = map(int, [args.startyear, args.endyear])
        download = args.forcedownload

        dirname = '%s%s-%s' % (catalog, startyear, endyear) if catalog else\
                  'all%s-%s' % (startyear, endyear)

        if download:
            try:
                os.makedirs(dirname)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
            datadf = getData(catalog, startyear, endyear, write=True)
        else:
            # Python 2
            try:
                try:
                    datadf = pd.read_csv('%s/%s.csv' % (dirname, dirname))
                except IOError:
                    try:
                        os.makedirs(dirname)
                    except OSError as exception:
                        if exception.errno != errno.EEXIST:
                            raise
                    datadf = getData(catalog, startyear, endyear, write=True)
            # Python 3
            except:
                try:
                    datadf = pd.read_csv('%s/%s.csv' % (dirname, dirname))
                except FileNotFoundError:
                    try:
                        os.makedirs(dirname)
                    except OSError as exception:
                        if exception.errno != errno.EEXIST:
                            raise
                    datadf = getData(catalog, startyear, endyear, write=True)

    else:
        from shutil import copy2
        dirname = '.'.join(args.specifyfile.split('.')[:-1])

        try:
            os.makedirs(dirname)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        datadf = pd.read_csv(args.specifyfile)
        copy2(args.specifyfile, dirname)

    datadf = datadf.sort_values(by='time').reset_index(drop=True)

    os.chdir(dirname)
    basicCatSum(datadf, dirname)
    mapDetecs(datadf, dirname)
    plt.close()
    mapDetecNums(datadf, dirname)
    plt.close()
    makeHist(datadf, 'mag', 0.1, dirname, xlabel='Magnitude')
    plt.close()
    makeHist(datadf, 'depth', 10, dirname, xlabel='Depth (km)')
    plt.close()
    makeTimeHist(datadf, 'hour', dirname)
    plt.close()
    makeTimeHist(datadf, 'day', dirname)
    plt.close()
    graphTimeSep(datadf, dirname)
    plt.close()
    catMagComp(datadf, dirname)
    plt.close()
    graphMagTime(datadf, dirname)
    plt.close()
    cumulMomentRelease(datadf, dirname)
    plt.close()
    catDupSearch(datadf, dirname)


if __name__ == '__main__':

    try:
        main()
    except (KeyboardInterrupt, SystemError):
        sys.stdout.write('\nProgram canceled. Exiting...\n')
        sys.exit()

