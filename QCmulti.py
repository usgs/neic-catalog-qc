#!/usr/bin/env python
"""Code for creating figures comparing two catalogs spanning the same time
frame. Run `QCmulti.py -h` for command line options.
"""
import os
import sys
import errno
import argparse
from datetime import datetime
from math import sqrt, degrees, radians, sin, cos, atan2, pi

import scipy.io
from scipy import stats
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from decorators import retry, printstatus

# Python 2
try:
    from urllib2 import urlopen, HTTPError
# Python 3
except ImportError:
    from urllib.request import urlopen, HTTPError


###############################################################################
############################## Auxiliary Functions ############################
###############################################################################


def progress_bar(count, total, status=''):
    """Show progress bar for whatever task."""
    barlen = 50
    filledlen = int(round(barlen * count / float(total)))

    percents = round(100. * count / float(total), 1)
    pbar = '#' * filledlen + '-' * (barlen-filledlen)

    if percents < 100.0:
        sys.stdout.write('[%s] %s%s  %s\r' % (pbar, percents, '%', status))
        sys.stdout.flush()
    else:
        sys.stdout.write('[%s] %s%s %s\n' % (pbar, percents, '%',
                                             'Done downloading data' +
                                             ' '*(len(status)-21)))
        sys.stdout.flush()


@retry(HTTPError, tries=5, delay=0.5, backoff=1)
def access_url(url):
    """Return list of lines in url; if HTTPError, repeat."""
    return [line.rstrip() for line in urlopen(url).readlines()]


def format_time(ogtime):
    """Increase readability of given catalog times."""
    return pd.Timestamp(' '.join(ogtime.split('T'))[:-1])


def to_epoch(ogtime):
    """Convert formatted time to Unix/epoch time."""
    fstr = '%Y-%m-%dT%H:%M:%S.%fZ'
    epoch = datetime(1970, 1, 1)
    finaltime = (datetime.strptime(ogtime, fstr) - epoch).total_seconds()

    return finaltime


def trim_times(cat1, cat2, otwindow):
    """Trim catalogs so they span the same time window."""
    mintime = max(cat1['time'].min(), cat2['time'].min())
    maxtime = min(cat1['time'].max(), cat2['time'].max())
    adjmin = mintime - otwindow
    adjmax = maxtime + otwindow

    cat1trim = cat1[cat1['time'].between(adjmin, adjmax, inclusive=True)
                   ].copy()
    cat2trim = cat2[cat2['time'].between(adjmin, adjmax, inclusive=True)
                   ].copy()
    cat1trim = cat1trim.reset_index(drop=True)
    cat2trim = cat2trim.reset_index(drop=True)

    return cat1trim, cat2trim


def eq_dist(lon1, lat1, lon2, lat2):
    """Calculate equirectangular distance between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    xval = (lon2 - lon1) * cos(0.5 * (lat2+lat1))
    yval = lat2 - lat1
    eqd = 6371 * sqrt(xval*xval + yval*yval)

    return eqd


def get_az(lon1, lat1, lon2, lat2):
    """Calculate azimuth from point 1 to point 2."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    londiff = lon2 - lon1
    xval = sin(londiff) * cos(lat2)
    yval = cos(lat1)*sin(lat2) - (sin(lat1)*cos(lat2)*cos(londiff))
    azi = (degrees(atan2(xval, yval)) + 360) % 360

    return azi


def get_azs_and_dists(cat1, cat2, cat1mids, cat2mids):
    """Calculate azimuths for all matches between two catalogs."""
    azimuths = []
    dists = []

    for idx, eid in enumerate(cat1mids):
        mask1 = cat1['id'] == eid
        mask2 = cat2['id'] == cat2mids[idx]

        lon1, lat1 = cat1[mask1].longitude, cat1[mask1].latitude
        lon2, lat2 = cat2[mask2].longitude, cat2[mask2].latitude

        azi = get_az(lon1, lat1, lon2, lat2)
        dist = eq_dist(lon1, lat1, lon2, lat2)
        azimuths.append(azi)
        dists.append(dist)

    return azimuths, dists


def cond(row1, row2, otwindow, distwindow):
    """Condition for event matching."""
    lon1, lat1 = row1.longitude, row1.latitude
    lon2, lat2 = row2.longitude, row2.latitude

    return (eq_dist(lon1, lat1, lon2, lat2) < distwindow) \
        & (abs(row1.time - row2.time) < otwindow)


def get_map_bounds(cat1, cat2):
    """Generate map bounds and gridsize."""
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


def round2bin(number, binsize, direction):
    """Round number to nearest histogram bin edge (either "up" or "down")."""
    if direction == 'down':
        return number - (number % binsize)
    if direction == 'up':
        return number - (number % binsize) + binsize


###############################################################################
################################# Main Functions ##############################
###############################################################################


def get_data(catalog1, catalog2, startyear, endyear, minmag=0.1, maxmag=10,
             write=False):
    """Download catalog data from earthquake.usgs.gov"""
    catalog1, catalog2 = catalog1.lower(), catalog2.lower()
    year = startyear
    alldata1, alldata2 = [], []
    cat1name = catalog1 if catalog1 else 'all'
    cat2name = catalog2 if catalog2 else 'all'

    if cat1name == cat2name:
        print('Catalogs cannot be the same. Exiting...')
        sys.exit()

    dirname = '%s-%s%s-%s' % (cat1name, cat2name, startyear, endyear)
    f1name = '%s%s-%s.csv' % (cat1name, startyear, endyear)
    f2name = '%s%s-%s.csv' % (cat2name, startyear, endyear)
    bartotal = 12 * (endyear - startyear + 1)
    barcount = 1

    while year <= endyear:

        month = 1
        yeardata1, yeardata2 = [], []

        while month <= 12:

            if month in [4, 6, 9, 11]:
                endday = 30

            elif month == 2:
                checkly = (year % 4)

                if checkly == 0:
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
                        '&minmagnitude=' + str(minmag) + '&maxmagnitude=' +
                        str(maxmag))
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
                        '&minmagnitude=' + str(minmag) + '&maxmagnitude=' +
                        str(maxmag))
            else:
                url2 = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                        '.csv?starttime=' + startd + '-1%2000:00:00'
                        '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                        '&catalog=' + catalog2 + '&minmagnitude=' +
                        str(minmag) + '&maxmagnitude=' + str(maxmag))

            monthdata1 = access_url(url1)
            monthdata2 = access_url(url2)

            if (month != 1) or (year != startyear):
                del monthdata1[0]
                del monthdata2[0]

            yeardata1.append(monthdata1)
            yeardata2.append(monthdata2)

            progress_bar(barcount, bartotal, 'Downloading data ... ')
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

    return alldatadf1, alldatadf2


@printstatus('Creating basic catalog summary')
def basic_cat_sum(catalog, catname, dirname):
    """Gather basic catalog summary statistics."""
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


@printstatus('Creating summary of comparison criteria')
def comp_criteria(cat1, cat2, dirname, reg, otwindow=16, distwindow=100):
    """Trim catalogs and summarize comparison criteria."""
    lines = []

    mintime = format_time(max(cat1['time'].min(), cat2['time'].min()))
    maxtime = format_time(min(cat1['time'].max(), cat2['time'].max()))

    lines.append('Overlapping time period: %s to %s\n' % (mintime, maxtime))
    lines.append('Region: %s\n' % reg)
    lines.append('Authoritative agency: %s\n' % reg)

    lines.append('Matching Criteria\n')
    lines.append('Time window: %s s\n' % otwindow)
    lines.append('Distance window: %s km\n\n' % distwindow)

    with open('%s_comparisoncriteria.txt' % dirname, 'w') as compfile:
        for line in lines:
            compfile.write(line)


@printstatus(status='Matching events')
def match_events(cat1, cat2, otwindow=16, distwindow=100):
    """Match events within two catalogs."""
    cat1.loc[:, 'time'] = [to_epoch(x) for x in cat1['time']]
    cat2.loc[:, 'time'] = [to_epoch(x) for x in cat2['time']]

    cat1, cat2 = trim_times(cat1, cat2, otwindow)

    cat1ids, cat2ids = [], []

    for i in range(len(cat1)):

        cat2ix = cat2[abs(cat2['time']
                          - cat1.ix[i]['time']) <= otwindow].index.values

        if len(cat2ix) != 0:

            carr = np.array([eq_dist(cat1.ix[i]['longitude'],
                                     cat1.ix[i]['latitude'],
                                     cat2.ix[x]['longitude'],
                                     cat2.ix[x]['latitude'])
                             for x in cat2ix])

            inds = np.where(carr < distwindow)[0]

            if len(inds) != 0:
                for ind in inds:
                    cat1id = cat1.ix[i]['id']
                    cat2id = cat2.ix[cat2ix[ind]]['id']

                    cat1ids.append(cat1id)
                    cat2ids.append(cat2id)

    cat1matched = cat1[cat1['id'].isin(cat1ids)].reset_index(drop=True)
    cat2matched = cat2[cat2['id'].isin(cat2ids)].reset_index(drop=True)

    return cat1ids, cat2ids, cat1matched, cat2matched


@printstatus('Mapping events from both catalogs')
def map_events(cat1, cat2, reg, dirname):
    """Map catalog events only within the appropriate region."""
    lllat, lllon, urlat, urlon, _, _, _, clon = get_map_bounds(cat1, cat2)

    regionmat = scipy.io.loadmat('../regions.mat')
    regdict = {regionmat['region'][i][0][0]: x[0]
               for i, x in enumerate(regionmat['coord'])}
    regzone = regdict[reg]
    reglons, reglats = [x[0] for x in regzone], [x[1] for x in regzone]

    cat1lons, cat1lats = cat1.longitude, cat1.latitude
    cat2lons, cat2lats = cat2.longitude, cat2.latitude

    plt.figure(figsize=(12, 7))
    mplmap = plt.axes(projection=ccrs.Robinson(central_longitude=clon))
    mplmap.set_extent([lllon, urlon, lllat, urlat], ccrs.PlateCarree())
    mplmap.coastlines('50m')

    mplmap.scatter(cat1lons, cat1lats, color='b', s=2, zorder=4,
                   transform=ccrs.PlateCarree())
    mplmap.scatter(cat2lons, cat2lats, color='r', s=2, zorder=4,
                   transform=ccrs.PlateCarree())
    mplmap.plot(reglons, reglats, c='k', linestyle='--', zorder=5,
                transform=ccrs.PlateCarree())

    plt.savefig('%s_mapmatcheddetecs.png' % dirname, dpi=300)


@printstatus('Graphing polar histogram of azimuths and distances')
def make_az_dist(cat1, cat2, cat1mids, cat2mids, dirname,
                 distwindow=100, numbins=16):
    """Make polar scatter/histogram of azimuth vs. distance."""
    azimuths, distances = get_azs_and_dists(cat1, cat2, cat1mids, cat2mids)

    width = 2*pi / numbins
    razimuths = list(map(radians, azimuths))
    bins = np.linspace(0, 2*pi, numbins+1)
    azhist = np.histogram(razimuths, bins=bins)[0]
    hist = (float(distwindow)/max(azhist)) * azhist
    bins = (bins + width/2)[:-1]

    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111, projection='polar')
    ax1.scatter(razimuths, distances, color='b', s=10)
    bars = ax1.bar(bins, hist, width=width)
    ax1.set_theta_zero_location('N')
    ax1.set_rmax(distwindow)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(112.5)

    for _, hbar in list(zip(hist, bars)):
        hbar.set_facecolor('b')
        hbar.set_alpha(0.5)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.11)

    plt.savefig('%s_polarazimuth.png' % dirname, dpi=300)


@printstatus('Comparing parameters of matched events')
def compare_params(cat1, cat2, cat1mids, cat2mids, param, dirname):
    """Compare parameters of matched events."""
    cat1params, cat2params = [], []

    for idx, eid in enumerate(cat1mids):
        param1 = float(cat1[cat1['id'] == eid][param])
        param2 = float(cat2[cat2['id'] == cat2mids[idx]][param])

        cat1params.append(param1)
        cat2params.append(param2)

    mval, bval, rval, _, _ = stats.linregress(cat1params, cat2params)
    linegraph = [mval*x + bval for x in cat1params]
    r2val = rval*rval

    plt.scatter(cat1params, cat2params, edgecolor='b', facecolor=None)
    plt.plot(cat1params, linegraph, c='r', linewidth=1,
             label=r'$\mathregular{R^2}$ = %0.2f' % r2val)
    plt.plot(cat1params, cat1params, c='k', linewidth=1, label='B = 1')
    plt.legend(loc='upper left')

    plt.savefig('%s_compare%s.png' % (dirname, param), dpi=300)


@printstatus('Graphing parameter differences between matched events')
def make_diff_hist(cat1, cat2, cat1mids, cat2mids, param, binsize, dirname,
                   title='', xlabel=''):
    """Make histogram of parameter differences between matched detections."""
    paramdiffs = []

    for idx, eid in enumerate(cat1mids):
        c1mask = cat1['id'] == eid
        c2mask = cat2['id'] == cat2mids[idx]
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

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Count', fontsize=14)

    plt.hist(paramdiffs, pardiffbins, alpha=0.7, color='b', edgecolor='k')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.11)
    plt.tick_params(labelsize=12)
    plt.xlim(pardiffdown+binsize/2., pardiffup+binsize/2.)
    plt.ylim(0)

    plt.savefig('%s_%sdiffs.png' % (dirname, param), dpi=300)


###############################################################################
###############################################################################
###############################################################################


def main():
    """Main function. Command line arguments defined here."""
    parser = argparse.ArgumentParser()

    parser.add_argument('catalog1', nargs='?', type=str,
                        help='pick first catalog to download data from')
    parser.add_argument('catalog2', nargs='?', type=str,
                        help='pick second catalog to download data from')
    parser.add_argument('startyear', nargs='?', type=int,
                        help='pick starting year')
    parser.add_argument('endyear', nargs='?', type=int,
                        help='pick end year (to get a single year of data, \
                        enter same year as startyear)')

    parser.add_argument('-sf', '--specifyfiles', nargs=2, type=str,
                        help='specify two existing .csv files to use')
    parser.add_argument('-fd', '--forcedownload', action='store_true',
                        help='forces downloading of data even if .csv file \
                        exists')

    args = parser.parse_args()

    if args.specifyfiles is None:

        if not args.catalog1:
            sys.stdout.write('No first catalog specified. Exiting...\n')
            sys.exit()
        elif not args.catalog2:
            sys.stdout.write('No second catalog specified. Exiting...\n')
            sys.exit()
        elif not args.startyear:
            sys.stdout.write('No starting year specified. Exiting...\n')
            sys.exit()
        elif not args.endyear:
            sys.stdout.write('No ending year specified. Exiting...\n')
            sys.exit()

        cat1, cat2 = args.catalog1.lower(), args.catalog2.lower()
        startyear, endyear = map(int, [args.startyear, args.endyear])
        download = args.forcedownload

        dirname = '%s-%s%s-%s' % (cat1, cat2, startyear, endyear)

        if download:
            try:
                os.makedirs(dirname)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
            datadf1, datadf2 = get_data(cat1, cat2, startyear, endyear,
                                        write=True)
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
                    datadf1, datadf2 = get_data(cat1, cat2, startyear, endyear,
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
                    datadf1, datadf2 = get_data(cat1, cat2, startyear, endyear,
                                                write=True)

    else:
        from shutil import copy2
        sfcat1, sfcat2 = args.specifyfiles
        cat1, cat2 = sfcat1.split('/')[-1][:-13], sfcat2.split('/')[-1][:-13]
        dirname = '_'.join(['.'.join(sfcat1.split('.')[:-1]).split('/')[-1],
                            '.'.join(sfcat2.split('.')[:-1]).split('/')[-1]])

        try:
            os.makedirs(dirname)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        datadf1, datadf2 = pd.read_csv(sfcat1), pd.read_csv(sfcat2)
        copy2(sfcat1, dirname)
        copy2(sfcat2, dirname)

    if len(datadf1) == 0:
        sys.stdout.write(('%s catalog has no data available for that time '
                          'period. Quitting...') % cat1.upper())
        sys.exit()

    if len(datadf2) == 0:
        sys.stdout.write(('%s catalog has no data available for that time '
                          'period. Quitting...') % cat2.upper())
        sys.exit()

    os.chdir(dirname)
    basic_cat_sum(datadf1, cat1, dirname)
    basic_cat_sum(datadf2, cat2, dirname)
    comp_criteria(datadf1, datadf2, dirname, cat1.upper())
    cat1ids, cat2ids, newcat1, newcat2 = match_events(datadf1, datadf2)
    map_events(newcat1, newcat2, cat1.upper(), dirname)
    plt.close()
    make_az_dist(newcat1, newcat2, cat1ids, cat2ids, dirname)
    plt.close()
    compare_params(newcat1, newcat2, cat1ids, cat2ids, 'mag', dirname)
    plt.close()
    compare_params(newcat1, newcat2, cat1ids, cat2ids, 'depth', dirname)
    plt.close()
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'time', 0.5, dirname,
                   xlabel='Time residuals (sec)')
    plt.close()
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'mag', 0.1, dirname,
                   xlabel='Magnitude residuals')
    plt.close()
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'depth', 2, dirname,
                   xlabel='Depth residuals (km)')


if __name__ == '__main__':

    try:
        main()
    except (KeyboardInterrupt, SystemError):
        sys.stdout.write('\nProgram canceled. Exiting...\n')
        sys.exit()
