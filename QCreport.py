#!/usr/bin/env python
"""Script for generating figures of catalog statistics. Run `QCreport.py -h`
for command line usage.
"""
import os
import sys
import errno
import argparse
from datetime import date, datetime
from math import sqrt, radians, cos

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
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


def get_map_bounds(catalog):
    """Generate map bounds and gridsize."""
    minlat, maxlat = catalog['latitude'].min(), catalog['latitude'].max()
    minlon, maxlon = catalog['longitude'].min(), catalog['longitude'].max()
    latdiff, londiff = (maxlat-minlat) / 5., (maxlon-minlon) / 5.
    lllat, lllon = max(minlat-latdiff, -90), max(minlon-londiff, -180)
    urlat, urlon = min(maxlat+latdiff, 90), min(maxlon+londiff, 180)

    if (lllon < 175) and (urlon > 175) \
            and (len(catalog[catalog['longitude'].between(-100, 100)]) == 0):

        lllon = catalog[catalog['longitude'] > 0].min()['longitude']
        urlon = 360 + catalog[catalog['longitude'] < 0].max()['longitude']
        clon = 180

    else:
        clon = 0

    gridsize = max(urlat-lllat, urlon-lllon) / 45.
    hgridsize, tgridsize = gridsize / 2., gridsize / 10.

    return lllat, lllon, urlat, urlon, gridsize, hgridsize, tgridsize, clon


def round2center(num, gridsize):
    """Round number to nearest grid-square center."""
    hgridsize = gridsize / 2

    return num - (num % gridsize) + hgridsize


def round2lon(num):
    """Round number to nearest timezone longitude."""
    return 15 * round(num / 15.)


def add_centers(catalog, gridsize):
    """Add corresponding centers to catalog."""
    zippedlatlon = list(zip(round2center(catalog['latitude'], gridsize),
                            round2center(catalog['longitude'], gridsize)))
    catalog = catalog.reset_index()
    catalog.loc[:, 'center'] = pd.Series(zippedlatlon)
    catalog = catalog.set_index('index')
    catalog.index.names = ['']

    return catalog


def group_lat_lons(catalog, minmag=0):
    """Group detections by nearest grid-square center, and return min/max
    of counts.
    """
    if not catalog['mag'].isnull().all():
        magmask = catalog['mag'] >= minmag
        groupedlatlons = catalog[magmask].groupby('center')
        groupedlatlons = groupedlatlons.count().sort_index()
    elif catalog['mag'].isnull().all() and (minmag != 0):
        groupedlatlons = catalog.groupby('center').count().sort_index()
        print("No magnitude data in catalog - plotting all events")
    else:
        groupedlatlons = catalog.groupby('center').count().sort_index()
    groupedlatlons = groupedlatlons[['id']]
    groupedlatlons.columns = ['count']
    cmin = min(list(groupedlatlons['count']) or [0])
    cmax = max(list(groupedlatlons['count']) or [0])

    return groupedlatlons, cmin, cmax


def range2rgb(rmin, rmax, numcolors):
    """Create a list of red RGB values using colmin and colmax with numcolors
    number of colors.
    """
    colors = np.linspace(rmax/255., rmin/255., numcolors)
    colors = [(min(1, x), max(0, x-1), max(0, x-1)) for x in colors]

    return colors


def draw_grid(lats, lons, col, alpha=1):
    """Draw rectangle with vertices given in degrees."""
    latlons = list(zip(lons, lats))
    poly = Polygon(latlons, facecolor=col, alpha=alpha, edgecolor='k',
                   zorder=11, transform=ccrs.PlateCarree())
    plt.gca().add_patch(poly)


def round2bin(number, binsize, direction):
    """Round number to nearest histogram bin edge (either "up" or "down")."""
    if direction == 'down':
        return number - (number % binsize)
    if direction == 'up':
        return number - (number % binsize) + binsize


def WW2000(mcval, mags, binsize):
    """Wiemer and Wyss (2000) method for determining a and b values."""
    mags = mags[~np.isnan(mags)]
    mags = np.around(mags, 1)
    mc_vec = np.arange(mcval-1.5, mcval+1.5+binsize/2., binsize)
    max_mag = max(mags)
    corr = binsize / 2.
    bvalue = np.zeros(len(mc_vec))
    std_dev = np.zeros(len(mc_vec))
    avalue = np.zeros(len(mc_vec))
    rval = np.zeros(len(mc_vec))

    for idx in range(len(mc_vec)):
        mval = mags[mags >= mc_vec[idx]-0.001]
        mag_bins_edges = np.arange(mc_vec[idx]-binsize/2., max_mag+binsize,
                                   binsize)
        mag_bins_centers = np.arange(mc_vec[idx], max_mag+binsize/2., binsize)

        cdf = np.zeros(len(mag_bins_centers))

        for jdx in range(len(cdf)):
            cdf[jdx] = np.count_nonzero(~np.isnan(mags[
                mags >= mag_bins_centers[jdx]-0.001]))

        bvalue[idx] = np.log10(np.exp(1))/(np.average(mval)
                                           - (mc_vec[idx]-corr))
        std_dev[idx] = bvalue[idx]/sqrt(cdf[0])

        avalue[idx] = np.log10(len(mval)) + bvalue[idx]*mc_vec[idx]
        log_l = avalue[idx] - bvalue[idx]*mag_bins_centers
        lval = 10.**log_l

        bval, _ = np.histogram(mval, mag_bins_edges)
        sval = abs(np.diff(lval))
        rval[idx] = (sum(abs(bval[:-1] - sval))/len(mval))*100

    ind = np.where(rval <= 10)[0]

    if len(ind) != 0:
        idx = ind[0]
    else:
        idx = list(rval).index(min(rval))

    mcval = mc_vec[idx]
    bvalue = bvalue[idx]
    avalue = avalue[idx]
    std_dev = std_dev[idx]
    mag_bins = np.arange(0, max_mag+binsize/2., binsize)
    lval = 10.**(avalue-bvalue*mag_bins)

    return mcval, bvalue, avalue, lval, mag_bins, std_dev


def to_epoch(ogtime):
    """Convert from ComCat time format to Unix/epoch time."""
    fstr = '%Y-%m-%dT%H:%M:%S.%fZ'
    epoch = datetime(1970, 1, 1)
    epochtime = (datetime.strptime(ogtime, fstr) - epoch).total_seconds()

    return epochtime


def eq_dist(lat1, lon1, lat2, lon2):
    """Calculate equirectangular distance between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    xval = (lon2 - lon1) * cos(0.5 * (lat2+lat1))
    yval = lat2 - lat1
    eqd = 6371 * sqrt(xval*xval + yval*yval)

    return eqd


###############################################################################
################################# Main Functions ##############################
###############################################################################


def get_data(catalog, startyear, endyear, minmag=0.1, maxmag=10, system=0,
             write=False):
    """Download catalog data from earthquake.usgs.gov"""
    systems = ['prod01', 'prod02', 'dev', 'dev01', 'dev02']
    syval = systems[system]
    catalog = catalog.lower()
    year = startyear
    alldata = []
    catname = catalog if catalog else 'all'
    if catname == 'all':
        catalog = ''
    dirname = '%s%s-%s' % (catname, startyear, endyear)
    fname = '%s.csv' % dirname
    bartotal = 12 * (endyear - startyear + 1)
    barcount = 1

    while year <= endyear:

        month = 1
        yeardata = []

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

            if not catalog:
                if system == 0:
                    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                           '.csv?starttime=' + startd + '-1%2000:00:00'
                           '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                           '&minmagnitude=' + str(minmag) + '&maxmagnitude=' +
                           str(maxmag))
                else:
                    url = ('https://' + syval + '-earthquake.cr.usgs.gov/'
                           'fdsnws/event/1/query.csv?starttime=' + startd +
                           '-1%2000:00:00&endtime=' + endd + '%2023:59:59'
                           '&orderby=time-asc&minmagnitude=' + str(minmag) +
                           '&maxmagnitude=' + str(maxmag))
            else:
                if system == 0:
                    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query'
                           '.csv?starttime=' + startd + '-1%2000:00:00'
                           '&endtime=' + endd + '%2023:59:59&orderby=time-asc'
                           '&catalog=' + catalog + '&minmagnitude=' +
                           str(minmag) + '&maxmagnitude=' + str(maxmag))
                else:
                    url = ('https://' + syval + '-earthquake.cr.usgs.gov/'
                           'fdsnws/event/1/query.csv?starttime=' + startd +
                           '-1%2000:00:00&endtime=' + endd + '%2023:59:59'
                           '&orderby=&catalog=' + catalog + 'time-asc'
                           '&minmagnitude=' + str(minmag) + '&maxmagnitude=' +
                           str(maxmag))

            monthdata = access_url(url)

            if (month != 1) or (year != startyear):
                del monthdata[0]

            yeardata.append(monthdata)

            progress_bar(barcount, bartotal, 'Downloading data ...')
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

    return alldatadf


@printstatus('Creating basic catalog summary')
def basic_cat_sum(catalog, dirname):
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

    with open('%s_catalogsummary.txt' % dirname, 'w') as sumfile:
        for line in lines:
            sumfile.write(line)


@printstatus('Mapping earthquake locations')
def map_detecs(catalog, dirname, minmag=0, mindep=-50, title=''):
    """Make scatter plot of detections with magnitudes (if applicable)."""
    catalog = catalog[(catalog['mag'] >= minmag)
                      & (catalog['depth'] >= mindep)].copy()

    # define map bounds
    lllat, lllon, urlat, urlon, _, _, _, clon = get_map_bounds(catalog)

    plt.figure(figsize=(12, 7))
    mplmap = plt.axes(projection=ccrs.PlateCarree(central_longitude=clon))
    mplmap.set_extent([lllon, urlon, lllat, urlat], ccrs.PlateCarree())
    mplmap.coastlines('50m')

    # if catalog has magnitude data
    if not catalog['mag'].isnull().all():
        bins = [0, 5, 6, 7, 8, 15]
        binnames = ['< 5', '5-6', '6-7', '7-8', r'$\geq$8']
        binsizes = [10, 25, 50, 100, 400]
        bincolors = ['g', 'b', 'y', 'r', 'r']
        binmarks = ['o', 'o', 'o', 'o', '*']
        catalog.loc[:, 'maggroup'] = pd.cut(catalog['mag'], bins,
                                            labels=binnames)

        for i, label in enumerate(binnames):
            mgmask = catalog['maggroup'] == label
            rcat = catalog[mgmask]
            lons, lats = list(rcat['longitude']), list(rcat['latitude'])
            if len(lons) > 0:
                mplmap.scatter(lons, lats, s=binsizes[i], marker=binmarks[i],
                               c=bincolors[i], label=binnames[i], alpha=0.8,
                               zorder=10, transform=ccrs.PlateCarree())

        plt.legend(loc='lower left')

    # if catalog does not have magnitude data
    else:
        lons, lats = list(catalog['longitude']), list(catalog['latitude'])
        mplmap.scatter(lons, lats, s=15, marker='x', c='r', zorder=10)

    plt.title(title)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.savefig('%s_mapdetecs.png' % dirname, dpi=300)


@printstatus('Mapping earthquake density')
def map_detec_nums(catalog, dirname, title='', numcolors=16, rmin=77, rmax=490,
                   minmag=0, pltevents=True):
    """Map detections and a grid of detection density. rmax=510 is white,
    rmin=0 is black.
    """
    # generate bounds for map
    mask = catalog['mag'] >= minmag

    lllat, lllon, urlat, urlon, gridsize, hgridsize, _, clon = \
        get_map_bounds(catalog[mask])

    catalog = add_centers(catalog, gridsize)
    groupedlatlons, _, cmax = group_lat_lons(catalog, minmag=minmag)

    # print message if there are no detections with magnitudes above minmag
    if cmax == 0:
        print("No detections over magnitude %s" % minmag)

    # create color gradient from light red to dark red
    colors = range2rgb(rmin, rmax, numcolors)

    # put each center into its corresponding color group
    colorgroups = list(np.linspace(0, cmax, numcolors))
    groupedlatlons.loc[:, 'group'] = np.digitize(groupedlatlons['count'],
                                                 colorgroups)

    # create map
    plt.figure(figsize=(12, 7))
    mplmap = plt.axes(projection=ccrs.PlateCarree(central_longitude=clon))
    mplmap.set_extent([lllon, urlon, lllat, urlat], ccrs.PlateCarree())
    mplmap.coastlines('50m')
    plt.title(title, fontsize=20)
    plt.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.05)

    # create color map based on rmin and rmax
    cmap = LinearSegmentedColormap.from_list('CM', colors)._resample(numcolors)

    # make dummy plot for setting color bar
    colormesh = mplmap.pcolormesh(colors, colors, colors, cmap=cmap, alpha=1,
                                  vmin=0, vmax=cmax)

    # format color bar
    cbticks = [x for x in np.linspace(0, cmax, numcolors+1)]
    cbar = plt.colorbar(colormesh, ticks=cbticks)
    cbar.ax.set_yticklabels([('%.0f' % x) for x in cbticks])
    cbar.set_label('# of detections', rotation=270, labelpad=15)

    # plot rectangles with color corresponding to number of detections
    for center, _, cgroup in groupedlatlons.itertuples():
        minlat, maxlat = center[0]-hgridsize, center[0]+hgridsize
        minlon, maxlon = center[1]-hgridsize, center[1]+hgridsize
        glats = [minlat, maxlat, maxlat, minlat]
        glons = [minlon, minlon, maxlon, maxlon]

        color = colors[cgroup-1]

        draw_grid(glats, glons, color, alpha=0.8)

    # if provided, plot detection epicenters
    if pltevents and not catalog['mag'].isnull().all():
        magmask = catalog['mag'] >= minmag
        lons = list(catalog['longitude'][magmask])
        lats = list(catalog['latitude'][magmask])
        mplmap.scatter(lons, lats, c='k', s=7, marker='x', zorder=5)
    elif catalog['mag'].isnull().all():
        lons = list(catalog['longitude'])
        lats = list(catalog['latitude'])
        mplmap.scatter(lons, lats, c='k', s=7, marker='x', zorder=5)

    plt.savefig('%s_eqdensity.png' % dirname, dpi=300)


@printstatus('Making histogram of given parameter')
def make_hist(catalog, param, binsize, dirname, title='', xlabel='',
              countlabel=False):
    """Plot histogram grouped by some parameter."""
    paramlist = catalog[pd.notnull(catalog[param])][param].tolist()
    minparam, maxparam = min(paramlist), max(paramlist)
    paramdown = round2bin(minparam, binsize, 'down')
    paramup = round2bin(maxparam, binsize, 'up')
    numbins = int((paramup-paramdown) / binsize)
    labelbuff = float(paramup-paramdown) / numbins * 0.5

    diffs = [abs(paramlist[i+1]-paramlist[i]) for i in range(len(paramlist))
             if i+1 < len(paramlist)]
    diffs = [round(x, 1) for x in diffs if x > 0]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Count', fontsize=14)

    if param == 'ms':
        parambins = np.linspace(paramdown, paramup, numbins+1)
        plt.xlim(paramdown, paramup)
    else:
        parambins = np.linspace(paramdown, paramup+binsize,
                                numbins+2) - binsize/2.
        plt.xlim(paramdown+binsize/2., paramup+binsize/2.)

    phist = plt.hist(paramlist, parambins, alpha=0.7, color='b', edgecolor='k')
    maxbarheight = max([phist[0][x] for x in range(numbins)] or [0])
    labely = maxbarheight / 50.

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.11)

    plt.ylim(0, maxbarheight*1.1+0.1)

    # put count numbers above the bars if countlabel=True
    if countlabel:
        for i in range(numbins):
            plt.text(phist[1][i]+labelbuff, phist[0][i]+labely,
                     '%0.f' % phist[0][i], size=12, ha='center')

    plt.savefig('%s_%shistogram.png' % (dirname, param), dpi=300)


@printstatus('Making histogram of given time duration')
def make_time_hist(catalog, timelength, dirname, title=''):
    """Make histogram either by hour of the day or by date."""
    timelist = catalog['time']

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Count', fontsize=14)

    if timelength == 'hour':
        lons = np.linspace(-180, 180, 25).tolist()
        hours = np.linspace(-12, 12, 25).tolist()

        tlonlist = catalog.loc[:, ['longitude', 'time']]
        tlonlist.loc[:, 'rLon'] = round2lon(tlonlist['longitude'])

        tlonlist.loc[:, 'hour'] = [int(x.split('T')[1].split(':')[0])
                                   for x in tlonlist['time']]
        tlonlist.loc[:, 'rhour'] = [x.hour + hours[lons.index(x.rLon)]
                                    for x in tlonlist.itertuples()]

        tlonlist.loc[:, 'rhour'] = [x+24 if x < 0 else x-24 if x > 23 else x
                                    for x in tlonlist['rhour']]

        hourlist = tlonlist.rhour.tolist()
        hourbins = np.linspace(-0.5, 23.5, 25)

        plt.hist(hourlist, hourbins, alpha=0.7, color='b', edgecolor='k')
        plt.xlabel('Hour of the Day', fontsize=14)
        plt.xlim(-0.5, 23.5)

    elif timelength == 'day':
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


@printstatus('Graphing mean time separation')
def graph_time_sep(catalog, dirname):
    """Make bar graph of mean time separation between events by date."""
    catalog.loc[:, 'convtime'] = [' '.join(x.split('T')).split('.')[0]
                                  for x in catalog['time'].tolist()]
    catalog.loc[:, 'convtime'] = catalog['convtime'].astype('datetime64[ns]')
    catalog.loc[:, 'dt'] = catalog.convtime.diff().astype('timedelta64[ns]')
    catalog.loc[:, 'dtmin'] = catalog['dt'] / pd.Timedelta(minutes=1)

    mindate = catalog['convtime'].min() - pd.Timedelta(days=15)
    maxdate = catalog['convtime'].max() - pd.Timedelta(days=15)

    fig = plt.figure(figsize=(10, 6))
    axfull = fig.add_subplot(111)
    axfull.set_ylabel('Time separation (min)')
    axfull.spines['top'].set_color('none')
    axfull.spines['bottom'].set_color('none')
    axfull.spines['left'].set_color('none')
    axfull.spines['right'].set_color('none')
    axfull.tick_params(labelcolor='w', top='off', bottom='off',
                       left='off', right='off')

    fig.add_subplot(311)
    plt.plot(catalog['convtime'], catalog['dtmin'], alpha=1, color='b')
    plt.xlabel('Date')
    plt.title('Time separation between events')
    plt.xlim(mindate+pd.Timedelta(days=15), maxdate)
    plt.ylim(0)

    fig.add_subplot(312)
    month_max = catalog.resample('1M', on='convtime').max()['dtmin']
    months = month_max.index.map(lambda x: x.strftime('%Y-%m')).tolist()
    months = [date(int(x[:4]), int(x[-2:]), 1) for x in months]
    plt.bar(months, month_max.tolist(), color='b', alpha=1, width=31)
    plt.xlabel('Month')
    plt.title('Maximum event separation by month')
    plt.xlim(mindate, maxdate)

    fig.add_subplot(313)
    month_med = catalog.resample('1M', on='convtime').median()['dtmin']
    plt.bar(months, month_med.tolist(), color='b', alpha=1, width=31)
    plt.xlabel('Month')
    plt.title('Median event separation by month')
    plt.tight_layout()
    plt.xlim(mindate, maxdate)

    plt.savefig('%s_timeseparation.png' % dirname, dpi=300)


@printstatus('Graphing magnitude completeness')
def cat_mag_comp(catalog, dirname, magbin=0.1):
    """Plot catalog magnitude completeness."""
    catalog = catalog[pd.notnull(catalog['mag'])]
    mags = np.array(catalog[catalog['mag'] > 0]['mag'])
    mags = np.around(mags, 1)

    minmag, maxmag = 0, max(mags)

    mag_centers = np.arange(minmag, maxmag + 2*magbin, magbin)
    cdf = np.zeros(len(mag_centers))

    for idx in range(len(cdf)):
        cdf[idx] = np.count_nonzero(
            ~np.isnan(mags[mags >= mag_centers[idx]-0.001]))

    mag_edges = np.arange(minmag - magbin/2., maxmag+magbin, magbin)
    g_r, _ = np.histogram(mags, mag_edges)
    idx = list(g_r).index(max(g_r))

    mc_est = mag_centers[idx]

    try:
        mc_est, bvalue, avalue, lval, mc_bins, std_dev = WW2000(mc_est, mags,
                                                                magbin)
    except:
        mc_est = mc_est + 0.3
        mc_bins = np.arange(0, maxmag + magbin/2., magbin)
        bvalue = np.log10(np.exp(1))/(np.average(mags[mags >= mc_est])
                                      - (mc_est-magbin/2.))
        avalue = np.log10(len(mags[mags >= mc_est])) + bvalue*mc_est
        log_l = avalue-bvalue*mc_bins
        lval = 10.**log_l
        std_dev = bvalue/sqrt(len(mags[mags >= mc_est]))

    plt.figure(figsize=(8, 6))
    plt.scatter(mag_centers[:-1], g_r, edgecolor='r', marker='o',
                facecolor='none', label='Incremental')
    plt.scatter(mag_centers, cdf, c='k', marker='+', label='Cumulative')
    plt.axvline(mc_est, c='r', linestyle='--', label='Mc = %2.1f' % mc_est)
    plt.plot(mc_bins, lval, c='k', linestyle='--',
             label='B = %1.3f%s%1.3f' % (bvalue, u'\u00B1', std_dev))

    ax1 = plt.gca()
    ax1.set_yscale('log')
    max_count = np.amax(cdf) + 100000
    ax1.set_xlim([0, maxmag])
    ax1.set_ylim([1, max_count])
    plt.title('Frequency-Magnitude Distribution', fontsize=18)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Log10 Count', fontsize=14)
    plt.legend(numpoints=1)

    plt.savefig('%s_catmagcomp.png' % dirname, dpi=300)


@printstatus('Graphing magnitude versus time for each earthquake')
def graph_mag_time(catalog, dirname):
    """Plot magnitudes vs. origin time."""
    catalog = catalog[pd.notnull(catalog['mag']) & (catalog['mag'] > 0)]
    catalog.loc[:, 'convtime'] = [' '.join(x.split('T')).split('.')[0]
                                  for x in catalog['time'].tolist()]
    catalog.loc[:, 'convtime'] = catalog['convtime'].astype('datetime64[ns]')

    times = catalog['time']
    mags = catalog['mag']

    plt.figure(figsize=(10, 6))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.plot_date(times, mags, alpha=0.7, markersize=2, c='b')
    plt.xlim(min(times), max(times))
    plt.ylim(0)

    plt.savefig('%s_magvtime.png' % dirname, dpi=300)


@printstatus('Graphing cumulative moment release')
def cumul_moment_release(catalog, dirname):
    """Graph cumulative moment release."""
    catalog = catalog[pd.notnull(catalog['mag']) & (catalog['mag'] > 0)]
    catalog.loc[:, 'convtime'] = [' '.join(x.split('T')).split('.')[0]
                                  for x in catalog['time'].tolist()]
    catalog.loc[:, 'convtime'] = catalog['convtime'].astype('datetime64[ns]')
    times = catalog['convtime']

    minday, maxday = min(times), max(times)

    mag0 = 10.**((3/2.)*(catalog['mag']+10.7))
    mag0 = mag0 * 10.**(-7)
    cumulmag0 = np.cumsum(mag0)

    plt.figure(figsize=(10, 6))
    plt.plot(times, cumulmag0, 'k-')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(r'Cumulative Moment Release (N$\times$m)', fontsize=14)
    plt.xlim(minday, maxday)
    plt.ylim(0)

    colors = ['r', 'm', 'c', 'y', 'g']
    largesteqs = catalog.sort_values('mag').tail(5)
    for i, eq in enumerate(largesteqs.itertuples()):
        plt.axvline(x=eq.time, color=colors[i], linestyle='--')

    plt.savefig('%s_cumulmomentrelease.png' % dirname, dpi=300)


@printstatus('Graphing cumulative event types')
def graph_event_types(catalog, dirname):
    """Graph number of cumulative events by type of event."""
    typedict = {}

    for evtype in catalog['type'].unique():
        typedict[evtype] = (catalog['type'] == evtype).cumsum()

    plt.figure(figsize=(12, 6))

    for evtype in typedict:
        plt.plot_date(catalog['time'], typedict[evtype], marker=None,
                      linestyle='-', label=evtype)

    plt.yscale('log')
    plt.legend()

    plt.savefig('%s_cumuleventtypes.png' % dirname, dpi=300)


@printstatus('Graphing possible number of duplicate events')
def cat_dup_search(catalog, dirname):
    """Graph possible number of duplicate events given various distances
    and time differences.
    """
    epochtimes = [to_epoch(row.time) for row in catalog.itertuples()]
    tdifsec = np.asarray(abs(np.diff(epochtimes)))

    lat1 = np.asarray(catalog.latitude[:-1])
    lon1 = np.asarray(catalog.longitude[:-1])
    lat2 = np.asarray(catalog.latitude[1:])
    lon2 = np.asarray(catalog.longitude[1:])
    ddelkm = [eq_dist(lat1[i], lon1[i], lat2[i], lon2[i])
              for i in range(len(lat1))]

    diffdf = pd.DataFrame({'tdifsec': tdifsec, 'ddelkm': ddelkm})

    kmlimits = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tmax = 16
    dtime = 0.05
    timebins = np.arange(0, tmax+dtime/2, dtime)

    numevents = np.empty([len(kmlimits), len(timebins)-1])

    for jdx in range(len(kmlimits)):

        cat_subset = diffdf[diffdf.ddelkm <= kmlimits[jdx]]

        for idx in range(len(timebins)-1):

            numevents[jdx][idx] = cat_subset[cat_subset.tdifsec.between(
                timebins[idx], timebins[idx+1])].count()[0]

    totmatch = np.transpose(np.cumsum(np.transpose(numevents), axis=0))

    plt.figure(figsize=(10, 6))
    for idx in range(len(kmlimits)):
        times = timebins[1:]
        matches = totmatch[idx]
        lab = str(kmlimits[idx]) + ' km'
        plt.plot(times, matches, label=lab)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Total Number of Events', fontsize=14)
    plt.xlim(0, tmax)
    plt.ylim(0, np.amax(totmatch)+0.5)
    plt.legend(loc=2, numpoints=1)

    plt.savefig('%s_catdupsearch.png' % dirname, dpi=300)


###############################################################################
###############################################################################
###############################################################################


def main():
    """Main function. Command line arguments defined here."""
    parser = argparse.ArgumentParser()

    parser.add_argument('catalog', nargs='?', type=str,
                        help='pick which catalog to download data from; to \
                        download data from all catalogs, use "all"')
    parser.add_argument('startyear', nargs='?', type=int,
                        help='pick starting year')
    parser.add_argument('endyear', nargs='?', type=int,
                        help='pick end year (to get a single year of data, \
                        enter same year as startyear)')

    parser.add_argument('-sf', '--specifyfile', type=str,
                        help='specify existing .csv file to use')
    parser.add_argument('-fd', '--forcedownload', action='store_true',
                        help='forces downloading of data even if .csv file \
                        exists')

    args = parser.parse_args()

    if args.specifyfile is None:

        if not args.catalog:
            sys.stdout.write('No catalog specified. Exiting...\n')
            sys.exit()
        elif not args.startyear:
            sys.stdout.write('No starting year specified. Exiting...\n')
            sys.exit()
        elif not args.endyear:
            sys.stdout.write('No ending year specified. Exiting...\n')
            sys.exit()

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
            datadf = get_data(catalog, startyear, endyear, write=True)
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
                    datadf = get_data(catalog, startyear, endyear, write=True)
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
                    datadf = get_data(catalog, startyear, endyear, write=True)

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

    if len(datadf) == 0:
        sys.stdout.write(('Catalog has no data available for that time period.'
                          ' Quitting...'))
        sys.exit()

    datadf = datadf.sort_values(by='time').reset_index(drop=True)
    datadf.loc[:, 'ms'] = datadf['time'].str[-4:-1].astype('float')

    os.chdir(dirname)
    basic_cat_sum(datadf, dirname)
    map_detecs(datadf, dirname)
    plt.close()
    map_detec_nums(datadf, dirname)
    plt.close()
    make_hist(datadf, 'mag', 0.1, dirname, xlabel='Magnitude')
    plt.close()
    make_hist(datadf, 'depth', 10, dirname, xlabel='Depth (km)')
    plt.close()
    make_hist(datadf, 'ms', 20, dirname, xlabel='Milliseconds')
    plt.close()
    make_time_hist(datadf, 'hour', dirname)
    plt.close()
    make_time_hist(datadf, 'day', dirname)
    plt.close()
    graph_time_sep(datadf, dirname)
    plt.close()
    cat_mag_comp(datadf, dirname)
    plt.close()
    graph_mag_time(datadf, dirname)
    plt.close()
    cumul_moment_release(datadf, dirname)
    plt.close()
    graph_event_types(datadf, dirname)
    plt.close()
    cat_dup_search(datadf, dirname)


if __name__ == '__main__':

    try:
        main()
    except (KeyboardInterrupt, SystemError):
        sys.stdout.write('\nProgram canceled. Exiting...\n')
        sys.exit()
