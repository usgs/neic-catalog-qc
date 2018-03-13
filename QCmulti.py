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

from scipy import stats
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth

# Python 2
try:
    from urllib2 import urlopen, HTTPError
# Python 3
except ImportError:
    from urllib.request import urlopen, HTTPError

import QCutils as qcu
from decorators import retry, printstatus


###############################################################################
###############################################################################
###############################################################################


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

    mintime = qcu.format_time(max(cat1['time'].min(), cat2['time'].min()))
    maxtime = qcu.format_time(min(cat1['time'].max(), cat2['time'].max()))

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
    cat1.loc[:, 'time'] = [qcu.to_epoch(x) for x in cat1['time']]
    cat2.loc[:, 'time'] = [qcu.to_epoch(x) for x in cat2['time']]

    cat1, cat2 = qcu.trim_times(cat1, cat2, otwindow)

    cat1ids, cat2ids = [], []

    for i in range(len(cat1)):

        cat2ix = cat2[abs(cat2['time']
                          - cat1.ix[i]['time']) <= otwindow].index.values

        if len(cat2ix) != 0:

            carr = np.array([gps2dist_azimuth(cat1.ix[i]['latitude'],
                cat1.ix[i]['longitude'], cat2.ix[x]['latitude'],
                cat2.ix[x]['longitude'])[0] / 1000. for x in cat2ix])

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
    """Map catalog events."""
    lllat, lllon, urlat, urlon, _, _, _, clon = qcu.get_map_bounds(cat1, cat2)

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
    mplmap.add_feature(cfeature.NaturalEarthFeature('cultural',
        'admin_1_states_provinces_lines', '50m', facecolor='none',
        edgecolor='k', zorder=9))
    mplmap.add_feature(cfeature.BORDERS)

    plt.savefig('%s_mapmatcheddetecs.png' % dirname, dpi=300)
    plt.close()


@printstatus('Graphing polar histogram of azimuths and distances')
def make_az_dist(cat1, cat2, cat1mids, cat2mids, dirname,
                 distwindow=100, numbins=16):
    """Make polar scatter/histogram of azimuth vs. distance."""
    azimuths, distances = qcu.get_azs_and_dists(cat1, cat2, cat1mids, cat2mids)

    cat1name, cat2name = dirname[0:2].upper(), dirname[3:5].upper()

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
    ax1.set_title('%s location relative to %s' % (cat1name, cat2name),
                  fontsize=20)

    for _, hbar in list(zip(hist, bars)):
        hbar.set_facecolor('b')
        hbar.set_alpha(0.5)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.11)

    plt.savefig('%s_polarazimuth.png' % dirname, dpi=300)
    plt.close()


@printstatus('Comparing parameters of matched events')
def compare_params(cat1, cat2, cat1mids, cat2mids, param, dirname):
    """Compare parameters of matched events."""
    cat1params, cat2params = [], []

    for idx, eid in enumerate(cat1mids):
        param1 = float(cat1[cat1['id'] == eid][param])
        param2 = float(cat2[cat2['id'] == cat2mids[idx]][param])

        cat1params.append(param1)
        cat2params.append(param2)

    minparam = max(min(cat1params), min(cat2params))
    maxparam = min(max(cat1params), max(cat2params))

    mval, bval, rval, _, _ = stats.linregress(cat1params, cat2params)
    linegraph = [mval*x + bval for x in cat1params]
    r2val = rval*rval

    cat1name, cat2name = dirname[0:2].upper(), dirname[3:5].upper()
    aparam = param if param != 'mag' else 'magnitude'
    tparam = aparam.capitalize()

    plt.figure(figsize=(8, 8))
    plt.scatter(cat1params, cat2params, edgecolor='b', facecolor=None)
    plt.plot(cat1params, linegraph, c='r', linewidth=1,
             label=r'$\mathregular{R^2}$ = %0.2f' % r2val)
    plt.plot(cat1params, cat1params, c='k', linewidth=1, label='B = 1')
    plt.legend(loc='upper left')
    plt.xlim(minparam, maxparam)
    plt.ylim(minparam, maxparam)
    plt.xlabel('%s %s' % (cat1name, aparam), fontsize=14)
    plt.ylabel('%s %s' % (cat2name, aparam), fontsize=14)

    plt.title('%s correlation' % tparam, fontsize=20)

    plt.savefig('%s_compare%s.png' % (dirname, param), dpi=300)
    plt.close()


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
    pardiffdown = qcu.round2bin(minpardiff, binsize, 'down')
    pardiffup = qcu.round2bin(maxpardiff, binsize, 'up')
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
    plt.close()


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
            datadf1, datadf2 = qcu.get_data(cat1, catalog2=cat2,
                startyear=startyear, endyear=endyear)
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
                    datadf1, datadf2 = qcu.get_data(cat1, catalog2=cat2,
                        startyear=startyear, endyear=endyear)
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
                    datadf1, datadf2 = qcu.get_data(cat1, catalog2=cat2,
                        startyear=startyear, endyear=endyear)

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
                          'period. Quitting...\n') % cat1.upper())
        sys.exit()

    if len(datadf2) == 0:
        sys.stdout.write(('%s catalog has no data available for that time '
                          'period. Quitting...\n') % cat2.upper())
        sys.exit()

    os.chdir(dirname)
    basic_cat_sum(datadf1, cat1, dirname)
    basic_cat_sum(datadf2, cat2, dirname)
    comp_criteria(datadf1, datadf2, dirname, cat1.upper())
    cat1ids, cat2ids, newcat1, newcat2 = match_events(datadf1, datadf2)

    map_events(newcat1, newcat2, cat1.upper(), dirname)
    make_az_dist(newcat1, newcat2, cat1ids, cat2ids, dirname)
    compare_params(newcat1, newcat2, cat1ids, cat2ids, 'mag', dirname)
    compare_params(newcat1, newcat2, cat1ids, cat2ids, 'depth', dirname)
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'time', 0.5, dirname,
                   xlabel='%s-%s time residuals (sec)' % (cat1.upper(),
                   cat2.upper()))
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'mag', 0.1, dirname,
                   xlabel='%s-%s magnitude residuals' % (cat1.upper(),
                   cat2.upper()))
    make_diff_hist(newcat1, newcat2, cat1ids, cat2ids, 'depth', 2, dirname,
                   xlabel='%s-%s depth residuals (km)' % (cat1.upper(),
                   cat2.upper()))


if __name__ == '__main__':

    try:
        main()
    except (KeyboardInterrupt, SystemError):
        sys.stdout.write('\nProgram canceled. Exiting...\n')
        sys.exit()
