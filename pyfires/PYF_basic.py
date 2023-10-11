#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Simon R Proud
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Basic reading functions for data preparation."""

from dask_image.ndfilters import convolve
import pyfires.PYF_Consts as PYFc
from satpy import Scene
import dask.array as da
from glob import glob

from pyspectral.rsr_reader import RelativeSpectralResponse
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def conv_kernel(indata, ksize=5):
    """Convolve a kernel with a dataset."""

    kern = np.ones((ksize, ksize))
    kern = kern / np.sum(kern)
    res = convolve(indata.data, kern)

    outds = indata.copy()
    outds.attrs['name'] = indata.attrs['name'] + f'_conv{ksize}'
    outds.data = res

    return outds


def calc_rad_fromtb(temp, cwl):
    """Modified version of eqn 2 in Wooster's paper. Compute radiance from BT (K) and central wavelength (micron)."""
    c1 = 1.1910429e8
    c2 = 1.4387752e4

    first = c1
    second = np.power(cwl, 5) * (np.exp(c2 / (cwl * temp)) - 1.)

    return first / second


def save_output(inscn, indata, name, odir, ref='B07'):
    inscn[name] = inscn[ref].copy()
    inscn[name].attrs['name'] = name
    inscn[name].data = indata
    inscn.save_dataset(name, base_dir=odir, enhance=False, dtype=np.float32)


def bt_to_rad(wvl, bt, d0, d1, d2):
    h = 6.62606957e-34
    c = 299792458.0
    k = 1.3806488e-23

    bt_c = d0 + d1 * bt + d2 * bt * bt

    a = 2 * h * c ** 2 / wvl ** 5
    b = np.exp((h * c) / (k * wvl * bt_c)) - 1
    return 1e-6 * a / b


def rad_to_bt(wvl, rad, c0, c1, c2):
    h = 6.62606957e-34
    c = 299792458.0
    k = 1.3806488e-23

    rad2 = rad * 1e6

    a = (h * c) / (wvl * k)
    b = 1 + (2 * h ** 2 * c) / (rad2 * wvl ** 5)

    bt = a / np.log(b)

    return c0 + c1 * bt + c2 * bt * bt


def convert_radiance(the_scn, blist):
    """Convert radiance from wavelength to wavenumber space.
    This uses the approach described in:
    Converting Advanced Himawari Imager (AHI) Radiance Units
    Mathew Gunshor, 2015
    https://cimss.ssec.wisc.edu/goes/calibration/Converting_AHI_RadianceUnits_24Feb2015.pdf

    Inputs:
     - the_scn: A satpy Scene containing the bands to convert.
     - inbands: A list of the band names that require processing.
    Returns:
     - the_scn: Modified scene containing converted radiances.
    """
    import scipy.integrate as integ

    det = 'det-1'

    for chan in blist:
        sat = the_scn[chan].attrs['platform_name']
        sensor = the_scn[chan].attrs['sensor'].upper()

        srf = RelativeSpectralResponse(sat, sensor)
        cur_rsr = srf.rsr[chan]

        wvl = cur_rsr[det]['wavelength']
        wvn = np.flip(1e4 / cur_rsr[det]['wavelength'])
        rsr = cur_rsr[det]['response']
        h_wvn = (wvn[-1] - wvn[0]) / len(wvn)
        h_wvl = (wvl[-1] - wvl[0]) / len(wvl)

        int_wvn = (h_wvn / 2.) * (rsr[0] + 2 * np.sum(rsr[1:-1]) + rsr[-1])
        int_wvl = (h_wvl / 2.) * (rsr[0] + 2 * np.sum(rsr[1:-1]) + rsr[-1])

        the_scn[chan].data = 1000. * the_scn[chan].data * (int_wvl / int_wvn)

    return the_scn


def initial_load(infiles_l1,
                 l1_reader,
                 bdict,
                 rad_dict,
                 infiles_cld=None,
                 cld_reader=None,
                 cmask_name=None,
                 lw_bt_thresh=0,
                 sw_bt_thresh=0,
                 do_load_lsm=True):
    """Read L1 and Cloudmask from disk based on user preferences.
    Inputs:
     - infiles_l1: List of L1 files to read.
     - l1_reader: Satpy reader to use for L1 files.
     - rad_dict: Dictionary of solar irradiance coefficients.
     - bdict: Dictionary of bands to read from L1 files.
     - infiles_cld: List of cloudmask files to read.
     - cld_reader: Satpy reader to use for cloudmask files.
     - cmask_name: Name of cloudmask band to read.
     - lw_bt_thresh: Threshold LWIR brightness temperature to use for cloudmasking.
     - sw_bt_thresh: Threshold SWIR brightness temperature to use for cloudmasking.
     - do_load_lsm: Boolean, whether to load a land-sea mask (default: True).
    Returns:
     - scn: A satpy Scene containing the data read from disk.
    """
    # Check that user has provided requested info.

    from satpy.modifiers.angles import compute_relative_azimuth, get_angles

    blist = []
    for item in bdict:
        blist.append(bdict[item])

    scn = Scene(infiles_l1, reader=l1_reader)
    scn.load(blist, calibration='radiance', generate=False)

    scn2 = Scene(infiles_l1, reader='ahi_hsd')
    scn2.load([bdict['lwi_band'], bdict['lw2_band'], bdict['swi_band']], generate=False)

    scn['VI1_RAD'] = scn[bdict['vi1_band']]
    scn['VI2_RAD'] = scn[bdict['vi2_band']]
    scn['SWI_RAD'] = scn[bdict['swi_band']]
    scn['LW1_RAD'] = scn[bdict['lwi_band']]
    scn['LW2_RAD'] = scn[bdict['lw2_band']]
    scn['LW1__BT'] = scn2[bdict['lwi_band']]
    scn['LW2__BT'] = scn2[bdict['lw2_band']]
    scn['SWI__BT'] = scn2[bdict['swi_band']]

    scn['BTD'] = scn['SWI__BT'] - scn['LW1__BT']
    scn['BTD'].attrs = scn['SWI_RAD'].attrs
    scn['BTD'].attrs['name'] = 'BTD'

    scn['BTD_LW'] = scn['LW1__BT'] - scn['LW2__BT']
    scn['BTD_LW'].attrs = scn['LW1__BT'].attrs
    scn['BTD_LW'].attrs['name'] = 'BTD_LW'

    sat = scn['LW1__BT'].attrs['platform_name']
    sen = scn['LW1__BT'].attrs['sensor']
    det = 'det-1'

    rsr = RelativeSpectralResponse(sat, sen)
    cur_rsr = rsr.rsr[bdict['swi_band']]
    wvl_swi = cur_rsr[det]['central_wavelength']

    scn = scn.resample(scn.coarsest_area(), resampler='native')

    exp_rad = calc_rad_fromtb(scn['LW1__BT'].data, wvl_swi)
    swi_diffrad = scn['SWI_RAD'].data - exp_rad
    swi_diffrad = da.where(np.isfinite(swi_diffrad), swi_diffrad, 0)

    swi_noir_name = 'SWI_RAD_NO_IR'
    scn[swi_noir_name] = scn['SWI_RAD'].copy()
    scn[swi_noir_name].data = swi_diffrad
    scn[swi_noir_name].attrs = scn['SWI_RAD'].attrs
    scn[swi_noir_name].attrs['name'] = swi_noir_name

    scn['VI1_RAD'].data = scn['VI1_RAD'].data * rad_dict['swi'] / rad_dict['vi1']
    scn['VI2_RAD'].data = scn['VI2_RAD'].data * rad_dict['swi'] / rad_dict['vi2']

    scn['VI1_DIFF'] = scn[swi_noir_name] - scn['VI1_RAD']
    scn['VI1_DIFF'].attrs = scn['SWI_RAD'].attrs
    scn['VI1_DIFF'].attrs['name'] = 'VI1_DIFF'

    scn['VI2_DIFF'] = scn[swi_noir_name] - scn['VI2_RAD']
    scn['VI2_DIFF'].attrs = scn['SWI_RAD'].attrs
    scn['VI2_DIFF'].attrs['name'] = 'VI2_DIFF'

    scn['sw_ndfi'] = scn['SWI_RAD'].copy()
    scn['sw_ndfi'].attrs['name'] = 'sw_ndfi'
    scn['sw_ndfi'].data = (scn['SWI__BT'].data - scn['LW1__BT'].data) / (scn['SWI__BT'].data + scn['LW1__BT'].data)

    scn['lw_ndfi'] = scn['SWI_RAD'].copy()
    scn['lw_ndfi'].attrs['name'] = 'lw_ndfi'
    scn['lw_ndfi'].data = (scn['LW1__BT'].data - scn['LW2__BT'].data) / (scn['LW2__BT'].data + scn['LW1__BT'].data)

    final_bnames = ['VI1_RAD', 'VI2_RAD', 'SWI_RAD', 'LW1_RAD', 'LW2_RAD',
                    'SWI__BT', 'LW1__BT', 'LW2__BT',
                    'SWI_RAD_NO_IR', 'VI1_DIFF', 'VI2_DIFF',
                    'sw_ndfi',]# 'lw_ndfi']

    for band in blist:
        del(scn[band])

    for band in final_bnames:
        scn[band].data = da.where(np.isfinite(scn[band].data), scn[band].data, np.nan)

        #if lw_bt_thresh > 100:
        #    scn[band].data = da.where(scn['LW1__BT'].data > lw_bt_thresh, scn[band].data, np.nan)
        #if sw_bt_thresh > 100:
        #    scn[band].data = da.where(scn['SWI__BT'].data > sw_bt_thresh, scn[band].data, np.nan)

    # Get the angles associated with the Scene
    vaa, vza, saa, sza = get_angles(scn['LW1__BT'])

    sza.data = da.where(sza.data > 90, 90, sza)

    raa = compute_relative_azimuth(vaa, saa)
    scn['VZA'] = scn['LW1__BT'].copy()
    scn['VZA'].data = vza.data
    scn['VZA'].attrs['name'] = 'VZA'
    scn['SZA'] = scn['LW1__BT'].copy()
    scn['SZA'].data = sza.data
    scn['SZA'].attrs['name'] = 'SZA'
    scn['RAA'] = scn['LW1__BT'].copy()
    scn['RAA'].data = raa.data
    scn['RAA'].attrs['name'] = 'RAA'

    if do_load_lsm:
        scn['LSM'] = load_lsm(scn['BTD'])

    return scn


def load_lsm(ds, xy_bbox=None, ll_bbox=None):
    """Select and load a land-sea mask based on the attributes of a provided satpy dataset.
    Inputs:
     - ds: The satpy dataset.
     - xy_bbox: Optional, a bounding box in x/y coordinates. If not given, full disk will be processed.
     - ll_bbox: Optional, a bounding box in lat/lon coordinates. If not given, full disk will be processed.
    Returns:
    - lsm: A land-sea mask dataset.

    Note: This function is only implemented for some geostationary sensors and at IR  / lowest spatial resolution.
    """
    import os
    dname = os.path.dirname(PYFc.__file__)

    platform = ds.attrs['platform_name']
    if platform in PYFc.plat_shortnames:
        satname = PYFc.plat_shortnames[platform]
    else:
        raise ValueError(f'Satellite {platform} not supported.')

    lon = ds.attrs['orbital_parameters']['projection_longitude']
    if lon < 0:
        prefix = 'M'
    else:
        prefix = 'P'

    lonstr = str(lon).replace('.', '')
    test_fname = f'{dname}/lsm/{satname}_{prefix}{lonstr}_LSM.tif'
    if os.path.exists(test_fname):
        iscn = Scene([test_fname], reader='generic_image')
        iscn.load(['image'])
        iscn['image'].attrs = ds.attrs
    else:
        raise ValueError(f'Land-sea mask for {platform} not found.')
    iscn['image'] = iscn['image'].squeeze()
    if xy_bbox is not None:
        iscn = iscn.crop(xy_bbox=xy_bbox)
    elif ll_bbox is not None:
        iscn = iscn.crop(ll_bbox=ll_bbox)

    data = iscn['image']
    data.attrs = ds.attrs
    data.attrs['name'] = 'Land-Sea mask'

    return data
