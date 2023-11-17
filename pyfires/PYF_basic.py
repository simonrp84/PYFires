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

from satpy.modifiers.angles import get_satellite_zenith_angle
from dask_image.ndfilters import convolve
import pyfires.PYF_Consts as PYFc
from satpy import Scene

from pyspectral.rsr_reader import RelativeSpectralResponse
from satpy.modifiers.angles import _get_sun_angles
import xarray as xr
import numpy as np


def vid_adjust_sza(in_vid, in_sza, sza_adj=82, min_v=0.04, max_v=0.135, slo_str=1.8, slo_rise=0.2):
    """Adjust the VI Difference based on solar zenith angle.
    Inputs:
    - in_vid: The VI Difference data
    - in_sza: The solar zenith angle in degrees
    - sza_adj: The SZA threshold at which adjustment begins. Default value: 82 degrees.
    - min_v: The minimum VID value, used during day. Default value: 0.04
    - max_v: The maximum VID value, used at night. Default value: 0.135
    - slo_str: The slope strength. Default value: 1.8
    - slo_rise: The slope rise. Default value: 0.2
    Returns:
    - adj_vid: The adjusted VI Difference data
     """

    adj_val = min_v + (max_v - min_v) * (1 / (1 + np.exp(-slo_str * (in_sza - sza_adj))) ** slo_rise)

    adj_vid = in_vid.copy()
    adj_vid.attrs['name'] = 'VI1_DIFF_ADJ'
    adj_vid.data = in_vid - adj_val
    return adj_vid


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


def calc_tb_fromrad(rad, cwl):
    """Inverse version of eqn 2 in Wooster's paper. Compute BT from radiance and central wavelength (micron)."""
    c1 = 1.1910429e8
    c2 = 1.4387752e4

    first = c2
    second = cwl * np.log(c1 / (rad * np.power(cwl, 5)) + 1)

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


def _get_band_solar(the_dict, bdict):
    """Compute the per-band solar irradiance for each loaded channel.
    Inputs:
    - the_dict: A dictionary containing bands to compute the irradiance for.
    - bdict: A dict of the band names and output names to compute the irradiance for.
    Returns:
    - irrad_dict: A dictionary containing the irradiance for each band.

    """
    from pyspectral.solar import SolarIrradianceSpectrum as SIS
    from pyspectral.rsr_reader import RelativeSpectralResponse

    irrad_dict = {}

    for band_name in bdict:
        sensor = the_dict[bdict[band_name]].attrs['sensor']
        platform = the_dict[bdict[band_name]].attrs['platform_name']
        srf = RelativeSpectralResponse(platform, sensor)
        cur_rsr = srf.rsr[bdict[band_name]]
        irrad = SIS().inband_solarirradiance(cur_rsr)
        irrad_dict[band_name] = irrad

    return irrad_dict


def compute_fire_datasets(indata_dict, irrad_dict, bdict):
    """Compute the intermediate datasets used for fire detection.
    Inputs:
    - indata_dict: A dictionary containing the input datasets.
    - irrad_dict: A dictionary containing the solar irradiance values for each band.
    - bdict: Dictionary of bands to read from L1 files.
    Returns:
    - indata_dict: The input dictionary, with the computed datasets added.
    """
    indata_dict['BTD'] = indata_dict['MIR__BT'] - indata_dict['LW1__BT']
    indata_dict['BTD'].attrs = indata_dict['MIR_RAD'].attrs
    indata_dict['BTD'].attrs['name'] = 'BTD'

    sat = indata_dict['LW1__BT'].attrs['platform_name']
    sen = indata_dict['LW1__BT'].attrs['sensor']
    det = 'det-1'

    rsr = RelativeSpectralResponse(sat, sen)
    cur_rsr = rsr.rsr[bdict['mir_band']]
    wvl_mir = cur_rsr[det]['central_wavelength']

    exp_rad = calc_rad_fromtb(indata_dict['LW1__BT'].data, wvl_mir)
    mir_diffrad = indata_dict['MIR_RAD'].data - exp_rad
    mir_diffrad = np.where(np.isfinite(mir_diffrad), mir_diffrad, 0)

    mir_noir_name = 'MIR_RAD_NO_IR'
    indata_dict[mir_noir_name] = indata_dict['MIR_RAD'].copy()
    indata_dict[mir_noir_name].data = mir_diffrad
    indata_dict[mir_noir_name].attrs = indata_dict['MIR_RAD'].attrs
    indata_dict[mir_noir_name].attrs['name'] = mir_noir_name

    indata_dict['VI1_RAD'].data = indata_dict['VI1_RAD'].data * irrad_dict['mir_band'] / irrad_dict['vi1_band']
    indata_dict['VI2_RAD'].data = indata_dict['VI2_RAD'].data * irrad_dict['mir_band'] / irrad_dict['vi2_band']

    indata_dict['RAD_ADD'] = indata_dict[mir_noir_name] + indata_dict['VI1_RAD']
    indata_dict['RAD_ADD'] = indata_dict['RAD_ADD'] + indata_dict['VI2_RAD'].data
    indata_dict['RAD_ADD'].attrs = indata_dict['MIR_RAD'].attrs
    indata_dict['RAD_ADD'].attrs['name'] = 'RAD_ADD'

    indata_dict['VI1_DIFF'] = indata_dict[mir_noir_name] - indata_dict['VI1_RAD']
    indata_dict['VI1_DIFF'].attrs = indata_dict['MIR_RAD'].attrs
    indata_dict['VI1_DIFF'].attrs['name'] = 'VI1_DIFF'

    indata_dict['mi_ndfi'] = indata_dict['MIR_RAD'].copy()
    indata_dict['mi_ndfi'].attrs['name'] = 'mi_ndfi'
    indata_dict['mi_ndfi'].data = (indata_dict['MIR__BT'].data - indata_dict['LW1__BT'].data) / (
                indata_dict['MIR__BT'].data + indata_dict['LW1__BT'].data)

    # Get the angles associated with the Scene
    indata_dict['SZA'], indata_dict['VZA'], indata_dict['pix_area'] = get_angles(indata_dict['LW1__BT'])

    # Compute the adjusted VI difference, with reduced daytime VIS component
    adj_vid = vid_adjust_sza(indata_dict['VI1_DIFF'], indata_dict['SZA'])
    indata_dict['VI1_DIFF_2'] = indata_dict['MIR_RAD'].copy()
    indata_dict['VI1_DIFF_2'].attrs['name'] = 'mi_ndfi'
    indata_dict['VI1_DIFF_2'].data = adj_vid

    #
    final_bnames = ['VI1_RAD', 'VI2_RAD', 'MIR_RAD', 'LW1_RAD', 'MIR__BT', 'LW1__BT',
                    'MIR_RAD_NO_IR', 'VI1_DIFF', 'mi_ndfi', ]
    for band in final_bnames:
        indata_dict[band].data = np.where(np.isfinite(indata_dict[band].data), indata_dict[band].data, np.nan)

        # if lw_bt_thresh > 100:
        #    scn[band].data = np.where(scn['LW1__BT'].data > lw_bt_thresh, scn[band].data, np.nan)
        # if mir_bt_thresh > 100:
        #    scn[band].data = np.where(scn['MIR__BT'].data > mir_bt_thresh, scn[band].data, np.nan)

    return indata_dict



def get_angles(ref_data):
    """Compute the solar and viewing zenith angles and the pixel size for a dataset.
    Inputs:
    - ref_data: A satpy dataset to be used as a reference.
    Returns:
    - sza: The solar zenith angle.
    - vza: The viewing zenith angle.
    - pix_area: The pixel area in km^2.
    """
    # Solar zenith
    saa, sza_data = _get_sun_angles(ref_data)
    sza = ref_data.copy()
    sza.data = sza_data.data
    sza.attrs['name'] = 'SZA'

    # Satellite zenith
    vza = ref_data.copy()
    vza.data = get_satellite_zenith_angle(ref_data)
    vza.attrs['name'] = 'VZA'

    # Compute the pixel area
    # Pixel sizes are in meters, convert to km
    pix_size = ref_data.attrs['area'].pixel_size_x * ref_data.attrs['area'].pixel_size_x * 1e-6

    # Multiply by inverse vza to gain estimate of pixel size across image.
    pix_area = ref_data.copy()
    pix_area.attrs['name'] = 'pix_area'
    pix_area.data = pix_size / np.cos(np.deg2rad(vza))

    # Return values, casting to float32 to save memory.
    return sza.astype(np.float32), vza.astype(np.float32), pix_area.astype(np.float32)


def initial_load(infiles_l1,
                 l1_reader,
                 bdict,
                 infiles_cld=None,
                 cld_reader=None,
                 cmask_name=None,
                 lw_bt_thresh=0,
                 mir_bt_thresh=0,
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
     - mir_bt_thresh: Threshold MIR brightness temperature to use for cloudmasking.
     - do_load_lsm: Boolean, whether to load a land-sea mask (default: True).
    Returns:
     - scn: A satpy Scene containing the data read from disk.
    """
    # Check that user has provided requested info.
    blist = []
    for item in bdict:
        blist.append(bdict[item])


    scn = Scene(infiles_l1, reader=l1_reader)
    scn.load(blist, calibration='radiance', generate=False)

    scn2 = Scene(infiles_l1, reader=l1_reader)
    scn2.load([bdict['lwi_band'], bdict['mir_band']], generate=False)

    scn = scn.resample(scn.coarsest_area(), resampler='native')
    scn2 = scn2.resample(scn.coarsest_area(), resampler='native')

    data_dict = {'VI1_RAD': scn[bdict['vi1_band']],
                 'VI2_RAD': scn[bdict['vi2_band']],
                 'MIR_RAD': scn[bdict['mir_band']],
                 'LW1_RAD': scn[bdict['lwi_band']],
                 'LW1__BT': scn2[bdict['lwi_band']].copy(),
                 'MIR__BT': scn2[bdict['mir_band']].copy()}

    # Compute the solar irradiance values
    irrad_dict = _get_band_solar(scn, bdict)

    # Compute the datasets required for fire detection.
    data_dict = compute_fire_datasets(data_dict, irrad_dict, bdict)

    # Lastly, load the land-sea mask
    if do_load_lsm:
        data_dict['LSM'] = load_lsm(data_dict['BTD'])

    return data_dict


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

    # Determind the platform used for the LSM
    platform = ds.attrs['platform_name']
    if platform in PYFc.plat_shortnames:
        satname = PYFc.plat_shortnames[platform]
    else:
        raise ValueError(f'Satellite {platform} not supported.')

    # Select correct longitude prefix
    lon = ds.attrs['orbital_parameters']['projection_longitude']
    if lon < 0:
        prefix = 'M'
    else:
        prefix = 'P'

    # Build the expected LSM filename
    lonstr = str(lon).replace('.', '')
    test_fname = f'{dname}/lsm/{satname}_{prefix}{lonstr}_LSM.tif'
    if os.path.exists(test_fname):
        iscn = Scene([test_fname], reader='generic_image')
        iscn.load(['image'])
        iscn['image'].attrs = ds.attrs
    else:
        raise ValueError(f'Land-sea mask for {platform} not found.')

    # Ensure there's no singleton dimensions
    iscn['image'] = iscn['image'].squeeze()
    if xy_bbox is not None:
        iscn = iscn.crop(xy_bbox=xy_bbox)
    elif ll_bbox is not None:
        iscn = iscn.crop(ll_bbox=ll_bbox)

    # Set up some attributes
    lsm_data = iscn['image']
    lsm_data.attrs = ds.attrs
    lsm_data.attrs['name'] = 'Land-Sea mask'

    # The LSM is often loaded with differing chunk sizes to the input data, so rechunk to match.
    lsm_data.data = lsm_data.data.rechunk(ds.data.chunks)
    lsm_data.coords['x'] = ds.coords['x']
    lsm_data.coords['y'] = ds.coords['y']

    return lsm_data


def comp_stat(x, a, b):
    """Function for the stage 5 confidence assignment (eqn 8 in Roberts)."""
    out = (x - a) / (b - a)
    out = xr.where(x < a, 0, out)
    out = xr.where(x > b, 1, out)
    return out


def do_stage5(btd, mea_btd, std_btd, bt_mir, mea_mir, std_mir,
              sza, nwin, ncld, nwat):
    """Compute the confidence value for each potential fire pixel.
    Inputs:
    - indict: The input data dictionary.
    Returns:
    - outarr_conf: An array of identical shape to the input satellite imagery containing confidence values.
    - outarr_frp: An array of identical shape to the input satellite imagery containing estimated FRP values.
    """

    outarr_conf = xr.zeros_like(btd)

    nig_sza = 60.

    sza_thr = xr.where(sza > nig_sza, nig_sza, sza)

    z4 = (bt_mir - mea_mir) / std_mir
    zdt = (btd - mea_btd) / std_btd

    min_c1_day = 287
    min_c1_nig = 280
    max_c1_day = 327
    max_c1_nig = 310

    min_c2 = 0.9
    max_c2 = 6.

    min_c3_day = 2
    min_c3_nig = 1.5
    max_c3_day = 6
    max_c3_nig = 5

    c1_grad = (min_c1_nig - min_c1_day) / nig_sza
    min_c1 = min_c1_day + c1_grad * sza_thr
    c1_grad = (max_c1_nig - max_c1_day) / nig_sza
    max_c1 = max_c1_day + c1_grad * sza_thr

    c3_grad = (min_c3_nig - min_c3_day) / nig_sza
    min_c3 = min_c3_day + c3_grad * sza_thr
    c3_grad = (max_c3_nig - max_c3_day) / nig_sza
    max_c3 = max_c3_day + c3_grad * sza_thr

    c1 = comp_stat(bt_mir, min_c1, max_c1)
    c2 = comp_stat(z4, min_c2, max_c2)
    c3 = comp_stat(zdt, min_c3, max_c3)

    c4 = 1 - comp_stat(ncld / (nwin / 2.), 0, 1)
    c5 = 1 - comp_stat(nwat / (nwin / 2.), 0, 1)

    outarr_conf = np.power(c1 * c2 * c3 * c4 * c5, 1./5.)

    return outarr_conf


def make_output_scene(data_dict):
    """Create a new satpy Scene from a dict of datasets."""
    scn = Scene()
    for ds in data_dict.keys():
        scn[ds] = data_dict[ds]
    return scn
