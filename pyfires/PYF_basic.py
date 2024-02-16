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
import dask.array
from satpy.modifiers.angles import _get_sensor_angles
from dask_image.ndfilters import convolve
from satpy import Scene, DataQuery
import pyfires.PYF_Consts as PYFc
from numba import prange

from pyspectral.rsr_reader import RelativeSpectralResponse
from satpy.modifiers.angles import _get_sun_angles
import xarray as xr
import numpy as np
import dask


def set_default_values(indict):
    """Set the default threshold, kernel and multiplier values for the fire detection."""
    indict['min_wsize'] = PYFc.min_win_size  # Minimum window size for windowed statistics
    indict['max_wsize'] = PYFc.max_win_size  # Maximum window size for windowed statistics
    indict['perc_thresh'] = PYFc.win_frac  # Percentage of good pixels in window for stats to be selected
    indict['lsm_val'] = PYFc.lsm_land_val  # Land-sea mask value for land
    indict['aniso_thresh'] = PYFc.aniso_thresh  # Anisotropy threshold for anisotropic diffusion test
    indict['vid2_thresh'] = PYFc.vid2_thresh  # Threshold for MIR - VIS - LWIR test
    indict['ksizes'] = PYFc.ksizes  # Kernel sizes for stage 1 tests
    indict['vid_stdm_mult'] = PYFc.vid_std_mult  # Multiplier factor for the VID radiance threshold
    indict['mir_stdm_mult'] = PYFc.mir_std_mult  # Multiplier factor for the MIR radiance threshold
    indict['kern_test_size'] = PYFc.kern_test_size  # Kernel size for the second stage kernel tests
    indict['iter_list'] = PYFc.aniso_iters  # Number of anisotropic diffusion iterations to perform
    indict['kern_thresh_btd'] = PYFc.kern_thresh_btd  # Threshold for the BTD kernel test
    indict['kern_thresh_sza_adj'] = PYFc.kern_thresh_sza_adj  # Threshold for the BTD kernel test
    indict['btddif_thresh'] = PYFc.btddif_thresh  # Threshold for the BTD difference test
    indict['viddif_thresh'] = PYFc.viddif_thresh  # Threshold for the VID difference test
    indict['mir_abs_thresh'] = PYFc.mir_abs_thresh  # Threshold for the MIR absolute test
    indict['min_btd_addback'] = PYFc.min_btd_addback  # Minimum BTD value for addback test
    indict['max_btd_addback'] = PYFc.max_btd_addback  # Maximum BTD value for addback test
    indict['main_perc_thresh'] = PYFc.main_perc_thresh  # Percentage threshold for main pixel window
    indict['sza_adj'] = PYFc.sza_adj  # The SZA threshold at which adjustment begins. Default value: 82 degrees.
    indict['sza_min_v'] = PYFc.sza_min_v  # The minimum VID value, used during day.
    indict['sza_max_v'] = PYFc.sza_max_v  # The maximum VID value, used at night.
    indict['sza_slo_str'] = PYFc.sza_slo_str  # The slope strength.
    indict['sza_slo_rise'] = PYFc.sza_slo_rise  # The slope rise.

    return indict


def vid_adjust_sza(in_vid, in_sza,
                   sza_adj=PYFc.sza_adj,
                   min_v=PYFc.sza_min_v,
                   max_v=PYFc.sza_max_v,
                   slo_str=PYFc.sza_slo_str,
                   slo_rise=PYFc.sza_slo_rise):
    """Adjust the VI Difference based on solar zenith angle.
    Inputs:
    - in_vid: The VI Difference data
    - in_sza: The solar zenith angle in degrees
    - sza_adj: The SZA threshold at which adjustment begins.
    - min_v: The minimum VID value, used during day.
    - max_v: The maximum VID value, used at night.
    - slo_str: The slope strength.
    - slo_rise: The slope rise.
    Returns:
    - adj_vid: The adjusted VI Difference data
     """

    adj_val = min_v + (max_v - min_v) * (1 / (1 + np.exp(-slo_str * (in_sza - sza_adj))) ** slo_rise)

    adj_vid = in_vid - adj_val
    return adj_vid


def conv_kernel(indata, ksize=5):
    """Convolve a kernel with a dataset."""

    kern = np.ones((ksize, ksize))
    kern = kern / np.sum(kern)
    res = convolve(indata, kern)

    return res


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


def save_output(inscn, indata, name, fname, ref='B07'):
    inscn[name] = inscn[ref].copy()
    inscn[name].attrs['name'] = name
    inscn[name].data = indata
    inscn.save_dataset(name, filename=fname, enhance=False, dtype=np.float32, fill_value=0)


def save_output_csv(indict, fname):
    """Save the output data to a CSV file."""
    minidict = {'LATS': indict['LATS'],
                'LONS': indict['LONS'],
                'frp_est': indict['frp_est'],
                'SZA': indict['SZA'],
                'MIR__BT': indict['MIR__BT'],
                'LW1__BT': indict['LW1__BT']}

    minidict = dask.array.compute(minidict)[0]
    with open(fname, 'w') as f:
        f.write('lat,lon,frp,sza,mir_bt,lw_bt\n')
        xs, ys = np.where(minidict["frp_est"] > 0)
        for x, y in zip(xs, ys):
            f.write(f'{minidict["LATS"][x, y]},'
                    f'{minidict["LONS"][x, y]},'
                    f'{minidict["frp_est"][x, y]},'
                    f'{minidict["SZA"][x, y]},'
                    f'{minidict["MIR__BT"][x, y]},'
                    f'{minidict["LW1__BT"][x, y]}\n')


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


def _get_band_solar(the_dict, bdict):
    """Compute the per-band solar irradiance for each loaded channel.
    Inputs:
    - the_dict: A dictionary containing bands to compute the irradiance for.
    - bdict: A dict of the band names and output names to compute the irradiance for.
    Returns:
    - irrad_dict: A dictionary containing the irradiance for each band.

    """
    from pyspectral.solar import SolarIrradianceSpectrum as SiS
    from pyspectral.rsr_reader import RelativeSpectralResponse

    irrad_dict = {}

    for band_name in bdict:
        sensor = the_dict['sensor']
        platform = the_dict['platform_name']
        srf = RelativeSpectralResponse(platform, sensor)
        cur_rsr = srf.rsr[bdict[band_name]]
        irrad = SiS().inband_solarirradiance(cur_rsr)
        irrad_dict[band_name] = irrad

    return irrad_dict


def compute_fire_datasets(indata_dict, irrad_dict, bdict, ref_band):
    """Compute the intermediate datasets used for fire detection.
    Inputs:
    - indata_dict: A dictionary containing the input datasets.
    - irrad_dict: A dictionary containing the solar irradiance values for each band.
    - bdict: Dictionary of bands to read from L1 files.
    Returns:
    - indata_dict: The input dictionary, with the computed datasets added.
    """
    indata_dict['BTD'] = indata_dict['MIR__BT'] - indata_dict['LW1__BT']

    sat = indata_dict['platform_name']
    sen = indata_dict['sensor']
    det = 'det-1'

    rsr = RelativeSpectralResponse(sat, sen)
    cur_rsr = rsr.rsr[bdict['mir_band']]
    wvl_mir = cur_rsr[det]['central_wavelength']

    exp_rad = calc_rad_fromtb(indata_dict['LW1__BT'], wvl_mir)
    mir_diffrad = indata_dict['MIR_RAD'] - exp_rad
    mir_diffrad = np.where(np.isfinite(mir_diffrad), mir_diffrad, 0)

    mir_noir_name = 'MIR_RAD_NO_IR'
    indata_dict[mir_noir_name] = mir_diffrad

    indata_dict['VI1_RAD'] = indata_dict['VI1_RAD'] * irrad_dict['mir_band'] / irrad_dict['vi1_band']

    indata_dict['VI1_DIFF'] = indata_dict[mir_noir_name] - indata_dict['VI1_RAD']

    indata_dict['mi_ndfi'] = (indata_dict['MIR__BT'] - indata_dict['LW1__BT']) / (
            indata_dict['MIR__BT'] + indata_dict['LW1__BT'])

    # Get the angles associated with the Scene
    indata_dict['SZA'], indata_dict['VZA'], indata_dict['RAA'], indata_dict['pix_area'] = get_angles(ref_band)
    indata_dict['glint_ang'] = calc_glint_ang(indata_dict)

    # Compute the adjusted VI difference, with reduced daytime VIS component
    adj_vid = vid_adjust_sza(indata_dict['VI1_DIFF'], indata_dict['SZA'])
    adj_vid = xr.where(np.isfinite(adj_vid), adj_vid, 0)
    indata_dict['VI1_DIFF_2'] = adj_vid

    # Get the latitudes, which are needed for contextually filtering background pixels
    lons, lats = ref_band.attrs['area'].get_lonlats_dask()
    indata_dict['LATS'] = lats.astype(np.float32)
    indata_dict['LATS'] = indata_dict['LATS'].rechunk(chunks=ref_band.chunks)
    indata_dict['LONS'] = lons.astype(np.float32)
    indata_dict['LONS'] = indata_dict['LONS'].rechunk(chunks=ref_band.chunks)

    final_bnames = ['VI1_RAD', 'MIR_RAD', 'LW1_RAD', 'MIR__BT', 'LW1__BT',
                    'MIR_RAD_NO_IR', 'VI1_DIFF', 'mi_ndfi']

    for bnum in prange(len(final_bnames)):
        band = final_bnames[bnum]
        indata_dict[band] = np.where(np.isfinite(indata_dict[band]), indata_dict[band], np.nan)

    return indata_dict


def py_aniso(in_img, pxv, nxv, pyv, nyv,
             niter=5,
             kappa=20,
             gamma=0.2):
    """A Python implementation of the anisotropic diffusion of a pixel.
    Inputs:
    - in_img: The input image to work on (float32 array)
    - niter: The number of iterations to perform
    - kappa: The conduction coefficient
    - gamma: The step value, suggested maximum is 0.25
    Returns:
    - img_diff: The diffused image (float32 array)
    """

    out_img = np.zeros_like(in_img)
    out_img[:, :] = in_img[:, :]

    for ii in range(0, niter):
        pxv[1:-1, :] = np.abs(out_img[2:, :] - out_img[1:-1, :])
        nxv[1:-1, :] = np.abs(out_img[0:-2, :] - out_img[1:-1, :])
        pyv[:, 1:-1] = np.abs(out_img[:, 2:] - out_img[:, 1:-1])
        nyv[:, 1:-1] = np.abs(out_img[:, 0:-2] - out_img[:, 1:-1])

        pxv = pxv * np.exp(-pxv ** 2 / kappa ** 2)
        nxv = nxv * np.exp(-nxv ** 2 / kappa ** 2)
        pyv = pyv * np.exp(-pyv ** 2 / kappa ** 2)
        nyv = nyv * np.exp(-nyv ** 2 / kappa ** 2)

        out_img = out_img + gamma * (pxv + nxv + pyv + nyv)

    return out_img


def get_aniso_diffs(vid_ds, niter_list):
    """Get the standard deviation in anisotropic diffusion of the VI Difference data.
    Inputs:
    - vid_ds: The VI Difference data.
    - niter_list: A list of the number of iterations to perform.
    Returns:
     - out_ds: The anisotropic diffusion of the VI Difference data.
    """

    out_list = [None] * (len(niter_list) + 1)
    out_list[0] = vid_ds

    pxv = np.zeros_like(vid_ds)
    nxv = np.zeros_like(vid_ds)
    pyv = np.zeros_like(vid_ds)
    nyv = np.zeros_like(vid_ds)

    for niter in range(0, len(niter_list)):
        out_list[niter + 1] = py_aniso(out_list[niter], pxv, nxv, pyv, nyv,
                                       niter=niter_list[niter] - niter_list[niter - 1],
                                       kappa=1)

    main_n = np.dstack(out_list)
    return np.nanstd(main_n, axis=2)


def get_aniso_diffs_bk(vid_ds, niter_list):
    """Get the standard deviation in anisotropic diffusion of the VI Difference data.
    Inputs:
    - vid_ds: The VI Difference data.
    - niter_list: A list of the number of iterations to perform.
    Returns:
     - out_ds: The anisotropic diffusion of the VI Difference data.
    """

    out_list = [None] * (len(niter_list) + 1)
    out_list[0] = vid_ds

    pxv = np.zeros_like(vid_ds)
    nxv = np.zeros_like(vid_ds)
    pyv = np.zeros_like(vid_ds)
    nyv = np.zeros_like(vid_ds)

    for niter in prange(0, len(niter_list)):
        out_list[niter + 1] = py_aniso(vid_ds, pxv, nxv, pyv, nyv, niter=niter_list[niter], kappa=1)

    main_n = np.dstack(out_list)
    return np.nanstd(main_n, axis=2)


def calc_glint_ang(indi):
    """Calculate the glint angle for a dataset."""

    gang = np.cos(np.deg2rad(indi['SZA'])) * np.cos(np.deg2rad(indi['VZA']))
    gang = gang - np.sin(np.deg2rad(indi['SZA'])) * np.sin(np.deg2rad(indi['VZA'])) * np.cos(np.deg2rad(indi['RAA']))
    return np.rad2deg(np.arccos(gang))


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
    saa, sza = _get_sun_angles(ref_data)

    # Satellite zenith
    vaa, vza = _get_sensor_angles(ref_data)

    # Relative azimuth
    raa = np.absolute(saa - vaa)
    raa = np.minimum(raa, 360 - raa)

    # Compute the pixel area
    # Pixel sizes are in meters, convert to km
    pix_size = ref_data.attrs['area'].pixel_size_x * ref_data.attrs['area'].pixel_size_x * 1e-6

    # Multiply by inverse vza to gain estimate of pixel size across image.
    pix_area = pix_size / np.cos(np.deg2rad(vza))

    # Return values, casting to float32 to save memory.
    return (sza.data.astype(np.float32), vza.data.astype(np.float32),
            raa.data.astype(np.float32), pix_area.data.astype(np.float32))


def initial_load(infiles_l1,
                 l1_reader,
                 bdict,
                 do_load_lsm=True,
                 bbox=None, ):
    """Read L1 data from disk based on user preferences.
    Inputs:
     - infiles_l1: List of L1 files to read.
     - l1_reader: Satpy reader to use for L1 files.
     - rad_dict: Dictionary of solar irradiance coefficients.
     - bdict: Dictionary of bands to read from L1 files.
     - do_load_lsm: Boolean, whether to load a land-sea mask (default: True).
     - bbox: Optional, a bounding box in x/y coordinates. If not given, full disk will be processed.
    Returns:
     - scn: A satpy Scene containing the data read from disk.
    """

    # Construct queries
    vi1_rad = DataQuery(name=bdict['vi1_band'], calibration="radiance")
    vi2_rad = DataQuery(name=bdict['vi2_band'], calibration="radiance")
    mir_rad = DataQuery(name=bdict['mir_band'], calibration="radiance")
    mir_bt = DataQuery(name=bdict['mir_band'], calibration="brightness_temperature")
    lwi_rad = DataQuery(name=bdict['lwi_band'], calibration="radiance")
    lwi_bt = DataQuery(name=bdict['lwi_band'], calibration="brightness_temperature")

    blist = [vi1_rad, vi2_rad, mir_rad, lwi_rad, mir_bt, lwi_bt]

    scn = Scene(infiles_l1, reader=l1_reader)
    scn.load(blist, calibration='radiance', generate=False)

    if bbox:
        scn = scn.crop(xy_bbox=bbox)

    scnr = scn.resample(scn.coarsest_area(), resampler='native')

    return sort_l1(scnr[vi1_rad],
                   scnr[vi2_rad],
                   scnr[mir_rad],
                   scnr[lwi_rad],
                   scnr[mir_bt],
                   scnr[lwi_bt],
                   bdict,
                   do_load_lsm=do_load_lsm)


def sort_l1(vi1_raddata,
            vi2_raddata,
            mir_raddata,
            lw1_raddata,
            mir_btdata,
            lw1_btdata,
            bdict,
            do_load_lsm=True):
    data_dict = {'VI1_RAD': vi1_raddata.data,
                 'VI2_RAD': vi2_raddata.data,
                 'MIR_RAD': mir_raddata.data,
                 'LW1_RAD': lw1_raddata.data,
                 'MIR__BT': mir_btdata.data,
                 'LW1__BT': lw1_btdata.data,
                 'platform_name': vi1_raddata.attrs['platform_name'],
                 'sensor': vi1_raddata.attrs['sensor']}

    # Get common attributes

    # Compute the solar irradiance values
    irrad_dict = _get_band_solar(data_dict, bdict)

    # Compute the datasets required for fire detection.
    data_dict = compute_fire_datasets(data_dict, irrad_dict, bdict, lw1_raddata)

    # Lastly, load the land-sea mask
    if do_load_lsm:
        data_dict['LSM'] = load_lsm(vi1_raddata)
    else:
        lsm = dask.array.zeros_like(vi1_raddata)
        lsm[:, :] = PYFc.lsm_land_val
        data_dict['LSM'] = lsm.astype(np.uint8)

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
    lonstr = f'{abs(lon):05}'.replace('.', '')

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

    # The LSM is often loaded with differing chunk sizes to the input data, so rechunk to match.
    lsm_data.data = lsm_data.data.rechunk(ds.data.chunks)
    lsm_data.coords['x'] = ds.coords['x']
    lsm_data.coords['y'] = ds.coords['y']

    return lsm_data.data


def make_output_scene(data_dict):
    """Create a new satpy Scene from a dict of datasets."""
    scn = Scene()
    for ds in data_dict.keys():
        if type(data_dict[ds]) is xr.DataArray:
            scn[ds] = data_dict[ds]
    return scn


def calc_frp(data_dict):
    """Calculate the fire radiative power for candidate fire pixels.
    Inputs:
     - data_dict: A dict containing the mask of candidates, plus contextual window stats.
    Returns:
     - data_dict: Updated dict now also containing the FRP estimates.
    """

    a_val = PYFc.rad_to_bt_dict[data_dict['platform_name']]
    frp_est = (data_dict['pix_area'] * PYFc.sigma / a_val) * (data_dict['MIR__BT'] - data_dict['mean_mir'])
    frp_est = xr.where(data_dict['mean_mir'] > 0, frp_est, 0)
    frp_est = xr.where(data_dict['fire_dets'] > 0, frp_est, 0)

    data_dict['frp_est'] = frp_est

    return data_dict
