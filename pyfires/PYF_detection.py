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

"""Functions for detecting fires in satellite data."""

import pyfires.PYF_Consts as PYFc
from dask_image.ndfilters import convolve, maximum_filter
from pyfires.PYF_WindowStats import get_mea_std_window, get_local_stats
from pyfires.PYF_basic import get_aniso_diffs, calc_frp
import dask.array as da
import numpy as np
import xarray as xr
import dask


def _make_kern(ksize):
    """Make a high pass kernel of given size.
    Inputs:
    - ksize: The size of the kernel, which will be square: (ksize, ksize)
    Returns:
    - arr: The kernel array.
    """
    if ksize == 3:
        arr = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    elif ksize == 5:
        arr = np.array([[-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, 25, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1]])
    elif ksize == 7:
        arr = np.array([[-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, 49, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1], ])
    elif ksize == 9:
        arr = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, 81, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1], ])
    elif ksize == 11:
        arr = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, 121, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    else:
        raise ValueError()
    return arr / np.sum(arr)


def set_initial_thresholds(sza,
                           mir_thresh_bt=PYFc.mir_thresh_bt, mir_thresh_sza_adj=PYFc.mir_thresh_sza_adj,
                           mir_thresh_limit=PYFc.mir_thresh_limit,
                           btd_thresh_bt=PYFc.btd_thresh_bt, btd_thresh_sza_adj=PYFc.btd_thresh_sza_adj,
                           btd_thresh_limit=PYFc.btd_thresh_limit):
    """Set the stage 1 thresholds for MIR and LWIR BTs
    Inputs:
    - sza: Solar zenith angle array / dataset.
    - mir_thresh_bt: The MIR BT threshold as defined in Roberts + Wooster. Default value: 310.5 K
    - mir_thresh_sza_adj: MIR BT threshold adjustment factor for SZA. Default value: -0.3 K/deg
    - mir_thresh_limit: Minimum acceptable MIR BT threshold. Default value: 280 K
    - btd_thresh_bt: The BTD threshold as defined in Roberts + Wooster. Default value: 1.75 K
    - btd_thresh_sza_adj: BTD threshold adjustment factor for SZA. Default value: -0.0049 K/deg
    - btd_thresh_limit: Minimum acceptable BTD threshold. Default value: 1 K
    Returns:
    - mir_thresh: The per-pixel MIR BT threshold
    - btd_thresh: The per-pixel BTD threshold

    """
    mir_thresh = mir_thresh_bt + mir_thresh_sza_adj * sza
    mir_thresh = xr.where(mir_thresh < mir_thresh_limit, mir_thresh_limit, mir_thresh)
    btd_thresh = btd_thresh_bt + btd_thresh_sza_adj * sza
    btd_thresh = xr.where(btd_thresh < btd_thresh_limit, btd_thresh_limit, btd_thresh)

    return mir_thresh, btd_thresh


def do_apply_stg1b_kern2(in_variable, ksize):
    """Convolve a high pass kernel of requested size with the input dataset.
    Inputs:
    - in_variable: The input data.
    - ksize: The size of the kernel, which will be square: (ksize, ksize)
    Returns:
    - out_ker: The convolved dataset.
    - out_std: The standard deviation of the convolved dataset.
    """

    out_ker = convolve(in_variable, _make_kern(ksize))
    out_std = da.nanstd(out_ker)

    return out_ker, out_std


def stage1_tests(in_mir,
                 in_btd,
                 in_vid,
                 in_sza,
                 in_lsm,
                 ksizes=[3, 5, 7],
                 kern_thresh_btd=PYFc.kern_thresh_btd,
                 kern_thresh_sza_adj=PYFc.kern_thresh_sza_adj,
                 do_lsm_mask=True,
                 lsm_land_val=PYFc.lsm_land_val):
    """Perform the stage 1a + 1b tests from Roberts + Wooster.
    Inputs:
    - in_mir: The MIR channel data (K)
    - in_btd: The BTD between MIR and LWIR (K)
    - in_vid: The MIR - VIS - LWIR radiance data
    - in_sza: The solar zenith angle (degrees)
    - in_lsm: The land-sea mask (2 = land, 1 = coast, 0 = sea)
    - ksizes: A list of kernel sizes to apply. By default, 3, 5 and 7 pixel width, as in Roberts + Wooster.
    - kern_thresh_btd: The BTD threshold as defined in Roberts + Wooster. Default value: 1.5 K
    - kern_thresh_sza_adj: BTD threshold adjustment factor for SZA. Default value: -0.012 K/deg
    - do_lsm_mask: Whether to apply the land-sea mask. Default value: True
    Returns:
    - A boolean mask with True for pixels that pass the tests
    """

    btd_kern_thr = kern_thresh_btd + kern_thresh_sza_adj * in_sza

    mir_thresh, btd_thresh = set_initial_thresholds(in_sza)
    main_testarr = da.zeros_like(in_mir)

    # This is stage 1b, applied before 1a to simplify processing.
    for ksize in ksizes:
        kerval, stdval = do_apply_stg1b_kern2(in_btd, ksize)
        tmpdata = xr.where(kerval >= stdval * btd_kern_thr, 1, 0)
        main_testarr = xr.where(tmpdata > 0, main_testarr + 1, main_testarr)

    main_testarr = xr.where(in_mir >= mir_thresh, main_testarr + 1, 0)
    main_testarr = xr.where(in_btd >= btd_thresh, main_testarr + 1, 0)

    # Apply land-sea mask
    if do_lsm_mask:
        main_testarr = xr.where(in_lsm == lsm_land_val, main_testarr, 0)

    # Only select pixels with positive MIR radiance after VIS and IR subtractions.
    main_testarr = xr.where(in_vid >= 0, main_testarr, 0)

    pfp_arr = xr.where(main_testarr >= PYFc.stage1_pass_thresh, 1, 0).astype(np.uint8)

    # Return only those pixels meeting test threshold
    return pfp_arr


def compute_background_rad(indata,
                           insza,
                           sza_thr=97,
                           rad_sum_thr=0.99,
                           hist_range=(0, 2)):
    """Compute the threshold background radiance for the VIS channel at night.
    This is used to detect bright nighttime pixels.

    Inputs:
     - indata: The VIS channel radiances
     - insza: The solar zenith angle
     - sza_thr: The SZA threshold to use for night detection. Default value: 97 degrees.
     - rad_sum_thr: The threshold for cumulative sum of all pixels, 99.9% by default.
     - hist_range: The range of radiance values to use for the histogram. Default value: (0, 2)
    Returns:
     - The required radiance threshold.
    """

    cum_sum = 0

    # Select only night pixels
    tmp = indata.ravel()
    tmp = tmp[insza.ravel() > sza_thr]
    # Select only valid radiance values
    tmp = tmp[tmp > 0]

    # Compute a histogram of values for binning
    hist, bins = np.histogram(tmp, bins=800, range=hist_range)
    hist = hist / np.sum(hist)

    if np.nanmax(hist) <= 0 or np.isnan(np.nanmax(hist)):
        return -999

    # Loop over histogram to find where threshold cumulative count is exceeded
    i = 0
    for i in range(0, len(hist)):
        cum_sum = cum_sum + hist[i]
        if cum_sum > rad_sum_thr:
            break

    # Return the radiance value for this bin.
    return bins[i]


def run_basic_night_detection(in_vi2_rad,
                              in_sza,
                              in_vid,
                              in_pfp,
                              opts={'def_fire_rad_vis': 0.5,
                                    'def_fire_rad_vid': 0.5,
                                    'sza_thresh': 97,
                                    'vid_thresh': 0.02}):
    """Detect large fires at night using PFP, VIS2.2 and VISDIF channels.
    Inputs:
    - inscn: The input scene containing SZA, PFP, VIS and VISDIF data.
    - opts: A dict containing optional parameters for the detection:
        sza_thresh: The SZA threshold to use for night detection. Default value: 97 degrees.
        vid_thresh: The VISDIF threshold to use for night detection. Default value: 0.02 W m-2 um-1 sr-1
        def_fire_rad_vis: Visible radiance used to assume detection of fire even if PFP is False. Default: 0.5
        def_fire_rad_vid: VIS_MIR_LWIR radiance diff used to assume detection of fire even if PFP is False. Default: 0.5
        Note: def_fire_rad_vis and def_fire_rad_vid are used together, both must pass for a pixel to be flagged.
    Outputs:
    - out_dets: The detected fire pixels

    The `sza_thresh` value should be set so that only pixels with no solar contribution are included.
    The default value was chosen assuming the use of a 2.2 micron VIS channel. The method also works,
    with lower sensitivity, with the 1.6 micron channel but the SZA threshold may need to be adjusted
    in this case.
    Note: The 2.2 micron channel is optimal for fire detection. 1.6 micron is less sensitive.
    """

    # Compute the appropriate VIS radiance threshold
    if np.nanmax(in_sza > opts['sza_thresh']):
        thr_vis = compute_background_rad(in_vi2_rad, in_sza, sza_thr=opts['sza_thresh'])
    else:
        return np.zeros_like(in_vi2_rad)

    if thr_vis == -999:
        return np.zeros_like(in_vi2_rad)

    # Compute the definite nighttime detections
    def_dets = (in_vi2_rad > opts['def_fire_rad_vis']).astype(np.uint8)
    def_dets = xr.where(in_vid > opts['def_fire_rad_vid'], def_dets, 0)
    def_dets = xr.where(in_sza > opts['sza_thresh'], def_dets, 0)

    # Select only pixels with a bright VIS radiance
    out_dets = (in_vi2_rad > thr_vis * 2).astype(np.uint8)
    # Select only pixels with a suitable VISDIFF radiance
    out_dets = xr.where(in_vid > opts['vid_thresh'], out_dets, 0)
    # Exclude pixels that were not selected as potential fire pixels by the Roberts + Wooster tests
    out_dets = xr.where(in_pfp == 1, out_dets, 0)

    # Sometimes, hot VIS pixels are mistaken as fire. Here we attempt to exclude those pixels
    # by ensuring that the candidate has the highest VID in a 3x3 window.
    # Get sum in 3x3 window
    det_sum = convolve(out_dets, np.ones((3, 3)))
    # Get max in 3x3 window
    det_max = maximum_filter(in_vid, (3, 3))

    det_pos = xr.where(det_sum == 1, in_vid, 0)
    det_pos = xr.where(det_pos == det_max, 1, 0).astype(np.uint8)

    out_dets = xr.where(det_sum == 1, det_pos, out_dets)

    # Apply def dets
    out_dets = xr.where(def_dets > 0, 1, out_dets)

    # Only select night pixels
    out_dets = xr.where(in_sza > opts['sza_thresh'], out_dets, 0)

    return out_dets, def_dets



def run_dets(data_dict):

    # Select potential fire pixels using the Roberts + Wooster Stage 1 + 2 tests

    data_dict['PFP'] = stage1_tests(data_dict['MIR__BT'],
                                    data_dict['BTD'],
                                    data_dict['VI1_DIFF'],
                                    data_dict['SZA'],
                                    data_dict['LSM'],
                                    ksizes=[5, 7, 9],
                                    do_lsm_mask=True)

    # For the potential fire pixels previously defined, compute the per-pixel windows stats
    wrap_get_mean_std = dask.delayed(get_mea_std_window)
    outa = wrap_get_mean_std(data_dict['PFP'],
                             data_dict['VI1_RAD'],  # VIS chan
                             data_dict['mi_ndfi'],  # NDFI
                             data_dict['LW1__BT'],  # LW Brightness Temperature
                             data_dict['BTD'],  # MIR-LW BTD
                             data_dict['MIR__BT'],  # MIR BT
                             data_dict['VI1_DIFF'],  # MIR-LWIR-VIS radiance diff
                             data_dict['LSM'],  # The land-sea mask
                             data_dict['LATS'],  # The pixel latitudes
                             255,  # The value denoting land in the LSM. If 255, ignore mask
                             25)

    outan = dask.array.from_delayed(outa,
                                    shape=(16, data_dict['BTD'].shape[0], data_dict['BTD'].shape[1]),
                                    dtype=np.single)

    perc_good = outan[0, :, :]
    n_winpix = outan[1, :, :]
    n_cloudpix = outan[2, :, :]
    n_waterpix = outan[3, :, :]
    mean_lw = outan[4, :, :]
    std_lw = outan[5, :, :]
    mean_nd = outan[6, :, :]
    std_nd = outan[7, :, :]
    mean_vi = outan[8, :, :]
    std_vi = outan[9, :, :]
    mean_btd = outan[10, :, :]
    std_btd = outan[11, :, :]
    mean_mir = outan[12, :, :]
    std_mir = outan[13, :, :]
    mean_vid = outan[14, :, :]
    std_vid = outan[15, :, :]

    # Define some test thresholds for further selection of potential fire pixels
    vi1_diff_stdm = (data_dict['VI1_DIFF'] - mean_vid) / std_vid
    mir_bt_stdm = (data_dict['MIR__BT'] - mean_mir) / std_mir
    vi1_rad_stdm = (data_dict['VI1_RAD'] - mean_vi) / std_vi

    # Compute the anisotropic diffusion of the MIR band at three iteration levels
    iter_list = [1, 2, 3]
    wrap_get_aniso_diffs = dask.delayed(get_aniso_diffs)
    aniso_std = dask.array.from_delayed(wrap_get_aniso_diffs(data_dict['VI1_DIFF_2'],
                                                             iter_list),
                                                     shape=data_dict['VI1_DIFF_2'].shape,
                                                     dtype=np.single)

    # Some additional fire screening tests
    # Fire pixels will have a high anisotropic diffusion value compared to background
    main_det_arr = (aniso_std > 0.01).astype(np.uint8)
    # Fire pixels will also have a radiance compared to non-fire pixels in the MIR
    main_det_arr = main_det_arr * (data_dict['VI1_DIFF_2'] > -0.15)
    # Only select pixels that pass the Roberts + Wooster tests
    main_det_arr = main_det_arr * data_dict['PFP']

    main_det_arr = main_det_arr * (vi1_diff_stdm > vi1_rad_stdm * 1.5)
    main_det_arr = main_det_arr * (mir_bt_stdm > 1.5)

    kern = np.ones((3, 3))

    fir_d_sum = convolve(main_det_arr, kern)
    local_max = maximum_filter(data_dict['VI1_DIFF'], (3, 3))
    tmp_out = (fir_d_sum == 1) * (data_dict['VI1_DIFF'] == local_max)

    main_out = main_det_arr * (fir_d_sum > 1) + tmp_out * main_det_arr

    main_out = main_out * xr.where(data_dict['MIR__BT'] > mean_mir + 2 * std_mir, main_out, 0)
    main_out = main_out * xr.where(data_dict['BTD'] > mean_btd + 2.5, main_out, 0)
    main_out = main_out * xr.where(data_dict['BTD'] > mean_btd + 2 * std_btd, main_out, 0)

    fir_d_sum = convolve(main_out, kern)
    local_max = maximum_filter(data_dict['MIR__BT'], (3, 3))
    tmp_out = (fir_d_sum == 1) * (data_dict['MIR__BT'] == local_max)
    main_out = main_out * (fir_d_sum > 1) + main_out * tmp_out

    kern = np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                     [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]) / 5

    resx = np.abs(convolve(data_dict['MIR__BT'], kern))
    kern = kern.T
    resy = np.abs(convolve(data_dict['MIR__BT'], kern))
    res = np.sqrt(resx * resx + resy * resy)
    main_out = main_out * (res < 500)


    delayed_local_stats = dask.delayed(get_local_stats)
    locarr = delayed_local_stats(main_out,
                                 data_dict['MIR__BT'],
                                 data_dict['BTD'],
                                 data_dict['VI1_DIFF'])
    locarrn = dask.array.from_delayed(locarr,
                                      shape=(data_dict['BTD'].shape[0], data_dict['BTD'].shape[1], 3),
                                      dtype=np.single)
    mirdif = locarrn[:, :, 0]
    btddif = locarrn[:, :, 1]
    viddif = locarrn[:, :, 2]

    main_out = main_out * (btddif > 1) * (viddif > 0.04)
    main_out = main_out * (data_dict['BTD'] > mean_btd + std_mir + std_btd)

    kern = np.ones((3, 3))
    fir_d_sum = convolve(main_out, kern)
    local_max = maximum_filter(data_dict['VI1_DIFF'], (3, 3))
    out5 = (fir_d_sum == 1) * (data_dict['VI1_DIFF'] == local_max)
    main_out = main_out * (fir_d_sum > 1) + out5 * main_out

    kern_ones = np.ones((3, 3))
    fir_d_sum = convolve(main_out, kern_ones)
    local_max = maximum_filter(data_dict['MIR__BT'], (3, 3))
    out5 = (fir_d_sum == 1) * (data_dict['MIR__BT'] == local_max)
    main_out = main_out * (fir_d_sum > 1) + out5 * main_out

    # Absolute MIR BT threshold before a pixel is declared 'fire'
    mir_abs_thresh = 350
    # BTD thresh for adding back missing pixels
    min_btd_addback = 2
    max_btd_addback = 15

    main_out_tmp = main_out + xr.where(data_dict['MIR__BT'] > mir_abs_thresh, 1, 0).astype(np.uint8)
    main_out_tmp = xr.where(main_out_tmp > 0, 1, 0).astype(np.uint8)

    fir_d_sum = convolve(main_out_tmp, kern_ones)

    # Threshold for adding missing fire pixels, as the algorithm removes some pixels adjacent to existing detections
    # We add back using the BTD weighted by the number of fire pixels adjacent to the candidate.
    btd_addback_thresh = (9 - fir_d_sum) * (
                8 / (max_btd_addback - min_btd_addback)) + min_btd_addback + mean_btd + std_btd
    btd_addback_thresh = btd_addback_thresh * data_dict['PFP']

    main_out = main_out_tmp + xr.where(data_dict['BTD'] > btd_addback_thresh, 1, 0).astype(np.uint8)
    main_out = (xr.where(main_out > 0, 1, 0).astype(np.uint8) *
                xr.where(data_dict['PFP'] > 0, 1, 0).astype(np.uint8) *
                xr.where(fir_d_sum > 0, 1, 0).astype(np.uint8))

    data_dict['mean_mir'] = mean_mir
    data_dict['mean_btd'] = mean_btd
    data_dict['std_btd'] = std_btd

    data_dict['fire_dets'] = main_out

    data_dict = calc_frp(data_dict)

    return data_dict['fire_dets'], data_dict['frp_est']