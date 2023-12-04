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

"""An example script showing how to detect fires using pyfires and Himawari/AHI data."""

from dask.diagnostics import Profiler, ResourceProfiler, visualize
from dask_image.ndfilters import convolve, maximum_filter

import dask

dask.config.set(num_workers=8)

import satpy

satpy.config.set({'cache_dir': "D:/sat_data/cache/"})
satpy.config.set({'cache_sensor_angles': False})
satpy.config.set({'cache_lonlats': True})

from pyfires.PYF_WindowStats import get_mea_std_window, get_local_stats
from pyfires.PYF_detection import stage1_tests
from pyfires.PYF_basic import *

from tqdm import tqdm
from glob import glob
import xarray as xr
import os

import warnings

warnings.filterwarnings('ignore')


def main(curfile, out_dir):
    pos = curfile.find('B07')
    dtstr = curfile[pos - 14:pos - 1]
    # if os.path.exists(f'{out_dir}/frp_estimate_{dtstr}00.tif'):
    #    print(f'Already processed {dtstr}')
    #    return
    # else:
    #    print("Processing", f'{out_dir}/frp_estimate_{dtstr}00.tif')
    ifiles_l15 = glob(f'{os.path.dirname(curfile)}/*{dtstr}*.DAT')

    # The bands used for processing:
    # VI1 should be a red band close to 0.6 micron.
    # VI2 should be a near-infrared band close to 2.2 micron, or 1.6 micron if 2.2 is unavailable.
    # MIR should be the mid-IR band closest to 3.8 micron.
    # LWI should be a longwave window channel such as 10.8 micron.
    bdict = {'vi1_band': 'B03',
             'vi2_band': 'B06',
             'mir_band': 'B07',
             'lwi_band': 'B13'}

    # Load the data
    data_dict = initial_load(ifiles_l15,  # List of files to load
                             'ahi_hsd',  # The reader to use, in this case the AHI HSD reader
                             bdict)  # The bands to load

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
    outa = wrap_get_mean_std(data_dict['PFP'].data,
                             data_dict['VI1_RAD'].data,  # VIS chan
                             data_dict['mi_ndfi'].data,  # NDFI
                             data_dict['LW1__BT'].data,  # LW Brightness Temperature
                             data_dict['BTD'].data,  # MIR-LW BTD
                             data_dict['MIR__BT'].data,  # MIR BT
                             data_dict['VI1_DIFF'].data,  # MIR-LWIR-VIS radiance diff
                             data_dict['LSM'].data,  # The land-sea mask
                             data_dict['LATS'].data,  # The pixel longitudes
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
    fir_d_sum = convolve(main_det_arr.data, kern)
    local_max = maximum_filter(data_dict['VI1_DIFF'].data, (3, 3))
    tmp_out = (fir_d_sum == 1) * (data_dict['VI1_DIFF'] == local_max)

    main_out = main_det_arr * (fir_d_sum > 1) + tmp_out * main_det_arr

    main_out = main_out * xr.where(data_dict['MIR__BT'] > mean_mir + 2 * std_mir, main_out, 0)
    main_out = main_out * xr.where(data_dict['BTD'] > mean_btd + 2.5, main_out, 0)
    main_out = main_out * xr.where(data_dict['BTD'] > mean_btd + 2 * std_btd, main_out, 0)

    fir_d_sum = convolve(main_out.data, kern)
    local_max = maximum_filter(data_dict['MIR__BT'].data, (3, 3))
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

    resx = np.abs(convolve(data_dict['MIR__BT'].data, kern))
    kern = kern.T
    resy = np.abs(convolve(data_dict['MIR__BT'].data, kern))
    res = np.sqrt(resx * resx + resy * resy)
    main_out = main_out * (res < 500)


    delayed_local_stats = dask.delayed(get_local_stats)
    locarr = delayed_local_stats(main_out.data,
                                 data_dict['MIR__BT'].data,
                                 data_dict['BTD'].data,
                                 data_dict['VI1_DIFF'].data)
    locarrn = dask.array.from_delayed(locarr,
                                      shape=(data_dict['BTD'].shape[0], data_dict['BTD'].shape[1], 3),
                                      dtype=np.single)
    mirdif = locarrn[:, :, 0]
    btddif = locarrn[:, :, 1]
    viddif = locarrn[:, :, 2]

    main_out = main_out * (btddif > 1) * (viddif > 0.04)
    main_out = main_out * (data_dict['BTD'] > mean_btd + std_mir + std_btd)

    kern = np.ones((3, 3))
    fir_d_sum = convolve(main_out.data, kern)
    local_max = maximum_filter(data_dict['VI1_DIFF'].data, (3, 3))
    out5 = (fir_d_sum == 1) * (data_dict['VI1_DIFF'] == local_max)
    main_out = main_out * (fir_d_sum > 1) + out5 * main_out

    kern_ones = np.ones((3, 3))
    fir_d_sum = convolve(main_out.data, kern_ones)
    local_max = maximum_filter(data_dict['MIR__BT'].data, (3, 3))
    out5 = (fir_d_sum == 1) * (data_dict['MIR__BT'] == local_max)
    main_out = main_out * (fir_d_sum > 1) + out5 * main_out

    # Absolute MIR BT threshold before a pixel is declared 'fire'
    mir_abs_thresh = 350
    # BTD thresh for adding back missing pixels
    min_btd_addback = 2
    max_btd_addback = 15

    main_out_tmp = main_out + xr.where(data_dict['MIR__BT'] > mir_abs_thresh, 1, 0).astype(np.uint8)
    main_out_tmp = xr.where(main_out_tmp > 0, 1, 0).astype(np.uint8)

    fir_d_sum = convolve(main_out_tmp.data, kern_ones)

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

    data_dict['fire_dets'] = data_dict['LW1__BT'].copy()
    data_dict['fire_dets'].attrs['name'] = 'fire_dets'
    data_dict['fire_dets'].attrs['units'] = ''
    data_dict['fire_dets'].data = main_out

    data_dict = calc_frp(data_dict)

    return data_dict['fire_dets'], data_dict['frp_est']


if __name__ == "__main__":
    # Specify the input and output directories for processing.
    # The input directory should contain all the AHI files without subdirectories.
    indir = 'D:/sat_data/ahi_main/in/'
    odir = 'D:/sat_data/ahi_main/out/'

    curfiles = glob(f'{indir}/1650/*B07*S01*.DAT', recursive=True)
    curfiles.sort()

    for curinf in tqdm(curfiles):
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
            fire_dets, frp_est = main(curinf, odir)
        visualize([prof, rprof], show=False, save=True, filename=odir+"../frp_vis.html")
        break
