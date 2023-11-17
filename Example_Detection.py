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

import dask

dask.config.set(num_workers=8)

import satpy

satpy.config.set({'cache_dir': "D:/sat_data/cache/"})
satpy.config.set({'cache_sensor_angles': False})
satpy.config.set({'cache_lonlats': True})

from pyfires.PYF_detection import stage1_tests, run_basic_night_detection
from pyfires.PYF_WindowStats import get_mea_std_window
from pyfires.PYF_basic import *

import dask.array as da
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
                             bdict,  # The bands to load
                             mir_bt_thresh=270,  # The threshold for the MIR band, pixels cooler than this are excluded.
                             lw_bt_thresh=260)  # The threshold for the LWIR band, pixels cooler than this are excluded.

    # Select potential fire pixels using the Roberts + Wooster Stage 1 + 2 tests
    data_dict['PFP'] = stage1_tests(data_dict['MIR__BT'],
                                    data_dict['BTD'],
                                    data_dict['VI1_DIFF'],
                                    data_dict['SZA'],
                                    data_dict['LSM'],
                                    ksizes=[5, 7, 9],
                                    do_lsm_mask=True)

    opts = {'def_fire_rad_vis': 0.5,
            'def_fire_rad_vid': 0.1,
            'sza_thresh': 97,
            'vid_thresh': 0.02
            }

    night_res = run_basic_night_detection(data_dict['VI2_RAD'],
                                          data_dict['SZA'],
                                          data_dict['VI1_DIFF'],
                                          data_dict['PFP'],
                                          opts)

    data_dict['VI1_DIFF_2'] = vid_adjust_sza(data_dict['VI1_DIFF'], data_dict['SZA'])

    dets_arr = (data_dict['VI1_DIFF_2'] > 0).astype(np.uint8)
    dets_arr.data = xr.where(data_dict['PFP'] > 0, dets_arr.data, 0)

    # This section computes windowed statistics around each candidate fire pixel.
    wrap_get_mean_std = dask.delayed(get_mea_std_window)
    outa = wrap_get_mean_std(dets_arr.data,  # Potential fire pixel candidates
                             data_dict['VI1_RAD'].data,  # VIS chan
                             data_dict['mi_ndfi'].data,  # NDFI
                             data_dict['LW1__BT'].data,  # LW Brightness Temperature
                             data_dict['BTD'].data,  # MIR-LW BTD
                             data_dict['MIR__BT'].data,  # MIR BT
                             data_dict['VI1_DIFF'].data,  # MIR-LWIR-VIS radiance diff
                             data_dict['LSM'].data,  # The land-sea mask
                             255,  # The value denoting land in the LSM. If 255, ignore mask
                             25)  # The maximum window size in pixels

    outan = da.from_delayed(outa, shape=(16, data_dict['BTD'].shape[0], data_dict['BTD'].shape[1]), dtype=np.single)

    # Get the results of the windows statistics code
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

    vis_window_arr = data_dict['VI1_RAD'] - (mean_vi + 1.5 * std_vi)
    ndfi_window_arr = data_dict['mi_ndfi'] - (mean_nd + 1.5 * std_nd)
    btd_window_arr = data_dict['BTD'] - (mean_btd + 1.5 * std_btd)
    mir_window_arr = data_dict['MIR__BT'] - (mean_mir + 1.5 * std_mir)
    visdif_window_arr = data_dict['VI1_DIFF'] - (mean_vid + 1.5 * std_vid)

    x1 = 0.02
    x2 = 8.

    t1 = 270
    t2 = 300

    grad = (x2 - x1) / (t2 - t1)

    lw_std_mult = grad * (data_dict['LW1__BT'] - t1)
    lw_std_mult = xr.where(lw_std_mult > x2, x2, lw_std_mult)
    lw_std_mult = xr.where(lw_std_mult < x1, x1, lw_std_mult)

    lw_window_arr = mean_lw - lw_std_mult * std_lw

    gd_t4 = mean_mir + 2 * std_mir
    gd_btd1 = mean_btd + 2.5
    gd_btd2 = mean_btd + 2 * std_btd

    lwbt_mult = (data_dict['LW1__BT'] - 270) / 30
    lwbt_mult = lwbt_mult.clip(0, 1)
    lwbt_mult = 1. / (0.25 + 0.75 * lwbt_mult)

    sza_mult = 0.4 + ((90 - data_dict['SZA']) / 90.)
    ndfi_thresh = 0.001 + ((0.006 * lwbt_mult) + (0.003 / perc_good)) * sza_mult

    dets_thresh = 10

    dets_outarr = (dets_arr > 0).astype(np.uint8)  # t1
    dets_outarr = dets_outarr + (ndfi_window_arr > ndfi_thresh).astype(np.uint8)  # t2
    dets_outarr = dets_outarr + (visdif_window_arr > 0).astype(np.uint8)  # t3
    dets_outarr = dets_outarr + (vis_window_arr < 0.1).astype(np.uint8)  # t4
    dets_outarr = dets_outarr + (data_dict['LW1__BT'] > 260).astype(np.uint8)  # t5
    dets_outarr = dets_outarr + (mir_window_arr > 0).astype(np.uint8)  # t6
    dets_outarr = dets_outarr + (data_dict['LSM'] == 2).astype(np.uint8)  # t7
    dets_outarr = dets_outarr + (data_dict['MIR__BT'] > gd_t4).astype(np.uint8)  # t8
    dets_outarr = dets_outarr + (data_dict['BTD'] > gd_btd1).astype(np.uint8)  # t9
    dets_outarr = dets_outarr + (data_dict['BTD'] > gd_btd2).astype(np.uint8)  # t10
    dets_outarr = dets_outarr + (data_dict['LW1__BT'] > lw_window_arr).astype(np.uint8)  # t11
    #dets_outarr = xr.where(perc_good > 0.4, dets_outarr, 0)  # t12
    #dets_outarr = xr.where(dets_arr > 0, dets_outarr, 0)  # final

   # dets_outarr = xr.where(dets_outarr >= dets_thresh, dets_outarr, 0)  # final

    vi_diff_def_thr = 0.25

    fire_dets = xr.where(data_dict['VI1_DIFF'] > vi_diff_def_thr, dets_outarr + 1, dets_outarr)

    data_dict['fire_detection'] =  data_dict['LW1__BT'].copy()
    data_dict['fire_detection'].attrs['name'] = 'fire_detection'
    data_dict['fire_detection'].data = fire_dets

    conf_val = do_stage5(data_dict['BTD'],
                         mean_btd,
                         std_btd,
                         data_dict['MIR__BT'],
                         mean_mir,
                         std_mir,
                         data_dict['SZA'],
                         n_winpix, n_cloudpix,
                         n_waterpix)

    conf_val = xr.where(dets_outarr >= dets_thresh, conf_val, 0)

    data_dict['conf_val'] = data_dict['LW1__BT'].copy()
    data_dict['conf_val'].attrs['name'] = 'fire_confidence'
    data_dict['conf_val'].data = conf_val

    a_val = PYFc.rad_to_bt_dict[data_dict['pix_area'].attrs['platform_name']]
    frp_est = (data_dict['pix_area'] * PYFc.sigma / a_val) * (data_dict['MIR__BT'] - mean_mir)
    frp_est = xr.where(mean_mir > 0, frp_est, 0)
    frp_est = xr.where(dets_outarr >= dets_thresh, frp_est, 0)

    data_dict['frp_est'] = data_dict['LW1__BT'].copy()
    data_dict['frp_est'].attrs['name'] = 'frp_estimate'
    data_dict['frp_est'].attrs['units'] = 'MW'
    data_dict['frp_est'].data = frp_est

    scn = make_output_scene(data_dict)

    scn.save_datasets(datasets=['PFP', 'frp_est', 'fire_detection'], base_dir=out_dir, enhance=False, dtype=np.float32)


if __name__ == "__main__":
    # Specify the input and output directories for processing.
    # The input directory should contain all the AHI files without subdirectories.
    indir = 'D:/sat_data/ahi_main/in/'
    odir = 'D:/sat_data/ahi_main/out/'

    curfiles = glob(f'{indir}/0840/*B07*S01*.DAT', recursive=True)
    curfiles.sort()

    for curinf in tqdm(curfiles):
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
            main(curinf, odir)
        visualize([prof, rprof], show=False, save=True, filename=odir+"../frp_vis.html")
        break
