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

import dask

dask.config.set(num_workers=8)

import satpy
satpy.config.set({'cache_dir': "/data/cache/"})
satpy.config.set({'cache_sensor_angles': False})
satpy.config.set({'cache_lonlats': True})


from pyfires.PYF_detection import stage1_tests, run_basic_night_detection
from pyfires.PYF_WindowStats import get_mea_std_window
from pyfires.PYF_basic import *

import dask.array as da
from tqdm import tqdm
from glob import glob
import os

import warnings
warnings.filterwarnings('ignore')


def main(curfile, out_dir):

    pos = curfile.find('B07')
    dtstr = curfile[pos-14:pos-1]
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
    scn = initial_load(ifiles_l15,  # List of files to load
                       'ahi_hsd',   # The reader to use, in this case the AHI HSD reader
                       bdict,       # The bands to load
                       mir_bt_thresh=270, # The threshold for the MIR band, pixels cooler than this are excluded.
                       lw_bt_thresh=260) # The threshold for the LWIR band, pixels cooler than this are excluded.

    # Select potential fire pixels using the Roberts + Wooster Stage 1 + 2 tests
    scn = stage1_tests(scn, ksizes=[5, 7, 9], do_lsm_mask=False)

    opts = {'def_fire_rad_vis': 0.5,
            'def_fire_rad_vid': 0.1,
            'sza_thresh': 97,
            'vid_thresh': 0.02
            }

    results = run_basic_night_detection(scn, opts)

    # This section computes windowed statistics around each candidate fire pixel.
    wrap_get_mean_std = dask.delayed(get_mea_std_window)
    outa = wrap_get_mean_std(results.astype(np.uint8),  # Potential fire pixel candidates
                             scn['VI1_RAD'].data,  # VIS chan
                             scn['mi_ndfi'].data,  # NDFI
                             scn['LW1__BT'].data,  # LW Brightness Temperature
                             scn['BTD'].data,  # MIR-LW BTD
                             scn['MIR__BT'].data,  # MIR BT
                             scn['VI1_DIFF'].data,  # MIR-LWIR-VIS radiance diff
                             scn['LSM'].data,  # The land-sea mask
                             255,  # The value denoting land in the LSM. If 255, ignore mask
                             25)

    outan = da.from_delayed(outa, shape=(16, scn['BTD'].shape[0], scn['BTD'].shape[1]), dtype=np.single)

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

    # Apply some additional thresholding to remove false positives due to VIS channel noise
    results = da.where(scn['BTD'].data > mean_btd + 1.5 * std_btd, results, 0)
    results = da.where(scn['VI1_DIFF'].data > mean_vid + 1.5 * std_vid, results, 0)

    # Compute the FRP estimate
    a_val = PYFc.rad_to_bt_dict[scn['pix_area'].attrs['platform_name']]
    frp_est = (scn['pix_area'] * PYFc.sigma / a_val) * (scn['MIR__BT'] - mean_mir)
    frp_est = np.where(mean_mir > 0, frp_est, 0)

    scn['frp_est'] = scn['VI1_DIFF'].copy()
    scn['frp_est'].attrs['name'] = 'frp_estimate'
    scn['frp_est'].attrs['units'] = 'MW'
    scn['frp_est'].data = frp_est

    scn.save_datasets(datasets=['LWIR_BT_RAW', 'frp_est'], base_dir=out_dir, enhance=False, dtype=np.float32)


if __name__ == "__main__":
    # Specify the input and output directories for processing.
    # The input directory should contain all the AHI files without subdirectories.
    indir = '/data/ahi_in/'
    odir = '/data/ahi_out/'

    curfiles = glob(f'{indir}/*/*B07*S01*.DAT', recursive=True)

    for curinf in tqdm(curfiles):
        main(curinf, odir)
