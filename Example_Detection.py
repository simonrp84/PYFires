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

from pyfires.PYF_detection import run_dets
from pyfires.PYF_basic import *

from tqdm import tqdm
from glob import glob
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
             #   'vi2_band': 'B06',
             'mir_band': 'B07',
             'lwi_band': 'B13'}

    # Load the data
    data_dict, s1, s2, sr1, sr2 = initial_load(ifiles_l15,  # List of files to load
                                               'ahi_hsd',  # The reader to use, in this case the AHI HSD reader
                                               bdict)  # The bands to load

    return run_dets(data_dict)


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
        visualize([prof, rprof], show=False, save=True, filename=odir + "../frp_vis.html")
        break
