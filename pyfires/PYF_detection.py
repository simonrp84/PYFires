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
from dask_image.ndfilters import convolve
import dask.array as da
import numpy as np


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
                           swir_thresh_bt=PYFc.swir_thresh_bt, swir_thresh_sza_adj=PYFc.swir_thresh_sza_adj,
                           swir_thresh_limit=PYFc.swir_thresh_limit,
                           btd_thresh_bt=PYFc.btd_thresh_bt, btd_thresh_sza_adj=PYFc.btd_thresh_sza_adj,
                           btd_thresh_limit=PYFc.btd_thresh_limit):
    """Set the stage 1 thresholds for SWIR and LWIR BTs
    Inputs:
    - sza: Solar zenith angle array / dataset.
    - swir_thresh_bt: The SWIR BT threshold as defined in Roberts + Wooster. Default value: 310.5 K
    - swir_thresh_sza_adj: SWIR BT threshold adjustment factor for SZA. Default value: -0.3 K/deg
    - swir_thresh_limit: Minimum acceptable SWIR BT threshold. Default value: 280 K
    - btd_thresh_bt: The BTD threshold as defined in Roberts + Wooster. Default value: 1.75 K
    - btd_thresh_sza_adj: BTD threshold adjustment factor for SZA. Default value: -0.0049 K/deg
    - btd_thresh_limit: Minimum acceptable BTD threshold. Default value: 1 K
    Returns:
    - swir_thresh: The per-pixel SWIR BT threshold
    - btd_thresh: The per-pixel BTD threshold

    """
    swir_thresh = swir_thresh_bt + swir_thresh_sza_adj * sza
    swir_thresh = da.where(swir_thresh < swir_thresh_limit, swir_thresh_limit, swir_thresh)
    btd_thresh = btd_thresh_bt + btd_thresh_sza_adj * sza
    btd_thresh = da.where(btd_thresh < btd_thresh_limit, btd_thresh_limit, btd_thresh)

    return swir_thresh, btd_thresh


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


def stage1_tests(in_mir, in_btd, in_vid, in_sza, in_lsm,
                 ksizes=[3, 5, 7],
                 kern_thresh_btd=PYFc.kern_thresh_btd,
                 kern_thresh_sza_adj=PYFc.kern_thresh_sza_adj):
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
    Returns:
    - A boolean mask with True for pixels that pass the tests
    """

    btd_kern_thr = kern_thresh_btd + kern_thresh_sza_adj * in_sza

    swir_thresh, btd_thresh = set_initial_thresholds(in_sza)
    main_testarr = da.zeros_like(in_mir)

    # This is stage 1b, applied before 1a to simplify processing.
    for ksize in ksizes:
        kerval, stdval = do_apply_stg1b_kern2(in_btd, ksize)
        tmpdata = da.where(kerval >= stdval * btd_kern_thr, 1, 0)
        main_testarr = da.where(tmpdata > 0, main_testarr + 1, main_testarr)

    main_testarr = da.where(in_mir >= swir_thresh, main_testarr + 1, 0)
    main_testarr = da.where(in_btd >= btd_thresh, main_testarr + 1, 0)

    # Apply land-sea mask
    main_testarr = da.where(in_lsm == 2, main_testarr, 0)

    # Only select pixels with positive SWIR radiance after VIS and IR subtractions.
    main_testarr = da.where(in_vid >= 0, main_testarr, 0)

    # Return only those pixels meeting test threshold.
    return da.where(main_testarr >= PYFc.stage1_pass_thresh, 1, 0).astype(np.uint8)

