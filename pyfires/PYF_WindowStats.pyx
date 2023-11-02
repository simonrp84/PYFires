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

"""Cython code for computing windowed statistics across all candidate fire pixels in a dataset."""

from libc.math cimport sqrt, isnan, isfinite
cimport numpy as np
import numpy as np
import cython

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef get_curwindow_pos(int xpos, int ypos, int winsize, int scn_width, int scn_height):
    """Find the correct position of the moving window in the input arrays.
    Inputs:
     - xpos: The x position of the candidate pixel
     - ypos: The y position of the candidate pixel
     - winsize: The radius of the window
     - scn_width: The width of the whole satellite scene
     - scn_height: The height of the whole satellite scene 
    Returns:
     - shape_sel: An array specifying the correct indices for the windowed region (x_0, x_1, y_0, y_1)
    """
    cdef int shape_sel[4]

    # Set up the output pixel selection
    shape_sel = [xpos - winsize, xpos + winsize + 1, ypos - winsize, ypos + winsize + 1]

    # Apply some range tests, so we can use cython in the non-bounds checking mode
    if shape_sel[0] < 0:
        shape_sel[0] = -999
    elif shape_sel[0] >= scn_width:
        shape_sel[0] = -999
    if shape_sel[1] < 0:
        shape_sel[0] = -999
    elif shape_sel[1] >= scn_height:
        shape_sel[0] = -999

    return shape_sel

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef get_window_mea_stdv(int winsize,
                         float[:,:] cur__vis,
                         float[:,:] cur_ndfi,
                         float[:,:] cur__lwi,
                         float[:,:] cur__btd,
                         float[:,:] cur__mir,
                         float[:,:] cur__vid,
                         unsigned char[:,:] cur__lsm,
                         float cen__mir,
                         float cen__btd,
                         unsigned char lsm_land_val = 2,
                         float mir_min_thresh = 275,
                         float mir_max_thresh = 340,
                         float btd_max_thresh = 7,
                         float lwi_min_thresh = 263.,
                         ):
    cdef int arrsize = winsize + winsize + 1
    cdef int npix = arrsize * arrsize
    cdef int i = 0
    cdef int j = 0

    cdef float n_good = 0
    cdef float perc_good = 0

    cdef float sum_val_vi = 0
    cdef float mea_val_vi = -999.
    cdef float std_val_vi = -999.

    cdef float sum_val_nd = 0
    cdef float mea_val_nd = -999.
    cdef float std_val_nd = -999.

    cdef float sum_val_lw = 0
    cdef float mea_val_lw = -999.
    cdef float std_val_lw = -999.

    cdef float sum_val_btd = 0
    cdef float mea_val_btd = -999.
    cdef float std_val_btd = -999.

    cdef float sum_val_mir = 0
    cdef float mea_val_mir = -999.
    cdef float std_val_mir = -999.

    cdef float sum_val_vid = 0
    cdef float mea_val_vid = -999.
    cdef float std_val_vid = -999.

    cdef float res[15]

    cdef float n_wat = 0
    cdef float n_cld = 0


    # Loop to find mean
    for i in range(0, arrsize):
        for j in range(0, arrsize):
            if i >= winsize - 1 and i <= winsize + 1 and j >= winsize - 1 and j <= winsize + 1:
                continue
            if cur__lwi[i,j] < lwi_min_thresh or isnan(cur__lwi[i, j]):
                n_cld = n_cld + 1
                continue
            if cur__mir[i,j] > cen__mir or isnan(cur__mir[i, j]):
                continue
            if cur__btd[i,j] > cen__btd or isnan(cur__btd[i, j]):
                continue
            if cur__mir[i,j] < mir_min_thresh or cur__mir[i,j] > mir_max_thresh:
                continue
            if cur__btd[i,j] > btd_max_thresh:
                continue
            if lsm_land_val != 255:
                if cur__lsm[i,j] != lsm_land_val:
                    n_wat = n_wat + 1
                    continue
            sum_val_vi = sum_val_vi + cur__vis[i, j]
            sum_val_nd = sum_val_nd + cur_ndfi[i, j]
            sum_val_lw = sum_val_lw + cur__lwi[i, j]
            sum_val_btd = sum_val_btd + cur__btd[i, j]
            sum_val_mir = sum_val_mir + cur__mir[i, j]
            sum_val_vid = sum_val_vid + cur__vid[i, j]
            n_good = n_good + 1
    perc_good = n_good / (npix - 9)

    mea_val_vi = sum_val_vi / n_good
    mea_val_nd = sum_val_nd / n_good
    mea_val_lw = sum_val_lw / n_good
    mea_val_btd = sum_val_btd / n_good
    mea_val_mir = sum_val_mir / n_good
    mea_val_vid = sum_val_vid / n_good

     # Loop to find standard deviation
    sum_val_vi = 0
    sum_val_nd = 0
    sum_val_lw = 0
    sum_val_btd = 0
    sum_val_mir = 0
    sum_val_vid = 0

    for i in range(0, arrsize):
        for j in range(0, arrsize):
            if i >= winsize - 1 and i <= winsize + 1 and j >= winsize - 1 and j <= winsize + 1:
                continue
            if cur__mir[i,j] > cen__mir or isnan(cur__mir[i, j]):
                continue
            if cur__btd[i,j] > cen__btd or isnan(cur__btd[i, j]):
                continue
            if cur__mir[i,j] < mir_min_thresh or cur__mir[i,j] > mir_max_thresh:
                continue
            if cur__btd[i,j] > btd_max_thresh:
                continue
            if cur__lwi[i,j] < lwi_min_thresh or isnan(cur__lwi[i, j]):
                continue
            if lsm_land_val != 255:
                if cur__lsm[i,j] != lsm_land_val:
                    continue
            sum_val_vi = sum_val_vi + (cur__vis[i, j] - mea_val_vi) * (cur__vis[i, j] - mea_val_vi)
            sum_val_nd = sum_val_nd + (cur_ndfi[i, j] - mea_val_nd) * (cur_ndfi[i, j] - mea_val_nd)
            sum_val_lw = sum_val_lw + (cur__lwi[i, j] - mea_val_lw) * (cur__lwi[i, j] - mea_val_lw)
            sum_val_btd = sum_val_btd + (cur__btd[i, j] - mea_val_btd) * (cur__btd[i, j] - mea_val_btd)
            sum_val_mir = sum_val_mir + (cur__mir[i, j] - mea_val_mir) * (cur__mir[i, j] - mea_val_mir)
            sum_val_vid = sum_val_vid + (cur__vid[i, j] - mea_val_vid) * (cur__vid[i, j] - mea_val_vid)

    std_val_lw = sqrt(sum_val_lw / n_good)
    std_val_nd = sqrt(sum_val_nd / n_good)
    std_val_vi = sqrt(sum_val_vi / n_good)
    std_val_btd = sqrt(sum_val_btd / n_good)
    std_val_mir = sqrt(sum_val_mir / n_good)
    std_val_vid = sqrt(sum_val_vid / n_good)

    res[0] = perc_good
    res[1] = n_cld
    res[2] = n_wat
    res[3] = mea_val_lw
    res[4] = std_val_lw
    res[5] = mea_val_nd
    res[6] = std_val_nd
    res[7] = mea_val_vi
    res[8] = std_val_vi
    res[9] = mea_val_btd
    res[10] = std_val_btd
    res[11] = mea_val_mir
    res[12] = std_val_mir
    res[13] = mea_val_vid
    res[14] = std_val_vid
    return res


def get_mea_std_window(unsigned char[:,:] pfp_data,
                       float[:,:] vi_rad_data,
                       float[:,:] ndfi_data,
                       float[:,:] lw_rad_data,
                       float[:,:] btd_arr,
                       float[:,:] mir_arr,
                       float[:,:] vid_arr,
                       unsigned char[:,:] lsm,
                       unsigned char lsm_land_val,
                       int winsize):

    cdef int scn_width = vi_rad_data.shape[0]
    cdef int scn_height = vi_rad_data.shape[1]

    cdef int x_0 = 0
    cdef int x_1 = scn_width
    cdef int y_0 = 0
    cdef int y_1 = scn_height

    # Counters for main loop
    cdef int x = 0
    cdef int y = 0
    cdef int wsize = winsize
    cdef int cur_win[4]

    # Intermediate variables
    cdef float sum_tot = 0
    cdef int n = 0
    cdef float cen_vid = 0
    cdef float cen_btd = 999
    cdef float cen_mir = 999
    cdef float cen_irr = 0
    cdef float cen_lwi = 0
    cdef float res[15]

    # Output datasets
    cdef np.ndarray[dtype=np.float32_t, ndim=3] outarr = np.zeros((16, scn_width, scn_height), dtype=np.single)
    cdef float[:,:, ::1] outarr_view = outarr

    min_wsize = 5
    max_wsize = 15
    perc_thresh = 0.65

    # Loop across all pixels in the image
    for x in range(x_0, x_1):
        for y in range(y_0, y_1):
            if pfp_data[x, y] != 1:
                outarr_view[:, x, y] = -999
                continue
            for wsize in range(min_wsize, max_wsize):
                # Determine coordinates of the current window
                cur_win = get_curwindow_pos(x, y, wsize, scn_width, scn_height)
                # Don't process if any coords are bad (usually at extreme edges of image)
                if cur_win[0] < 0:
                    continue

                cur__vis = vi_rad_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur_ndfi = ndfi_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__lwi = lw_rad_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__btd = btd_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__mir = mir_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__vid = vid_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__lsm = lsm[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]

                cen_mir = mir_arr[x, y]
                cen_btd = btd_arr[x, y]

                # Compute mean and standard deviation across window
                res = get_window_mea_stdv(wsize,
                                          cur__vis,
                                          cur_ndfi,
                                          cur__lwi,
                                          cur__btd,
                                          cur__mir,
                                          cur__vid,
                                          cur__lsm,
                                          cen_mir,
                                          cen_btd,
                                          lsm_land_val)
                #print(f'{res[0]:4.4f} {res[1]:4.4f} {res[2]:4.4f} {res[3]:4.4f} {res[4]:4.4f} {res[5]:4.4f}{res[6]:4.4f} {res[7]:4.4f} {res[8]:4.4f} {res[9]:4.4f} {res[10]:4.4f} {res[11]:4.4f} {res[12]:4.4f} {res[13]:4.4f} {res[14]:4.4f} {res[15]:4.4f}')
                if res[0] > perc_thresh:
                    break

            # Set output array to:
            # [0]: Percentage of acceptable pixels in window
            outarr_view[0, x, y] = res[0]            # Percentage of good pixels
            outarr_view[1, x, y] = wsize * wsize     # Number of pixels in window
            outarr_view[2, x, y] = res[1]            # Number of cloudy pixels in window
            outarr_view[3, x, y] = res[2]            # Number of water body pixels in window
            outarr_view[4, x, y] = res[3]            # LW BT mean
            outarr_view[5, x, y] = res[4]            # LW BT std-dev
            outarr_view[6, x, y] = res[5]            # NDFI mean
            outarr_view[7, x, y] = res[6]            # NDFI std-dev
            outarr_view[8, x, y] = res[7]            # VIS radiance mean
            outarr_view[9, x, y] = res[8]            # VIS radiance std-dev
            outarr_view[10, x, y] = res[9]           # BTD mean
            outarr_view[11, x, y] = res[10]          # BTD std-dev
            outarr_view[12, x, y] = res[11]          # MIR BT mean
            outarr_view[13, x, y] = res[12]          # MIR BT std-dev
            outarr_view[14, x, y] = res[13]          # VisDiff mean
            outarr_view[15, x, y] = res[14]          # VisDiff std-dev

    return outarr