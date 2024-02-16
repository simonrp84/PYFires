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

import pyfires.PYF_Consts as PYFc
from libc.stdlib cimport qsort, malloc, free
from libc.math cimport isnan, cos, M_PI
cimport numpy as np
import numpy as np
import cython
import dask


def get_local_stats(procpix, btd, vid):
    """Python wrapper for cython local statistics function.
    Inputs:
     - proc_pix: A binary mask indicating which pixels to process (NxM array)
     - btd: The brightness temperature difference (NxM array)
     - vid: The MIR radiance minus VIS and LWIR components (NxM array)
    Returns:
     - diffarr: The average differences between the pixel and its neighbours for
                each of the three input arrays: mir, btd, vid)  (NxMx2 array)
    """

    cdef int scn_width = int(procpix.shape[0])
    cdef int scn_height = int(procpix.shape[1])

    cdef np.ndarray[dtype=np.float32_t, ndim=3] outarr = np.zeros((scn_width, scn_height, 2), dtype=np.float32)
    cdef np.float32_t[:, :, ::1] outarr_view = outarr

    _get_local_stats(procpix, btd, vid, outarr_view, scn_width, scn_height)

    return outarr


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _get_local_stats(unsigned char[:,:] proc_pix,
                      float [:, :] btd,
                      float [:, :] vid,
                      np.float32_t[:, :, :] outarr,
                      int scn_width,
                      int scn_height) noexcept nogil:  #noqa
    """Compute the local statistics for the current pixel.
    Inputs:
     - proc_pix: A binary mask indicating which pixels to process (NxM array)
     - mirbt: The MIR brightness temperature (NxM array)
     - btd: The brightness temperature difference (NxM array)
     - vid: The MIR radiance minus VIS and LWIR components (NxM array)
    Returns:
     - diffarr: The average differences between the pixel and its neighbours for
                each of the three input arrays: mir, btd, vid)  (NxMx3 array)
    """
    cdef int xpos = 0
    cdef int ypos = 0
    cdef int n_good = 0

    cdef float btdif = 0
    cdef float viddif = 0

    for xpos in range(1, scn_width - 1):
        for ypos in range(1, scn_height - 1):
            # If current pixel is not a fire candidate, skip
            if proc_pix[xpos, ypos] == 0:
                continue

            # Initialise counter
            n_good = 0

            if proc_pix[xpos - 1, ypos + 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos - 1, ypos + 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos - 1, ypos + 1])
                n_good = n_good + 1
            if proc_pix[xpos -1 , ypos] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos - 1, ypos])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos - 1, ypos])
                n_good = n_good + 1
            if proc_pix[xpos - 1, ypos - 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos - 1, ypos - 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos - 1, ypos - 1])
                n_good = n_good + 1

            if proc_pix[xpos, ypos + 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos, ypos + 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos, ypos + 1])
                n_good = n_good + 1
            if proc_pix[xpos, ypos - 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos, ypos - 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos, ypos - 1])
                n_good = n_good + 1

            if proc_pix[xpos + 1, ypos + 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos + 1, ypos + 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos + 1, ypos + 1])
                n_good = n_good + 1
            if proc_pix[xpos + 1, ypos] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos + 1, ypos])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos + 1, ypos])
                n_good = n_good + 1
            if proc_pix[xpos + 1, ypos - 1] == 0:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] +  (btd[xpos, ypos] - btd[xpos + 1, ypos - 1])
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] +  (vid[xpos, ypos] - vid[xpos + 1, ypos - 1])
                n_good = n_good + 1

            if n_good < 1:
                outarr[xpos, ypos, 0] = 1.
                outarr[xpos, ypos, 1] = 1.
            else:
                outarr[xpos, ypos, 0] = outarr[xpos, ypos, 0] / float(n_good)
                outarr[xpos, ypos, 1] = outarr[xpos, ypos, 1] / float(n_good)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int compare_twofloats(const void *a, const void *b) noexcept nogil: #noqa
    cdef int a_val = (<const int *> a)[0]
    cdef int b_val = (<const int *> b)[0]

    if a_val < b_val:
        return -1
    if a_val > b_val:
        return 1
    return 0


def py_get_curwindow_pos(int xpos, int ypos, int winsize, int scn_width, int scn_height):
    return get_curwindow_pos(xpos, ypos, winsize, scn_width, scn_height)

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
        return shape_sel
    elif shape_sel[0] >= scn_width:
        shape_sel[0] = -999
        return shape_sel
    if shape_sel[1] < 0:
        shape_sel[0] = -999
        return shape_sel
    elif shape_sel[1] >= scn_width:
        shape_sel[0] = -999
        return shape_sel
    if shape_sel[2] < 0:
        shape_sel[0] = -999
        return shape_sel
    elif shape_sel[2] >= scn_height:
        shape_sel[0] = -999
        return shape_sel
    if shape_sel[3] < 0:
        shape_sel[0] = -999
        return shape_sel
    elif shape_sel[3] >= scn_height:
        shape_sel[0] = -999
        return shape_sel

    return shape_sel

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef get_window_mea_stdv(int winsize,
                         unsigned char[:,:] pfp,
                         float[:,:] cur__vis,
                         float[:,:] cur__lwi,
                         float[:,:] cur__btd,
                         float[:,:] cur__mir,
                         float[:,:] cur__vid,
                         unsigned char[:,:] cur__lsm,
                         float cen__mir,
                         float cen__btd,
                         float cen__lat,
                         unsigned char lsm_land_val,
                         float mir_minlat_thresh,
                         float mir_maxlat_thresh,
                         float mir_max_thresh,
                         float btd_max_thresh,
                         float lwi_min_thresh,
                         ):
    cdef int arrsize_x = winsize + winsize + 1
    cdef int arrsize_y = winsize + winsize + 1
    cdef int i = 0
    cdef int j = 0

    cdef int arr_shp_x = int(cur__lwi.shape[0])
    cdef int arr_shp_y = int(cur__lwi.shape[1])

    cdef int n_good = 0
    cdef float perc_good = 0
    cdef int n_pfp = 0
    cdef float perc_pfp = 0

    cdef float sum_val_vi = 0
    cdef float mea_val_vi = -999.
    cdef float std_val_vi = -999.

    cdef float sum_val_btd = 0
    cdef float mea_val_btd = -999.
    cdef float std_val_btd = -999.

    cdef float sum_val_mir = 0
    cdef float mea_val_mir = -999.
    cdef float std_val_mir = -999.

    cdef float sum_val_vid = 0
    cdef float mea_val_vid = -999.
    cdef float std_val_vid = -999.

    cdef float mir_slope = mir_maxlat_thresh - mir_minlat_thresh
    cdef float mir_min_thresh = mir_minlat_thresh + mir_slope * float(cos(cen__lat * M_PI / 180.))

    cdef float res[12]

    cdef int n_wat = 0
    cdef int n_cld = 0

    if arrsize_x > arr_shp_x:
        arrsize_x = arr_shp_x

    if arrsize_y > arr_shp_y:
        arrsize_y = arr_shp_y

    cdef int npix = arrsize_x * arrsize_y

    # Loop to find mean
    for i in range(0, arrsize_x):
        for j in range(0, arrsize_y):
            if i >= winsize - 1 and i <= winsize + 1 and j >= winsize - 1 and j <= winsize + 1:
                continue
            if pfp[i,j] == 1:
                n_pfp = n_pfp + 1
            if cur__mir[i,j] > cen__mir or isnan(cur__mir[i, j]):
                continue
            if cur__lwi[i,j] < lwi_min_thresh or isnan(cur__lwi[i, j]):
                n_cld = n_cld + 1
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
            n_good = n_good + 1
    perc_good = n_good / (npix - 9.)
    perc_pfp = (n_good + n_pfp) / (npix - 9.)

    cdef float *mainmed_mir = <float *> malloc(n_good * sizeof(float))
    cdef float *mainmed_btd = <float *> malloc(n_good * sizeof(float))
    cdef float *mainmed_vi = <float *> malloc(n_good * sizeof(float))
    cdef float *mainmed_vid = <float *> malloc(n_good * sizeof(float))

    cdef int cur_n = 0

    for i in range(0, arrsize_x):
        for j in range(0, arrsize_y):
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
            mainmed_mir[cur_n] = cur__mir[i, j]
            mainmed_btd[cur_n] = cur__btd[i, j]
            mainmed_vi[cur_n] = cur__vis[i, j]
            mainmed_vid[cur_n] = cur__vid[i, j]
            cur_n = cur_n + 1

    with cython.nogil:
        qsort(mainmed_mir, n_good, sizeof(float), compare_twofloats)
        qsort(mainmed_btd, n_good, sizeof(float), compare_twofloats)
        qsort(mainmed_vi, n_good, sizeof(float), compare_twofloats)
        qsort(mainmed_vid, n_good, sizeof(float), compare_twofloats)

        if n_good % 2 == 0:
            mea_val_vi = (mainmed_vi[n_good / 2] + mainmed_vi[n_good / 2 - 1]) / 2
            mea_val_btd = (mainmed_btd[n_good / 2] + mainmed_btd[n_good / 2 - 1]) / 2
            mea_val_mir = (mainmed_mir[n_good / 2] + mainmed_mir[n_good / 2 - 1]) / 2
            mea_val_vid = (mainmed_vid[n_good / 2] + mainmed_vid[n_good / 2 - 1]) / 2
        else:
            mea_val_vi = mainmed_vi[n_good / 2]
            mea_val_btd = mainmed_btd[n_good / 2]
            mea_val_mir = mainmed_mir[n_good / 2]
            mea_val_vid = mainmed_vid[n_good / 2]

    free(mainmed_mir)
    free(mainmed_btd)
    free(mainmed_vi)
    free(mainmed_vid)

     # Loop to find standard deviation
    sum_val_vi = 0
    sum_val_btd = 0
    sum_val_mir = 0
    sum_val_vid = 0
    for i in range(0, arrsize_x):
        for j in range(0, arrsize_y):
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
            sum_val_vi = sum_val_vi + abs(cur__vis[i, j] - mea_val_vi)
            sum_val_btd = sum_val_btd + abs(cur__btd[i, j] - mea_val_btd)
            sum_val_mir = sum_val_mir + abs(cur__mir[i, j] - mea_val_mir)
            sum_val_vid = sum_val_vid + abs(cur__vid[i, j] - mea_val_vid)

    std_val_vi = sum_val_vi / n_good
    std_val_btd = sum_val_btd / n_good
    std_val_mir = sum_val_mir / n_good
    std_val_vid = sum_val_vid / n_good

    res[0] = perc_good
    res[1] = perc_pfp
    res[2] = float(npix)
    res[3] = float(n_wat)
    res[4] = mea_val_vi
    res[5] = std_val_vi
    res[6] = mea_val_btd
    res[7] = std_val_btd
    res[8] = mea_val_mir
    res[9] = std_val_mir
    res[10] = mea_val_vid
    res[11] = std_val_vid
    return res


def get_mea_std_window(unsigned char[:,:] pfp_data,
                       float[:,:] vi_rad_data,
                       float[:,:] lw_bt_data,
                       float[:,:] btd_arr,
                       float[:,:] mir_arr,
                       float[:,:] vid_arr,
                       unsigned char[:,:] lsm,
                       float[:,:] lats,
                       int scn_width,
                       int scn_height,
                       unsigned char lsm_land_val,
                       int min_wsize,
                       int max_wsize,
                       float perc_thresh,
                       mir_minlat_thresh,
                       mir_maxlat_thresh,
                       mir_max_thresh,
                       btd_max_thresh,
                       lwi_min_thresh):

    cdef int x_0 = 0
    cdef int x_1 = scn_width
    cdef int y_0 = 0
    cdef int y_1 = scn_height

    # Counters for main loop
    cdef int x = 0
    cdef int y = 0
    cdef int wsize = min_wsize
    cdef int cur_win[4]

    # Intermediate variables
    cdef float sum_tot = 0
    cdef int n = 0
    cdef float cen_btd = 999
    cdef float cen_mir = 999
    cdef float cen_lon = 999
    cdef float res[12]

    # Output datasets
    cdef np.ndarray[dtype=np.float32_t, ndim=3] outarr = np.zeros((12, scn_width, scn_height), dtype=np.single)
    cdef float[:,:, ::1] outarr_view = outarr

    # Loop across all pixels in the image
    for x in range(x_0, x_1):
        for y in range(y_0, y_1):
            try:
                if pfp_data[x + max_wsize, y + max_wsize] != 1:
                    outarr_view[:, x, y] = -999
                    continue
            except:
                print(pfp_data.shape)
                print(x, y,  max_wsize)
                raise
            for wsize in range(min_wsize, max_wsize+1):
                # Determine coordinates of the current window
                cur_win = get_curwindow_pos(x + max_wsize, y + max_wsize, wsize, scn_width, scn_height)
                # Don't process if any coords are bad (usually at extreme edges of image)
                if cur_win[0] < 0:
                    outarr_view[:, x, y] = -999
                    continue

                cur__vis = vi_rad_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__lwi = lw_bt_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__btd = btd_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__mir = mir_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__vid = vid_arr[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__lsm = lsm[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]
                cur__pfp = pfp_data[cur_win[0]:cur_win[1], cur_win[2]:cur_win[3]]

                cen_mir = mir_arr[x + max_wsize, y + max_wsize]
                cen_btd = btd_arr[x + max_wsize, y + max_wsize]
                cen_lon = lats[x + max_wsize, y + max_wsize]

                # Compute mean and standard deviation across window
                res = get_window_mea_stdv(wsize,
                                          cur__pfp,
                                          cur__vis,
                                          cur__lwi,
                                          cur__btd,
                                          cur__mir,
                                          cur__vid,
                                          cur__lsm,
                                          cen_mir,
                                          cen_btd,
                                          cen_lon,
                                          lsm_land_val,
                                          mir_minlat_thresh,
                                          mir_maxlat_thresh,
                                          mir_max_thresh,
                                          btd_max_thresh,
                                          lwi_min_thresh)
                if res[0] > perc_thresh or res[1] > perc_thresh + 0.1:
                    break

            # Set output array to:
            # [0]: Percentage of acceptable pixels in window
            outarr_view[0, x, y] = res[0]     # Percentage of good pixels
            outarr_view[1, x, y] = res[1]     # Number of pixels in window
            outarr_view[2, x, y] = res[2]     # Number of cloudy pixels in window
            outarr_view[3, x, y] = res[3]     # Number of water body pixels in window
            outarr_view[4, x, y] = res[4]     # VIS radiance mean
            outarr_view[5, x, y] = res[5]     # VIS radiance std-dev
            outarr_view[6, x, y] = res[6]   # BTD mean
            outarr_view[7, x, y] = res[7]   # BTD std-dev
            outarr_view[8, x, y] = res[8]   # MIR BT mean
            outarr_view[9, x, y] = res[9]   # MIR BT std-dev
            outarr_view[10, x, y] = res[10]   # VisDiff mean
            outarr_view[11, x, y] = res[11]   # VisDiff std-dev

    return outarr

def py_get_window_mea_stdv(wsize, cpfp, cvis, clwi, cbtd, cmir, cvid, clsm, cen_mir, cen_btd, cen_lon, lsm_land_val):
    mir_minlat_thresh = PYFc.mir_minlat_thresh
    mir_maxlat_thresh = PYFc.mir_maxlat_thresh
    mir_max_thresh = PYFc.mir_max_thresh
    btd_max_thresh = PYFc.btd_max_thresh
    lwi_min_thresh = PYFc.lwi_min_thresh

    return get_window_mea_stdv(wsize, cpfp, cvis, clwi, cbtd, cmir, cvid, clsm, cen_mir, cen_btd, cen_lon, lsm_land_val,
                               mir_minlat_thresh, mir_maxlat_thresh, mir_max_thresh, btd_max_thresh, lwi_min_thresh)