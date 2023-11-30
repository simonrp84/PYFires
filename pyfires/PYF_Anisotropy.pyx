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

"""Cython code for computing the anisotropic diffusion of a pixel."""

from libc.math cimport exp
cimport numpy as np
import numpy as np
import cython



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
def aniso_diff(float[:,:] in_img,
                int niter=5,
                float kappa=20.,
                float gamma=0.2):
    """Compute the anisotropic diffusion of a pixel.
    Based on the paper by Cohen and Wu (2021): https://doi.org/10.1016/j.ascom.2021.100507
    
    Inputs:
    - in_img: The input image to work on (float32 array)
    - niter: The number of iterations to perform
    - kappa: The conduction coefficient
    - gamma: The step value, suggested maximum is 0.25
    - step_x: The step distance between pixels on x axis
    - step_y: The step distance between pixels on y axis
    Returns:
    - img_diff: The diffused image (float32 array)
     """
    # Variables for use during processing
    cdef int ii, x, y
    cdef int scn_width = int(in_img.shape[0])
    cdef int scn_height = int(in_img.shape[1])
    cdef float deltas, deltae, gs, ge

    # Output datasets
    cdef np.ndarray[dtype=np.float32_t, ndim=2] outarr = np.zeros((scn_width, scn_height), dtype=np.single)
    cdef float[:, ::1] outarr_view = outarr
    cdef np.ndarray[dtype=np.float32_t, ndim=2] S = np.zeros((scn_width, scn_height), dtype=np.single)
    cdef float[:, ::1] sview = S
    cdef np.ndarray[dtype=np.float32_t, ndim=2] E = np.zeros((scn_width, scn_height), dtype=np.single)
    cdef float[:, ::1] eview = E

    # Initialise output
    outarr_view[:, :] = in_img[:, :]

    # Loop over iterations
    for ii in range(0, niter):
        for x in range(1, scn_width):
            for y in range(1, scn_height):

                # Prev and next pixel indices
                px = x - 1
                nx = x + 1
                py = y - 1
                ny = y + 1

                # Bounds checking
                if px < 0: px = 0
                if nx >= scn_width: nx = scn_width - 1
                if py < 0: py = 0
                if ny >= scn_height: ny = scn_height - 1

                pxv = abs(outarr_view[px, y] - outarr_view[x, y])
                nxv = abs(outarr_view[nx, y] - outarr_view[x, y])
                pyv = abs(outarr_view[x, py] - outarr_view[x, y])
                nyv = abs(outarr_view[x, ny] - outarr_view[x, y])

                pxc = exp(-pow(pxv / kappa, 2.))
                nxc = exp(-pow(nxv / kappa, 2.))
                pyc = exp(-pow(pyv / kappa, 2.))
                nyc = exp(-pow(nyv / kappa, 2.))

                outarr_view[x, y] = outarr_view[x, y] + gamma * (pxv * pxc + nxv * nxc + pyv * pyc + nyv * nyc)

    return outarr
