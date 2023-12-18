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

from cython.parallel import prange
from libc.math cimport exp,  pow, abs
cimport numpy as np
import numpy as np
import cython


def aniso_diff(in_img,
               niter=5,
               kappa=20,
               gamma=0.2):
    """A wrapper for the Cython function to compute the anisotropic diffusion of a pixel.
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
    return _aniso_diff(in_img, niter, kappa, gamma)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef float get_single_aniso(float cval,
                            float pxval,
                            float nxval,
                            float pyval,
                            float nyval,
                            float kappa,
                            float gamma,
                            ) nogil:
    """Compute the anisotropic diffusion of a pixel."""

    cdef float pxv = abs(pxval - cval)
    cdef float nxv = abs(nxval - cval)
    cdef float pyv = abs(pyval - cval)
    cdef float nyv = abs(nyval - cval)

    cdef float pxc = exp(-pow(pxv / kappa, 2.))
    cdef float nxc = exp(-pow(nxv / kappa, 2.))
    cdef float pyc = exp(-pow(pyv / kappa, 2.))
    cdef float  nyc = exp(-pow(nyv / kappa, 2.))

    cdef float ret = cval + gamma * (pxv * pxc + nxv * nxc + pyv * pyc + nyv * nyc)

    return ret



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef _aniso_diff(float[:,:] in_img,
                      int n_iter=5,
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
    cdef int ii, x, y, px, nx, py, ny
    cdef float pxc, nxc, pyc, nyc
    cdef int scn_width = int(in_img.shape[0])
    cdef int scn_height = int(in_img.shape[1])
    cdef float deltas, deltae, gs, ge

    cdef int niter = int(n_iter)

    # Output datasets
    cdef np.ndarray[dtype=np.float32_t, ndim=2] outarr = np.zeros((scn_width, scn_height), dtype=np.single)
    cdef float[:, ::1] outarr_view = outarr

    # Initialise output
    outarr_view[:, :] = in_img[:, :]

    # Loop over iterations
    for ii in range(0, niter):
        for x in prange(1, scn_width, nogil=True):
            for y in range(1, scn_height):

                # Prev and next pixel indices
                px = x - 1
                nx = x + 1
                py = y - 1
                ny = y + 1

                # Bounds checking
                if px < 0:
                    px = 0
                if nx >= scn_width:
                    nx = scn_width - 1
                if py < 0:
                    py = 0
                if ny >= scn_height:
                    ny = scn_height - 1

                outarr[x, y] = get_single_aniso(outarr[x, y],
                                                outarr[px, y],
                                                outarr[nx, y],
                                                outarr[x, py],
                                                outarr[x, ny],
                                                kappa,
                                                gamma)
    return outarr
