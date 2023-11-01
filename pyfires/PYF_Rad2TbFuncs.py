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

"""Helper functions for computing sensor specific coefficients used in FRP calculation."""

import numpy as np


def _bt_approx(bt, a, b):
    return a * np.power(bt, b)


def calc_rad_fromtb(temp, cwl):
    """Modified version of eqn 2 in Wooster's paper. Compute radiance from BT (K) and central wavelength (micron)."""
    c1 = 1.1910429e8
    c2 = 1.4387752e4

    first = c1
    second = np.power(cwl, 5) * (np.exp(c2 / (cwl * temp)) - 1.)

    return first / second


def compute_aval(idict, min_bt=300, max_bt=1500):
    """Compute the value of 'a' defined in equation 3 of doi:10.1016/S0034-4257(03)00070-1
    Inputs:
    - idict: A dictionary containing either: 'cwl' the band central wavelength or 'platform', 'inst' and 'chan' denoting
             the satellite platform name, instrument name and channel name respectively.
    - min_bt: The minimum BT for the computation, defaults to 300
    - max_bt: The maximum BT for the computation, defaults to 1500
    Returns:
    - a value (float)
    """
    from scipy.optimize import curve_fit

    bt = np.linspace(min_bt, max_bt, 1301)
    if 'cwl' in idict:
        ref_rad = calc_rad_fromtb(bt, idict['cwl'])
    elif 'platform' in idict and 'chan' in idict and 'inst' in idict:
        from pyspectral.rsr_reader import RelativeSpectralResponse
        from pyspectral.utils import get_central_wave

        rsr = RelativeSpectralResponse(idict['platform'], idict['inst'])
        cwl = get_central_wave(rsr.rsr[idict['chan']]['det-1']['wavelength'],
                               rsr.rsr[idict['chan']]['det-1']['response'])
        ref_rad = calc_rad_fromtb(bt, cwl)
    else:
        raise ValueError('input dict must contain either "cwl" or "platform", "inst" and "chan"')

    # Fit the curve
    popt, pcov = curve_fit(_bt_approx, bt, ref_rad)

    return popt[0]
