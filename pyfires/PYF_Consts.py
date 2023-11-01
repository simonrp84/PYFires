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

"""Constant values used in the fire detection."""


# Platform short names used for selecting land-sea mask
plat_shortnames = {'Himawari-8': 'HIMA',
                   'Himawari-9': 'HIMA',
                   'GOES-16': 'GOES',
                   'GOES-17': 'GOES',
                   'GOES-18': 'GOES',
                   'Meteosat-8': 'MSG',
                   'Meteosat-9': 'MSG',
                   'Meteosat-10': 'MSG',
                   'Meteosat-11': 'MSG', }


# These are the radiance-to-temperature conversion coefficients for the FRP calculation.
# See: doi:10.1016/S0034-4257(03)00070-1 for more info
# The utility function `compute_aval` in PYF_Rad2TbFuncs can be used to compute the value of 'a'.
rad_to_bt_dict = {
    'Himawari-8': 2.91848e-07,
    'Himawari-9': 2.33101e-07,
    'Meteosat-8': 3.34039e-07,
    'Meteosat-9': 3.30150e-07,
    'Meteosat-10': 3.34262e-07,
    'Meteosat-11': 3.20514e-07,
    'Meteosat-12': 2.02416e-07,
    'GOES-16': 3.01110e-07,
    'GOES-17': 2.94711e-07,
    'GOES-18': 3.15338e-07,
    'Geo-kompsat-2A': 2.34269e-07,
    'fy-4a': 1.53966e-07,
    'fy-4b': 1.53536e-07
}

# Stefan-Boltzmann constant
sigma = 5.670373e-8

# *** Stage 1 ***
# Values used in stage 1a tests
mir_thresh_bt = 310.5
mir_thresh_sza_adj = -0.3
mir_thresh_limit = 280
btd_thresh_bt = 1.75
btd_thresh_sza_adj = -0.0049
stage1_pass_thresh = 4

# Values used in stage 1b tests
btd_thresh_limit = 1
kern_thresh_btd = 1.5
kern_thresh_sza_adj = -0.012

# *** Stage 2 ***
# Number of cloudy pixels in 15x15 window to consider 'cloudy'
glint_cldcnt_thresh = 1

# Sunglint vis ratio threshold for cloud within background window
glint_vis_thresh_cld = 0.7
# Sunglint IR ratio threshold for cloud in window
glint_ir_thresh_cld = 0.018

# Sunglint vis ratio threshold for when no cloud in window
glint_vis_thresh_clr = 0.4

# For contextual tests
# Fraction of valid pixels in window required for continuation of processing
win_frac = 0.65
mir_cand_minbt_thresh = 270
mir_cand_maxbt_thresh = 330
mir_cand_maxbtd_thresh = 8

# Dictionary holding threshold values for the stage 6 / post-processing filters.
ppf_thresh_dict = {
    'ppf1_mir_lwir_bt': 5,
    'ppf1_mir_vis_rad': 0.87,
    'ppf1_firconf': 0.2,
    'ppf1_max_sza': 70,
    'ppf2_vi2_rad': 0.04,
    'ppf2_mir_lwir_bt': 3,
    'ppf2_firconf': 0.2,
    'ppf2_min_sza': 90,
    'ppf3_mir_lwir_bt': 5,
    'ppf3_mir_vis_rad': 1.,
    'ppf3_firconf': 0.1,
    'ppf3_min_sza': 65,
    'ppf3_max_sza': 90,
    'ppf4_mir_lwir_bt': 5,
    'ppf4_mir_vis_rad': 0.75,
    'ppf4_min_sza': 65,
    'ppf4_max_sza': 90
}
