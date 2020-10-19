#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:18:15 2020

@author: davsan06
"""

# Goal: to split Redmagic catalogue between objects with Spectroscopic info 
## and those whith just Photometric info

from astropy.io import fits
# from astropy.table import Table
import os
import numpy as np


redmagic_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

# Reading FULL catalogue
cat = fits.open(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_full.fits'))[1].data

# Splitting
mask = cat['Z_SPEC'] == -1

cat_photo = cat[mask]
cat_photo = fits.BinTableHDU(data=cat_photo)
cat_photo.writeto(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_photometric.fits'))

cat_spec = cat[np.logical_not(mask)]
cat_spec = fits.BinTableHDU(data=cat_spec)
cat_spec.writeto(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_spectroscopic.fits'))
