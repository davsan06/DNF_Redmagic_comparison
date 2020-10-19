#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:32:46 2020

@author: davsan06
"""

# from astropy.io import fits
from astropy.table import Table, vstack, unique
import os

redmagic_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

# Reading HIGH LUMINOSITY catalogue
cat_lum = Table.read(os.path.join(redmagic_path, 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_redmagic_highlum.fit'))
cat_lum.info()

# Reading HIGH DENSITY catalogue
cat_dens = Table.read(os.path.join(redmagic_path, 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_redmagic_highdens.fit'))
cat_dens.info()

# Stacking both catalogues
cat = vstack([cat_lum, cat_dens])
cat.info()

# Looking for duplicated rows
unique(cat).info()

# Number of objects with spectroscopic redshift
sum(cat['zspec'] != -1)

# Saving
cat.write(os.path.join(redmagic_path, 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_redmagic_combined.fits'))
