#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:55:36 2020

@author: davsan06
"""

# from astropy.io import fits
from astropy.table import Table, vstack
import os
import matplotlib.pyplot as plt
import seaborn as sns

redmagic_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

# Reading BRIGHT catalogue
cat_bright = Table.read(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_bright.fits'))
cat_bright.info()

# Reading FAINT catalogue
cat_faint = Table.read(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_faint.fits'))
cat_faint.info()

# Stacking both catalogues
cat = vstack([cat_bright, cat_faint])
cat.info()

# Saving
cat.write(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_full.fits'))

# Exploring redshift from redmagic
z_red = cat['ZREDMAGIC']
z_red_err = cat['ZREDMAGIC_ERR']
z_spec = cat['Z_SPEC']

plt.hist(z_red)
plt.hist(z_red_err)
plt.hist2d(z_red, z_red_err)

sns.kdeplot(z_red[:10000], z_red_err[:10000])
plt.xlabel('$z_{photo}$')
plt.ylabel('$\Delta z_{photo}$')

# Percentage of galaxies with spectroscopic redshift
len(z_spec[z_spec != -1])/len(z_spec)*100

# Exploring SPECTROSCOPIC sample
z_red = cat['ZREDMAGIC'][cat['Z_SPEC'] != -1]
z_spec = cat['Z_SPEC'][cat['Z_SPEC'] != -1]

sns.kdeplot(z_red, z_spec)
plt.xlabel('$z_{photo}$')
plt.ylabel('$z_{spec}$')

plt.scatter(z_red, z_spec)
plt.xlabel('$z_{photo}$')
plt.ylabel('$z_{spec}$')