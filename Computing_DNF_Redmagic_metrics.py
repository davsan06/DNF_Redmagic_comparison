#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:36:58 2020

@author: davsan06
"""
from astropy.table import Table, unique
# from astropy.io import fits
# import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

# Reading Redmagic matched with DES Y3 Gold catalog
cat_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

cat = Table.read(os.path.join(cat_path, 'merge_catalog_desy3gold_redmagic.fits'))
cat.info()

unique(cat).info()

# Subsetting variables from the catalog
cat = cat['COADD_OBJECT_ID',
          'DNF_ZMC_SOF',
          'DNF_ZMEAN_SOF',
          'DNF_ZSIGMA_SOF',
          'zredmagic',
          'zredmagic_e',
          'zspec']

# Subsetting objects with spectroscopic info
mask = cat['zspec'] != -1
cat = cat[mask]

# Removing objects with cat['zspec'] == -9.99
mask_2 = cat['zspec'] != -9.99
cat = cat[mask_2]

###############################################################################
## Scatter plot: Fig.1 z_spec vs DNF_MEAN_SOF
##               Fig.2 z_spec vs Redmagic
###############################################################################

xy_dnf = np.vstack([cat['zspec'], cat['DNF_ZMEAN_SOF']])
dens_dnf = gaussian_kde(xy_dnf)(xy_dnf)

xy_red = np.vstack([cat['zspec'], cat['zredmagic']])
dens_red = gaussian_kde(xy_red)(xy_red)

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,4))

sc1 = axs[0].scatter(x=cat['zspec'], y=cat['DNF_ZMEAN_SOF'],
            s=1,
            c=dens_dnf,
            cmap='magma')
axs[0].set_xlim([0.0, 1.1])
axs[0].set_ylim([0.0, 1.1])
axs[0].set_title('DNF', fontsize=18)
axs[0].set_xlabel('$z_{spec}$', fontsize=16)
axs[0].set_ylabel('$z_{photo}$', fontsize=16)
plt.colorbar(sc1, ax=axs[0])

sc2 = axs[1].scatter(x=cat['zspec'], y=cat['zredmagic'],
            s=1,
            c=dens_red,
            cmap='magma')
axs[1].set_xlim([0.0, 1.1])
axs[1].set_ylim([0.0, 1.1])
axs[1].set_title('RedMaGic', fontsize=18)
axs[1].set_xlabel('$z_{spec}$', fontsize=16)
axs[1].set_ylabel('$z_{photo}$', fontsize=16)
plt.colorbar(sc2, ax=axs[1])

plt.savefig(os.path.join(cat_path, 'Results', 'dispersion_photoz_methods.png'),
            format='png',
            dpi=150)
plt.show()

###############################################################################
## Histogram plot: comparison of n(z)'s distribution computed with DNF,
##                  Redmagic and spectros
###############################################################################

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,4))

color = 'tab:red'
axs[0].hist(cat['zspec'], 
         bins=40, 
         histtype='step', 
         label='Spectroscopic',
         color='red')
axs[0].set_xlim([0.0, 1.1])
axs[0].set_xlabel('$z (redshift)$', fontsize=16)
axs[0].set_ylabel('N(spectros)', color=color, fontsize=14)
axs[0].tick_params(axis='y', labelcolor=color)

ax0 = axs[0].twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax0.hist(cat['DNF_ZMC_SOF'], 
         bins=40, 
         histtype='step', 
         label='DNF')
ax0.set_ylabel('N(DNF_ZMC)', color=color, fontsize=14)
ax0.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
axs[1].hist(cat['zspec'], 
         bins=40, 
         histtype='step', 
         label='Spectroscopic',
         color='red')
axs[1].set_xlim([0.0, 1.1])
axs[1].set_xlabel('$z (redshift)$', fontsize=16)
axs[1].set_ylabel('N(spectros)', color=color, fontsize=14)
axs[1].tick_params(axis='y', labelcolor=color)

ax1 = axs[1].twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax1.hist(cat['zredmagic'],
         bins=40,
         histtype='step',
         label='Redmagic',
         color='green')
ax1.set_ylabel('N(REDMAGIC)', color=color, fontsize=14)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(os.path.join(cat_path, 'Results', 'nz_methods_comparison.png'),
            format='png',
            dpi=150)
plt.show()