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
import scipy

# Reading Redmagic matched with DES Y3 Gold catalog
cat_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

cat = Table.read(os.path.join(cat_path, 'merge_catalog_desy3gold_redmagic.fits'))
# cat.info()

# Are objetcs unique?
# unique(cat).info()

# Reading TRAINING SET
train_path = '/pool/data1/des/catalogs/y3/train/'
train = Table.read(os.path.join(train_path, 'y3_train_april2018_nvp_y4.fits'))
# train.info()
train_ids = train['COADD_OBJECT_ID']

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

# Removing objects with IDs in the TRAIN sample
mask_3 = np.logical_not(np.isin(cat['COADD_OBJECT_ID'], train_ids))
cat = cat[mask_3]
cat.info

# Saving catalogue
# cat.write(os.path.join(cat_path, 'catalogue_redmagic_dnf_wo_train_only_zs.fit'))

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

# plt.savefig(os.path.join(cat_path, 'Results', 'dispersion_photoz_methods_v2.png'),
#             format='png',
#             dpi=150)
plt.show()

###############################################################################
## Histogram plot: comparison of n(z)'s distribution computed with DNF,
##                  Redmagic and spectros
###############################################################################

# binning = np.arange(0.05, 1.25, 0.05)
binning = np.arange(0.05, 1.25, 0.01)


fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,4))

color = 'tab:red'
axs[0].hist(cat['zspec'], 
         bins=binning, 
         histtype='step', 
         label='Spectroscopic',
         color='red')
axs[0].set_xlim([0.0, 1.1])
axs[0].set_xlabel('$z (redshift)$', fontsize=16)
axs[0].set_ylabel('N(spectros)', color=color, fontsize=14)
axs[0].tick_params(axis='y', labelcolor=color)

ax0 = axs[0].twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax0.hist(cat['DNF_ZMEAN_SOF'], 
         bins=binning, 
         histtype='step', 
         label='DNF')
ax0.set_ylabel('N(DNF_ZMEAN_SOF)', color=color, fontsize=14)
ax0.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
axs[1].hist(cat['zspec'], 
         bins=binning, 
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
         bins=binning,
         histtype='step',
         label='Redmagic',
         color='green')
ax1.set_ylabel('N(REDMAGIC)', color=color, fontsize=14)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(os.path.join(cat_path, 'Results', 'using_dnf_z_mean', 'nz_methods_comparison_v2.png'),
            format='png',
            dpi=150)
plt.show()

###############################################################################
## Histogram plot: comparison of n(z)'s distribution computed with DNF,
##                  Redmagic and spectros per redshift bin
###############################################################################
# binning = np.arange(0.05, 1.25, 0.01)
binning = np.arange(0.05, 1.25, 0.01/2)

# Redshift bin is defined using 'DNF_ZMEAN_SOF' variable
# v3 figures: Redshift bin is defined using 'z_spec' variable

for x in np.arange(0.1, 1.1, 0.1):
    # x = 0.1
    x = round(x, 2)
    
    cond = (cat['zspec']>=x)*(cat['zspec']<x+0.1)
    plt.hist(cat['zspec'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='Spectroscopic')
    plt.hist(cat['zredmagic'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='Redmagic')
    plt.hist(cat['DNF_ZMEAN_SOF'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='DNF')
    plt.xlabel('$z(redshift)$', fontsize=16)
    plt.xlim([x-0.05, x+0.15])
    plt.title("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=18)
    plt.legend()
    plt.savefig(os.path.join(cat_path, 'Results', 'using_dnf_z_mean', 'nz_bin_{}_{}_v3_width_0.01_zoom_in.png'.format(round(x,2), round(x+0.1,2))),
             format='png',
             dpi=150)
    plt.show()

###############################################################################
## Histogram plot: comparison of n(z)'s distribution computed with the
##                  differences:
##                      1. (z_mc_DNF - z_spec)
##                      2. (z_redmagic - z_spec)
##                      3. (z_mc_DNF-z_redmagic)    
###############################################################################
binning = np.arange(-0.5 - 0.01/4, 0.5 - 0.01/4, 0.01/2)

for x in np.arange(0.1, 1.1, 0.1):
    # x = 0.1
    x = round(x, 2)
    
    cond = (cat['zspec']>=x)*(cat['zspec']<x+0.1)
    
    plt.hist(cat['DNF_ZMEAN_SOF'][cond] - cat['zspec'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='DNF - Spect.')
    plt.hist(cat['zredmagic'][cond] - cat['zspec'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='Redmagic - Spect.')
    plt.hist(cat['DNF_ZMEAN_SOF'][cond] - cat['zredmagic'][cond],
         bins=binning,
         histtype='step',
         density=True,
         label='DNF - Redmagic')
    plt.axvline(x=0, alpha=0.4, linestyle='--', color='k')
    plt.xlabel('$\delta z(redshift)$', fontsize=16)
    plt.xlim([-0.1, 0.1])
    plt.title("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=18)
    plt.legend()
    plt.savefig(os.path.join(cat_path, 'Results', 'using_dnf_z_mean', 'differences_nz_bin_{}_{}_v2_new_binning.png'.format(round(x,2), round(x+0.1,2))),
                format='png',
                dpi=150)
    plt.show()    

###############################################################################
## Histogram plot: comparison of n(z)'s distribution computed with the
##                  differences:
##                      1. (z_mc_DNF - z_spec)
##                      2. (z_redmagic - z_spec)
##                  and FITTING by the sum of two Gaussian     with IMINUIT
##    MODIFICATION: central bin centered in zero
###############################################################################
from iminuit import Minuit
# from iminuit.cost import LeastSquares
from scipy import stats

def bi_norm(x, m1, m2, s1, s2, k1, k2):
    # m1, m2, s1, s2, k1, k2 = args
    ret = k1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
    ret += k2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
    return(ret)
    
def fcn_dnf(m1, m2, s1, s2, k1, k2):
    expt=bi_norm(bin_centers, m1, m2, s1, s2, k1, k2)
    delta=(n_dnf-expt)/n_dnf_error
    return((delta[n_dnf>0.]**2).sum())

def fcn_red(m1, m2, s1, s2, k1, k2):
    expt=bi_norm(bin_centers, m1, m2, s1, s2, k1, k2)
    delta=(n_red-expt)/n_red_error
    return((delta[n_red>0.]**2).sum())

binning = np.arange(-0.5 - 0.01/4, 0.5 - 0.01/4, 0.01/2)
width = binning[1] - binning[0]
bin_centers = np.append(np.array([0.5 * (binning[i] + binning[i+1]) for i in range(len(binning)-1)]), 4.95e-01)
i = 1

for x in np.arange(0.1, 1.1, 0.1):
    # x = 0.1
    x = round(x, 2)
    
    cond = (cat['zspec']>=x)*(cat['zspec']<x+0.1)
    
    # Plotting distributions
    n_dnf, bins_dnf, patches = plt.hist(cat['DNF_ZMEAN_SOF'][cond] - cat['zspec'][cond],
                                bins=binning,
                                histtype='step',
                                density=True,
                                label='DNF - Spect.')
    n_dnf = np.append(n_dnf, 0)
    n_dnf_error = np.sqrt(n_dnf)
    plt.errorbar(x=bin_centers, y=n_dnf, yerr=n_dnf_error/2, color='b', fmt='none', lw=0.7, capsize=1)
    
    n_red, bins_red, patches = plt.hist(cat['zredmagic'][cond] - cat['zspec'][cond],
                                       bins=binning,
                                       histtype='step',
                                       density=True,
                                       label='Redmagic - Spect.')
    n_red = np.append(n_red, 0)
    n_red_error = np.sqrt(n_red)
    plt.errorbar(x=bin_centers, y=n_red, yerr=n_red_error/2, color='y', fmt='none', lw=0.7, capsize=1)
    
    # Computing skewness
    print('bin {}: skew(DNF-Spec)={}, skew(Redma-Spec)={}'.format(i,
          round(scipy.stats.skew(cat['DNF_ZMEAN_SOF'][cond] - cat['zspec'][cond], axis=0, bias=True), 2),
          round(scipy.stats.skew(cat['zredmagic'][cond] - cat['zspec'][cond], axis=0, bias=True), 2)))
    i = i+1
    
    plt.axvline(x=0, alpha=0.4, linestyle='--', color='k')
    plt.xlabel('$\delta z(redshift)$', fontsize=16)
    plt.xlim([-0.1, 0.1])
    plt.title("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=18)
    
    # Fitting
    m_dnf = Minuit(fcn_dnf, m1=0.002, m2=-0.002, s1=0.01, s2=0.01, k1=1, k2=1)
    m_dnf.migrad()
        
    chi2_dnf = fcn_dnf(m1=m_dnf.values['m1'],
                       m2=m_dnf.values['m2'],
                       s1=m_dnf.values['s1'],
                       s2=m_dnf.values['s2'],
                       k1=m_dnf.values['k1'],
                       k2=m_dnf.values['k2'])
    dominio = np.linspace(min(bins_dnf), max(bins_dnf), 1000)
    p_value_dnf = stats.ttest_ind(n_dnf, bi_norm(dominio, 
                       m1=m_dnf.values['m1'],
                       m2=m_dnf.values['m2'],
                       s1=m_dnf.values['s1'],
                       s2=m_dnf.values['s2'],
                       k1=m_dnf.values['k1'],
                       k2=m_dnf.values['k2']), equal_var=False)
        
    m_red = Minuit(fcn_red, m1=0.002, m2=-0.002, s1=0.01, s2=0.01, k1=1, k2=1)
    m_red.migrad()
    
    chi2_red = fcn_red(m1=m_red.values['m1'],
                       m2=m_red.values['m2'],
                       s1=m_red.values['s1'],
                       s2=m_red.values['s2'],
                       k1=m_red.values['k1'],
                       k2=m_red.values['k2'])
    
    # Plotting the fit
    p_value_red = stats.ttest_ind(n_red, bi_norm(dominio, m1=m_red.values['m1'],
                        m2=m_red.values['m2'],
                        s1=m_red.values['s1'],
                        s2=m_red.values['s2'],
                        k1=m_red.values['k1'],
                        k2=m_red.values['k2']), equal_var=False)
    
    plt.plot(dominio, bi_norm(dominio, 
                       m1=m_dnf.values['m1'],
                       m2=m_dnf.values['m2'],
                       s1=m_dnf.values['s1'],
                       s2=m_dnf.values['s2'],
                       k1=m_dnf.values['k1'],
                       k2=m_dnf.values['k2']),
                        color='b', label='fit(DNF - Spect.)')
    plt.fill_between(dominio, bi_norm(dominio, m1=m_dnf.values['m1'],
                       m2=m_dnf.values['m2'],
                       s1=m_dnf.values['s1'],
                       s2=m_dnf.values['s2'],
                       k1=m_dnf.values['k1'],
                       k2=m_dnf.values['k2']), color='b', alpha=0.15)
    
    plt.plot(dominio, bi_norm(dominio, m1=m_red.values['m1'],
                       m2=m_red.values['m2'],
                       s1=m_red.values['s1'],
                       s2=m_red.values['s2'],
                       k1=m_red.values['k1'],
                       k2=m_red.values['k2']), 
                       color='y', label='fit(Redmagic - Spect.)')
    plt.fill_between(dominio, bi_norm(dominio, m1=m_red.values['m1'],
                       m2=m_red.values['m2'],
                       s1=m_red.values['s1'],
                       s2=m_red.values['s2'],
                       k1=m_red.values['k1'],
                       k2=m_red.values['k2']), 
                       color='y', alpha=0.15)

    plt.legend(fontsize=8)
    # plt.savefig(os.path.join(cat_path, 'Results', 'using_dnf_z_mean', 'method_comparison_central_bin_iminuit',
    #                           'two_gaussian_fit_{}_{}_centered_bin_iminuit_half_error.png'.format(round(x,2), round(x+0.1,2))),
    #             format='png',
    #             dpi=150,
    #             bbox_inches='tight')
    plt.show()
    print('Redmagic \n')
    for par in m_red.values:
        print(par, ":", np.round(m_red.values[par], 4), '+-', np.round(m_red.errors[par], 4))
    print('chi^2', np.round(chi2_red, 2))
    print('t-value', np.round(p_value_red[0], 2))
    print('p-value', np.round(p_value_red[1], 2))

    print('DNF \n')
    for par in m_dnf.values:
        print(par, ":", np.round(m_dnf.values[par], 4), '+-', np.round(m_dnf.errors[par], 4))
    print('chi^2', np.round(chi2_dnf, 2))
    print('t-value', np.round(p_value_dnf[0], 2))
    print('p-value', np.round(p_value_dnf[1], 2))   
