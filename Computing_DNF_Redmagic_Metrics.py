#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:43:02 2020

@author: davsan06
"""

from astropy.table import Table, unique
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import curve_fit

# Reading Redmagic matched with DES Y3 Gold catalog
cat_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

# Path to save figures of metric evolution with redshift
path_save_metric_evol = os.path.join(cat_path, 'Results', 'metrics_evolution_redshift')

cat = Table.read(os.path.join(cat_path, 'merge_catalog_desy3gold_redmagic.fits'))
# cat.info()

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

# Variables
z_spec = cat['zspec']
z_red = cat['zredmagic']
z_dnf = cat['DNF_ZMC_SOF']

###############################################################################
## Bias
###############################################################################

def bias(z_spec, z_red, z_dnf):
    # Function to compute the bias
    
    # Redmagic
    mu_red = 1/len(z_spec)*sum(z_red - z_spec)
    # DNF
    mu_dnf = 1/len(z_spec)*sum(z_dnf - z_spec)

    return(mu_red, mu_dnf)

###############################################################################
## Dispersion
###############################################################################

def dispersion(z_spec, z_red, z_dnf):
    # Function to compute the dispersion
    
    # Redmagic
    mu_red = bias(z_spec, z_red, z_dnf)[0]
    sigma_red = np.sqrt(1/len(z_spec)*sum((z_red - z_spec - mu_red)**2))

    # DNF
    mu_dnf = bias(z_spec, z_red, z_dnf)[1]
    sigma_dnf = np.sqrt(1/len(z_spec)*sum((z_dnf - z_spec - mu_dnf)**2))

    return(sigma_red, sigma_dnf)   
    
###############################################################################
## Bias & Dispersion via Fit
###############################################################################

def gaussian(x, A, mu, sigma):
    return(A*np.exp(-1.0*(x-mu)**2/(2*sigma**2)))

def bias_dispersion_error(z_spec, z_red, z_dnf):
    
    # Data to fit
    delta_red = z_spec - z_red
    delta_dnf = z_spec - z_dnf

    bins = np.arange(-0.2, 0.2, 0.01)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    data_red, bins_red = np.histogram(delta_red, bins=bins)    
    data_dnf, bins_dnf = np.histogram(delta_dnf, bins=bins)

    # Fit the function to the histogram data.
    params_red, cov_red = curve_fit(gaussian, xdata=binscenters, ydata=data_red)
    params_dnf, cov_dnf = curve_fit(gaussian, xdata=binscenters, ydata=data_dnf)
    
    (mu_red_err, disp_red_err) = [np.sqrt(cov_red[1,1]), np.sqrt(cov_red[2,2])]
    (mu_dnf_err, disp_dnf_err) = [np.sqrt(cov_dnf[1,1]), np.sqrt(cov_dnf[2,2])]

    return((mu_red_err, disp_red_err), (mu_dnf_err, disp_dnf_err))
    
###############################################################################
## Precision in 68-quantile
###############################################################################
    
def sigma_68(z_spec, z_red, z_dnf, show_plots):
    # show_plots = True/False
    
    # Redmagic
    delta_z_red = np.array(z_red - z_spec)
    
    # sort the data:
    delta_z_red_sorted = np.sort(delta_z_red)

    # calculate the proportional values of samples
    p_red = 1. * np.arange(len(delta_z_red)) / (len(delta_z_red) - 1)
    
    # Interpolate CDF to get percentiles
    f_red = lambda x: np.interp(x, p_red, delta_z_red_sorted)

    sigma_68_red = 0.5*(f_red(0.84) - f_red(0.16))
    
    # DNF
    delta_z_dnf = np.array(z_dnf - z_spec)
    
    # sort the data:
    delta_z_dnf_sorted = np.sort(delta_z_dnf)
    
    # calculate the proportional values of samples
    p_dnf = 1. * np.arange(len(delta_z_dnf)) / (len(delta_z_dnf) - 1)
    
    # Interpolate CDF to get percentiles
    f_dnf = lambda x: np.interp(x, p_dnf, delta_z_dnf_sorted)

    sigma_68_dnf = 0.5*(f_dnf(0.84) - f_dnf(0.16))
    
    if (show_plots == True):
    
        # plot the sorted data:
        fig = plt.figure(figsize=[8, 4])
        fig.suptitle('RedMaGic', fontsize=16)
        ax1 = fig.add_subplot(121)
        ax1.hist(delta_z_red, bins=40, range=[-0.15, 0.15], histtype='step')
        ax1.set_xlabel('$\Delta z$', fontsize=14)
        ax1.set_ylabel('$freq(\Delta z)$', fontsize=14)

        ax2 = fig.add_subplot(122)
        ax2.plot(delta_z_red_sorted, p_red, label="$\sigma_{68}$ = %1.3f" % sigma_68_red)
        ax2.set_xlim([-0.15, 0.15])
        ax2.set_xlabel('$CDF(\Delta z)$', fontsize=14)
        ax2.legend()
    
        # plot the sorted data:
        fig = plt.figure(figsize=[8, 4])
        fig.suptitle('DNF', fontsize=16)
        ax1 = fig.add_subplot(121)
        ax1.hist(delta_z_dnf, bins=40, range=[-0.15, 0.15], histtype='step')
        ax1.set_xlabel('$\Delta z$', fontsize=14)
        ax1.set_ylabel('$freq(\Delta z)$', fontsize=14)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(delta_z_dnf_sorted, p_dnf, label="$\sigma_{68}$ = %1.3f" % sigma_68_dnf)
        ax2.set_xlim([-0.15, 0.15])
        ax2.set_xlabel('$CDF(\Delta z)$', fontsize=14)
        ax2.legend()
    
    return(sigma_68_red, sigma_68_dnf)
    
# sigma_68(z_spec, z_red, z_dnf, 1)
    
###############################################################################
## Fraction of outlier to n sigmas
###############################################################################
    
def fraction_outliers_n_sigmas(z_spec, z_red, z_dnf, n_sigmas):
    # n_sigmas = 2 or 3 (paper DNF)
    # n_sigmas = 3
    
    # Redmagic
    delta_z_red = np.array(z_red - z_spec)
    bias_red = bias(z_spec, z_red, z_dnf)[0]
    dispersion_red = dispersion(z_spec, z_red, z_dnf)[0]
    
    lhs_red = np.abs(delta_z_red - bias_red*np.ones(len(delta_z_red)))
    
    W_red = 1*(lhs_red > n_sigmas*dispersion_red*np.ones(len(lhs_red)))
    
    f_sigma_red = 1/len(delta_z_red)*sum(W_red)
    
    # DNF
    delta_z_dnf = np.array(z_dnf - z_spec)
    bias_dnf = bias(z_spec, z_red, z_dnf)[1]    
    dispersion_dnf = dispersion(z_spec, z_red, z_dnf)[1]

    lhs_dnf = np.abs(delta_z_dnf - bias_dnf*np.ones(len(delta_z_dnf)))     
    
    W_dnf = 1*(lhs_dnf > n_sigmas*dispersion_dnf*np.ones(len(lhs_dnf)))
    
    f_sigma_dnf = 1/len(delta_z_dnf)*sum(W_dnf)
    
    return(f_sigma_red, f_sigma_dnf)
    
###############################################################################
## Poisson
###############################################################################     
    
def poisson(z_spec, z_red, z_dnf, n_bins):
    # Function to compute distance between bin distributions: photo vs spec
    # To have bin width = 0.05 in 0.0 < z < 1.1 we need
    # n_bins = 40
    
    # Spectroscopic
    hist_spec = np.histogram(z_spec, 
                             bins=n_bins, 
                             range=[0.0, 1.1], 
                             density=True)[0]
    # Redmagic
    hist_red = np.histogram(z_red, 
                            bins=n_bins, 
                            range=[0.0, 1.1], 
                            density=True)[0]
    # DNF
    hist_dnf = np.histogram(z_dnf,
                            bins=n_bins,
                            range=[0.0, 1.1],
                            density=True)[0]
    
    n_poisson_red = np.sqrt(1/n_bins*sum(np.divide((hist_red - hist_spec)**2, hist_spec, 
                                                   out=np.zeros_like(hist_spec), where=hist_spec!=0)))
    n_poisson_dnf = np.sqrt(1/n_bins*sum(np.divide((hist_dnf - hist_spec)**2, hist_spec, 
                                                   out=np.zeros_like(hist_spec), where=hist_spec!=0)))
    
    return(n_poisson_red, n_poisson_dnf)
###############################################################################
## Kolmogorov-Smirnov test
###############################################################################        
    
# def ks_test(z_spec, z_red, z_dnf):
#    
#    domain = np.linspace(0, 1, 1000)
#
#    # Compute cumulative distributions    
#    z_spec_sorted = np.sort(z_spec)
#    p_spec = 1. * np.arange(len(z_spec)) / (len(z_spec) - 1)
#    f_spec = lambda x: np.interp(x, p_spec, z_spec_sorted)
#    f_spec_eval = np.array(list(map(f_spec, domain)))
    # plt.plot(f_spec_eval, domain)
    
#    z_red_sorted = np.sort(z_red)
#    p_red = 1. * np.arange(len(z_red)) / (len(z_red) - 1)
#    f_red = lambda x: np.interp(x, p_red, z_red_sorted)
#    f_red_eval = np.array(list(map(f_red, domain)))
    # plt.plot(domain, f_red_eval)
    
#    z_dnf_sorted = np.sort(z_dnf)
#    p_dnf = 1. * np.arange(len(z_dnf)) / (len(z_dnf) - 1)       
#    f_dnf = lambda x: np.interp(x, p_dnf, z_dnf_sorted)    
#    f_dnf_eval = np.array(list(map(f_dnf, domain)))    
    # plt.plot(domain, f_dnf_eval)    
    
#    ks_red = max(np.abs(f_red_eval - f_spec_eval))
#    ks_dnf = max(np.abs(f_dnf_eval - f_spec_eval))
    
#    return(ks_red, ks_dnf)
    
def ks_test_v2(z_spec, z_red, z_dnf):
    
    # Gnerating reference z array
    z_fid = np.linspace(min(z_spec), max(z_spec), len(z_spec))

    f_cum_spec = np.cumsum(1.0*(np.sort(z_spec) < z_fid))
    f_cum_spec = 100*f_cum_spec/max(f_cum_spec)
    f_cum_red = np.cumsum(1.0*(np.sort(z_red) < z_fid))
    f_cum_red = 100*f_cum_red/max(f_cum_red)
    f_cum_dnf = np.cumsum(1.0*(np.sort(z_dnf) < z_fid))
    f_cum_dnf = 100*f_cum_dnf/max(f_cum_dnf)

    ks_red = max(abs(f_cum_red - f_cum_spec))
    ks_dnf = max(abs(f_cum_dnf - f_cum_spec))
    
    return(ks_red, ks_dnf)
    
###############################################################################
## Results
############################################################################### 
    
mu_red, mu_dnf = bias(z_spec, z_red, z_dnf)
sigma_red, sigma_dnf = dispersion(z_spec, z_red, z_dnf)
(mu_red_err, disp_red_err), (mu_dnf_err, disp_dnf_err) = bias_dispersion_error(z_spec, z_red, z_dnf)
sigma_68_red, sigma_68_dnf = sigma_68(z_spec, z_red, z_dnf, show_plots=True)
f_2s_red, f_2s_dnf = fraction_outliers_n_sigmas(z_spec, z_red, z_dnf, n_sigmas=2)
f_3s_red, f_3s_dnf = fraction_outliers_n_sigmas(z_spec, z_red, z_dnf, n_sigmas=3)
n_poisson_red, n_poisson_dnf = poisson(z_spec, z_red, z_dnf, n_bins=22)
ks_red, ks_dnf = ks_test_v2(z_spec, z_red, z_dnf)


fig, axs = plt.subplots(2, 2, sharey=False, figsize=(12, 8))

axs[0, 0].errorbar(x=mu_red, y=sigma_red,
                   xerr=mu_red_err, yerr=disp_red_err,
                   ms=8,
                   lw=1,
                   marker="s",
                   label="Redmagic")
axs[0, 0].errorbar(x=mu_dnf, y=sigma_dnf,
                   xerr=mu_dnf_err, yerr=disp_dnf_err,
                   ms=8,
                   lw=1,
                   marker="d",   
                   label='DNF')
axs[0, 0].set_xlabel('$\mu$', fontsize=16)
axs[0, 0].set_ylabel('$\sigma$', fontsize=16)
# axs[0, 0].set_xlim([-0.004, 0.001])
# axs[0, 0].set_ylim([0.036, 0.043])
axs[0, 0].axvline(x=0, color='black', alpha=0.3)
axs[0, 0].grid(alpha=0.3)
axs[0, 0].legend()

axs[0, 1].errorbar(x=mu_red, y=sigma_68_red,
                   xerr=mu_red_err,
                   ms=8,
                   lw=1,
                   marker="s",
                   label="Redmagic")
axs[0, 1].errorbar(x=mu_dnf, y=sigma_68_dnf,
                   xerr=mu_dnf_err,
                   ms=8,
                   lw=1,
                   marker='d',
                   label='DNF')
axs[0, 1].set_xlabel('$\mu$', fontsize=16)
axs[0, 1].set_ylabel('$\sigma_{68}$', fontsize=16)
axs[0, 1].set_xlim([-0.005, 0.002])
axs[0, 1].set_ylim([0.015, 0.03])
axs[0, 1].axvline(x=0, color='black', alpha=0.3)
axs[0, 1].grid(alpha=0.3)
axs[0, 1].legend()

axs[1, 0].scatter(f_2s_red, f_3s_red,
                   s=80,
                   marker="s",
                   label="Redmagic")
axs[1, 0].scatter(f_2s_dnf, f_3s_dnf,
                   s=80,
                   marker="d",
                   label="DNF")
axs[1, 0].set_xlabel('$f_{2\sigma}$', fontsize=16)
axs[1, 0].set_ylabel('$f_{3\sigma}$', fontsize=16)
axs[1, 0].set_xlim([0.024, 0.030])
axs[1, 0].set_ylim([0.01, 0.0115])
axs[1, 0].grid(alpha=0.3)
axs[1, 0].legend()

axs[1, 1].scatter(x=ks_red, y=n_poisson_red,
                   s=80,
                   marker="s",
                   label="Redmagic")
axs[1, 1].scatter(ks_dnf, n_poisson_dnf,
                   s=80,
                   marker="d",
                   label="DNF")
axs[1, 1].set_xlabel("$ks$", fontsize=16)
axs[1, 1].set_ylabel("$N_{Poisson}$", fontsize=16)


axs[1, 1].grid(alpha=0.3)
axs[1, 1].legend()

# plt.savefig(os.path.join(cat_path, 'Results', 'metrics_v2.png'),
#             format='png',
#             dpi=150)
plt.show()

###############################################################################
## Results per redshift bin
############################################################################### 

for x in np.arange(0.1, 1.1, 0.1):
    # x = 0.1
    x = round(x, 2)
    
    cond = (cat['zspec']>=x)*(cat['zspec']<x+0.1)
    
    mu_red, mu_dnf = bias(z_spec[cond], z_red[cond], z_dnf[cond])
    sigma_red, sigma_dnf = dispersion(z_spec[cond], z_red[cond], z_dnf[cond])
    (mu_red_err, disp_red_err), (mu_dnf_err, disp_dnf_err) = bias_dispersion_error(z_spec[cond], z_red[cond], z_dnf[cond])
    sigma_68_red, sigma_68_dnf = sigma_68(z_spec[cond], z_red[cond], z_dnf[cond], show_plots=True)
    f_2s_red, f_2s_dnf = fraction_outliers_n_sigmas(z_spec[cond], z_red[cond], z_dnf[cond], n_sigmas=2)
    f_3s_red, f_3s_dnf = fraction_outliers_n_sigmas(z_spec[cond], z_red[cond], z_dnf[cond], n_sigmas=3)
    n_poisson_red, n_poisson_dnf = poisson(z_spec[cond], z_red[cond], z_dnf[cond], n_bins = 40)
    ks_red, ks_dnf = ks_test_v2(z_spec[cond], z_red[cond], z_dnf[cond])
    
    fig, axs = plt.subplots(2, 2, sharey=False, figsize=(12, 8))

    axs[0, 0].errorbar(x=mu_red, y=sigma_red,
                       xerr=mu_red_err, yerr=disp_red_err,
                       ms=8,
                       lw=1,
                       marker="s",
                       label="Redmagic")
    axs[0, 0].errorbar(x=mu_dnf, y=sigma_dnf,
                       xerr=mu_dnf_err, yerr=disp_dnf_err,
                       ms=8,
                       lw=1,
                       marker="d",   
                           label='DNF')
    axs[0, 0].set_xlabel('$\mu$', fontsize=16)
    axs[0, 0].set_ylabel('$\sigma$', fontsize=16)
    # axs[0, 0].set_xlim([-0.004, 0.001])
    # axs[0, 0].set_ylim([0.036, 0.043])
    axs[0, 0].axvline(x=0, color='black', alpha=0.3)
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend()
    
    axs[0, 1].errorbar(x=mu_red, y=sigma_68_red,
                       xerr=mu_red_err,
                       ms=8,
                       lw=1,
                       marker="s",
                       label="Redmagic")
    axs[0, 1].errorbar(x=mu_dnf, y=sigma_68_dnf,
                       xerr=mu_dnf_err,
                       ms=8,
                       lw=1,
                       marker='d',
                       label='DNF')
    axs[0, 1].set_xlabel('$\mu$', fontsize=16)
    axs[0, 1].set_ylabel('$\sigma_{68}$', fontsize=16)
    # axs[0, 1].set_xlim([-0.005, 0.002])
    # axs[0, 1].set_ylim([0.015, 0.03])
    axs[0, 1].axvline(x=0, color='black', alpha=0.3)
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend()
    
    axs[1, 0].scatter(f_2s_red, f_3s_red,
                       s=80,
                       marker="s",
                       label="Redmagic")
    axs[1, 0].scatter(f_2s_dnf, f_3s_dnf,
                       s=80,
                       marker="d",
                       label="DNF")
    axs[1, 0].set_xlabel('$f_{2\sigma}$', fontsize=16)
    axs[1, 0].set_ylabel('$f_{3\sigma}$', fontsize=16)
    # axs[1, 0].set_xlim([0.024, 0.030])
    # axs[1, 0].set_ylim([0.01, 0.0115])
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend()
    
    axs[1, 1].scatter(x=ks_red, y=n_poisson_red,
                       s=80,
                       marker="s",
                       label="Redmagic")
    axs[1, 1].scatter(ks_dnf, n_poisson_dnf,
                       s=80,
                       marker="d",
                       label="DNF")
    axs[1, 1].set_xlabel("$ks$", fontsize=16)
    axs[1, 1].set_ylabel("$N_{Poisson}$", fontsize=16)
    
    plt.suptitle("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=22)
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend()
    
    plt.savefig(os.path.join(cat_path, 'Results', 'metrics_v2_bin_{}_{}.png'.format(round(x,2), round(x+0.1,2))),
                format='png',
                dpi=150)
    plt.show()
    
    
###############################################################################
## Evolution of metrics with redshift
############################################################################### 
    
bin_edges = np.arange(0.1, 1.1, 0.1) 
bin_centers = np.append(np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)]), 1.05)
    
mu_red_array = np.array([]) # Its error will be appended here
mu_dnf_array = np.array([])

sigma_red_array = np.array([]) # Its error will be appended here
sigma_dnf_array = np.array([])

sigma_68_red_array = np.array([])
sigma_68_dnf_array = np.array([])

f_2s_red_array = np.array([])
f_2s_dnf_array = np.array([])

f_3s_red_array = np.array([])
f_3s_dnf_array = np.array([])

n_poisson_red_array = np.array([])
n_poisson_dnf_array = np.array([])

ks_red_array = np.array([])
ks_dnf_array = np.array([])

for x in bin_edges:
    # x = 0.1
    x = round(x, 2)
    
    cond = (cat['zspec']>=x)*(cat['zspec']<x+0.1)
    
    mu_red, mu_dnf = bias(z_spec[cond], z_red[cond], z_dnf[cond])
    sigma_red, sigma_dnf = dispersion(z_spec[cond], z_red[cond], z_dnf[cond])
    (mu_red_err, disp_red_err), (mu_dnf_err, disp_dnf_err) = bias_dispersion_error(z_spec[cond], z_red[cond], z_dnf[cond])
    sigma_68_red, sigma_68_dnf = sigma_68(z_spec[cond], z_red[cond], z_dnf[cond], show_plots=False)
    f_2s_red, f_2s_dnf = fraction_outliers_n_sigmas(z_spec[cond], z_red[cond], z_dnf[cond], n_sigmas=2)
    f_3s_red, f_3s_dnf = fraction_outliers_n_sigmas(z_spec[cond], z_red[cond], z_dnf[cond], n_sigmas=3)
    n_poisson_red, n_poisson_dnf = poisson(z_spec[cond], z_red[cond], z_dnf[cond], n_bins = 40)
    ks_red, ks_dnf = ks_test_v2(z_spec[cond], z_red[cond], z_dnf[cond])   
    
    mu_red_array = np.append(mu_red_array, [mu_red, mu_red_err])  
    mu_dnf_array = np.append(mu_dnf_array, [mu_dnf, mu_dnf_err])

    sigma_red_array = np.append(sigma_red_array, [sigma_red, disp_red_err])    
    sigma_dnf_array = np.append(sigma_dnf_array, [sigma_dnf, disp_dnf_err])    

    sigma_68_red_array = np.append(sigma_68_red_array, sigma_68_red)    
    sigma_68_dnf_array = np.append(sigma_68_dnf_array, sigma_68_dnf)    

    f_2s_red_array = np.append(f_2s_red_array, f_2s_red)    
    f_2s_dnf_array = np.append(f_2s_dnf_array, f_2s_dnf)    

    f_3s_red_array = np.append(f_3s_red_array, f_3s_red)    
    f_3s_dnf_array = np.append(f_3s_dnf_array, f_3s_dnf)    

    n_poisson_red_array = np.append(n_poisson_red_array, n_poisson_red)
    n_poisson_dnf_array = np.append(n_poisson_dnf_array, n_poisson_dnf)

    ks_red_array = np.append(ks_red_array, ks_red)    
    ks_dnf_array = np.append(ks_dnf_array, ks_dnf)    
    
# Reshape mu and sigma in order to have two columns: first with value, second with error
mu_red_array = np.reshape(mu_red_array, [np.int(len(mu_red_array)/2),2])
mu_dnf_array = np.reshape(mu_dnf_array, [np.int(len(mu_dnf_array)/2),2])
    
sigma_red_array = np.reshape(sigma_red_array, [np.int(len(sigma_red_array)/2),2])
sigma_dnf_array = np.reshape(sigma_dnf_array, [np.int(len(sigma_dnf_array)/2),2])    
    
# Plot bias
plt.errorbar(x=bin_centers, y=mu_red_array[:,0],
             yerr=mu_red_array[:,1],
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=mu_dnf_array[:,0],
             yerr=mu_dnf_array[:,1],
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('bias', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$\mu$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'mu_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot sigma
plt.errorbar(x=bin_centers, y=sigma_red_array[:,0],
             yerr=sigma_red_array[:,1],
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=sigma_dnf_array[:,0],
             yerr=sigma_dnf_array[:,1],
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('dispersion', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$\sigma$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'sigma_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot sigma_68
plt.errorbar(x=bin_centers, y=sigma_68_red_array,
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=sigma_68_dnf_array,
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('dispersion perc. 68', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$\sigma_{68}$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'sigma_68_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot outliers 2 sigma
plt.errorbar(x=bin_centers, y=f_2s_red_array,
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=f_2s_dnf_array,
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('outliers 2$\sigma$', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$f_{2\sigma}$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'f_2sigma_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot outliers 3 sigma
plt.errorbar(x=bin_centers, y=f_3s_red_array,
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=f_3s_dnf_array,
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('outliers 3$\sigma$', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$f_{3\sigma}$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'f_3sigma_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot N Poisson (at high z Redmagic distr is more similar to real 
# distribution due to the spare train sample at high z in DNF)
plt.errorbar(x=bin_centers, y=n_poisson_red_array,
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=n_poisson_dnf_array,
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('Poisson distance', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$N_{Poisson}$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'n_poisson_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()

# Plot KS
plt.errorbar(x=bin_centers, y=ks_red_array,
             ms=5,
             lw=1,
             marker="s",
             label='Redmagic')
plt.errorbar(x=bin_centers, y=ks_dnf_array,
             ms=5,
             lw=1,
             marker="d",
             label='DNF')
plt.title('KS Test', fontsize=16)
plt.xticks(bin_centers)
plt.xlabel('z (redshift)', fontsize=14)
plt.ylabel('$ks$', fontsize=14)
plt.legend()
plt.savefig(os.path.join(path_save_metric_evol, 'ks_test_z.png'),
            dpi=150, 
            bbox_inches='tight')
plt.show()










    