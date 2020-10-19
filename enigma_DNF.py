#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:58:23 2020

@author: davsan06
"""
# Goal: understand why z_mc is a better redshift estimation than z_mean

from astropy.table import Table
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy import signal

# Reading Redmagic catalogue without train objects
cat_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'
cat = Table.read(os.path.join(cat_path, 'catalogue_redmagic_dnf_wo_train_only_zs.fit'))

cat.info

# Variables
z_mc = cat['DNF_ZMC_SOF']
z_mean = cat['DNF_ZMEAN_SOF']
z_spec = cat['zspec']
z_error = cat['DNF_ZSIGMA_SOF']

###############################################################################
######                      Plotting the differences                  #########
###############################################################################

binning = np.arange(-0.2 - 0.01/4, 0.2 - 0.01/4, 0.01/2)

for x in np.arange(0.1, 1.1, 0.1):
# x = 0.1
    x=round(x,2)

    cond = (z_spec>x)*(z_spec<=x+0.1)

    plt.hist(z_spec[cond] - z_mc[cond],
             bins=binning,
             histtype='step',
             density=True,
             label='Spec. - $z_{mc}$')
    plt.hist(z_spec[cond] - z_mean[cond],
             bins=binning,
             histtype='step',
             density=True,
             label='Spec. - $z_{mean}$')
    plt.title("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=18)
    plt.xlim([-0.2, 0.2])
    plt.xlabel('$\Delta z$', fontsize=14)
    plt.axvline(0, alpha=0.3, color='k')
    plt.legend()
    plt.show()
    
###############################################################################
######                      Plotting N(z)'s                           #########
###############################################################################

    
binning = np.arange(0.05, 1.25, 0.01)

for x in np.arange(0.1, 1.1, 0.1):
# x = 0.1
    x=round(x,2)

    cond = (z_spec>x)*(z_spec<=x+0.1)
    for tipo, lab in zip(['zspec', 'DNF_ZMC_SOF', 'DNF_ZMEAN_SOF'], ['Spectros.', 'DNF_MC', 'DNF_ZMEAN']):
        # tipo='zspec'
        # lab='Spectros.'
        plt.hist(cat[tipo][cond], 
                 bins=binning, 
                 histtype='step',
                 label=lab)
        plt.title("{} $\leq$ z < {}".format(round(x,2), round(x+0.1,2)), fontsize=18)
        plt.xlim([x-0.1, x+0.2])
        plt.xlabel('$z (redshift)$', fontsize=16)
        plt.ylabel('N(z)', fontsize=14)
        plt.legend()
    plt.savefig(os.path.join(cat_path, 'mc_vs_mean_z_spec_bin', 'mc_vs_mean_z_spec_binning_{}_{}.png'.format(round(x,2), round(x+0.1,2))),
                format='png',
                dpi=150,
                bbox_inches='tight')    
    plt.show()

###############################################################################
######                      Bias & dispersion                         ########
###############################################################################

def bias(z_spec, z_one, z_two):
    # Function to compute the bias
    
    # Sample 1
    mu_one = 1/len(z_spec)*sum(z_one - z_spec)
    # Sample 2
    mu_two = 1/len(z_spec)*sum(z_two - z_spec)

    return(mu_one, mu_two)

def dispersion(z_spec, z_one, z_two):
    # Function to compute the dispersion
    
    # Sample 1
    mu_one = bias(z_spec, z_one, z_two)[0]
    sigma_one = np.sqrt(1/len(z_spec)*sum((z_one - z_spec - mu_one)**2))

    # Sample 2
    mu_two = bias(z_spec, z_one, z_two)[1]
    sigma_two = np.sqrt(1/len(z_spec)*sum((z_two - z_spec - mu_two)**2))

    return(sigma_one, sigma_two)  
    
    
bias(z_spec=z_spec, z_one=z_mc, z_two=z_mean)
dispersion(z_spec=z_spec, z_one=z_mc, z_two=z_mean)

###############################################################################
######Relationship between number ob objects and dispersion per z-bin #########                 
###############################################################################

binning = np.arange(-0.2 - 0.01/4, 0.2 - 0.01/4, 0.01/2)

dispersion_mc = np.array([])
dispersion_mean = np.array([])

for x in np.arange(0.1, 1.1, 0.1):
# x = 0.1
    x=round(x,2)

    cond = (z_spec>x)*(z_spec<=x+0.1)

    # Computing disperion
    (disp_mc, disp_mean) = dispersion(z_spec=z_spec[cond], z_one=z_mc[cond], z_two=z_mean[cond])
    
    dispersion_mc = np.append(dispersion_mc, disp_mc)
    dispersion_mean = np.append(dispersion_mean, disp_mean)
    
    print('bias ',bias(z_spec=z_spec[cond], z_one=z_mc[cond], z_two=z_mean[cond]), '\n')
    print('dispersion ', dispersion(z_spec=z_spec[cond], z_one=z_mc[cond], z_two=z_mean[cond]), '\n')

plt.plot(np.arange(0.1, 1.1, 0.1), dispersion_mc, label = 'mc')
plt.plot(np.arange(0.1, 1.1, 0.1), dispersion_mean, label = 'mean')
plt.legend()
plt.show()


###############################################################################
######                          Convolution                           #########                 
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import signal
import seaborn as sns

z_bin_center = np.arange(-2.0, 2.0, 0.01)

n_spec, _, _ = plt.hist(z_spec, bins=z_bin_center, density=True)
n_spec = np.append(n_spec, 0)

n_mean, _, _ = plt.hist(z_mean, bins=z_bin_center, density=True)
n_mean = np.append(n_mean, 0)

n_delta, _, _ = plt.hist(z_mean - z_spec, bins=len(z_bin_center), density=True)

conv_pmf = signal.fftconvolve(n_spec, n_delta,'same')
conv_pmf = conv_pmf/sum(0.01*conv_pmf)

plt.plot(z_bin_center, n_spec, label='Spec')
# plt.plot(z_bin_center, n_delta, label='Delta')
plt.plot(z_bin_center - 0.85*np.ones_like(z_bin_center), conv_pmf, label='Convolucion Spec Mean')
plt.plot(z_bin_center, n_mean, label='Mean')
plt.legend(loc='best'), plt.suptitle('PDFs')
plt.xlim([0.0, 1.25])
plt.show()

n_mc, _, _ = plt.hist(z_mc, bins=z_bin_center, density=True)
n_mc = np.append(n_mc, 0)

n_delta, _, _ = plt.hist(z_mc - z_spec, bins=len(z_bin_center), density=True)

conv_pmf = signal.fftconvolve(n_spec, n_delta,'same')
conv_pmf = conv_pmf/sum(0.01*conv_pmf)

plt.plot(z_bin_center, n_spec, label='Spec')
plt.plot(z_bin_center - 0.82*np.ones_like(z_bin_center), conv_pmf, label='Convolucion Spec MC')
plt.plot(z_bin_center, n_mc, label='MC')
plt.legend(loc='best'), plt.suptitle('PDFs')
plt.xlim([0.0, 1.25])
plt.show()









