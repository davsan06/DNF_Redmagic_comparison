#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:28:51 2020

@author: davsan06
"""

from astropy.io import fits
import pandas as pd
import numpy as np
# from astropy.table import Table
import os
# import matplotlib.pyplot as plt

# Reading catalogues
redmagic_path = '/scratch/davsan06/REDMAGIC_CATALOGUE/'

cat_photo = fits.getdata(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_photometric.fits'))
cat_photo = pd.DataFrame(np.array(cat_photo).byteswap().newbyteorder())

cat_spec = fits.getdata(os.path.join(redmagic_path, 'redmagic_dr8_public_v6.3_spectroscopic.fits'))
cat_spec = pd.DataFrame(np.array(cat_spec).byteswap().newbyteorder())

# Shuffling and subsetting 5.000 spectroscopic observations
# cat_spec['RA']
# cat_spec.info()

cat_spec = cat_spec.sample(frac = 1)[:5000]

# type(cat_spec['RA'])
# cat_spec.info()

# Analysis of representativeness
# plt.hist(cat_spec['Z_SPEC'], bins = 40, histtype = 'step', label = 'Spectros')
# plt.hist(cat_spec['ZREDMAGIC'], bins = 40, histtype = 'step', label = 'Photo')
# plt.xlabel('z')
# plt.ylabel('$N_{gal}$')
# plt.legend()
# plt.show()

########################################################
### Definition of functions
######################################################################

def directional_neighbour_distance(pri):
    # Computing directional distance
    # pri = photometric row index
    # pri = 3
    mags_photo = np.array([cat_photo['MODEL_MAG_U'].iloc[pri],
                           cat_photo['MODEL_MAG_G'].iloc[pri],
                           cat_photo['MODEL_MAG_R'].iloc[pri],
                           cat_photo['MODEL_MAG_I'].iloc[pri],
                           cat_photo['MODEL_MAG_Z'].iloc[pri]])
    dn = []
    indices = []
    for ind in np.arange(cat_spec.shape[0]):
        # ind = 4999
        mags_spec = np.array([cat_spec['MODEL_MAG_U'].iloc[ind],
                              cat_spec['MODEL_MAG_G'].iloc[ind],
                              cat_spec['MODEL_MAG_R'].iloc[ind],
                              cat_spec['MODEL_MAG_I'].iloc[ind],
                              cat_spec['MODEL_MAG_Z'].iloc[ind]])
        # print(mags_photo - mags_spec)
        dist = sum((mags_spec - mags_photo)**2)*(1 - (np.dot(mags_photo, mags_spec)/(np.dot(mags_photo, mags_photo)*np.dot(mags_spec, mags_spec)))**2)
        dn.append(dist)
        indices.append(np.int(ind))
    return(np.transpose(np.vstack((indices, np.array(dn)))))
    
    # directional_neighbour_distance(0)
    
# plt.plot(directional_neighbour_distance(2))
    
def get_neighbors(pri, num_neighbors):
    # pri = photometric row index
    # num_neighbours
    # pri = 0
    # num_neighbors = 20
    
    # Compute the DN distance between the Photo-row and all the Spectroscopic dataset
    distance = directional_neighbour_distance(pri)
    # Ordering by lower distance in order to get the first 'num_neighbours' neighbors
    # type(distance)
    k_neighbors = distance[distance[:,1].argsort()][:num_neighbors]
    # Return the whole info of the spectroscopic neighbours to compute z_DNF
    cat_spec_neighbours = cat_spec.iloc[k_neighbors[:, 0]]
    cat_spec_neighbours['DN_DIST'] = k_neighbors[:, 1]
    
    return(cat_spec_neighbours)
    
# get_neighbors(pri = 0, num_neighbors = 20)
    
def d_kNN_photoz_estimator(pri, num_neighbors):
    # alpha formula
    # pri = 1
    # num_neighbors = 20
    
    cat_spec_neighbours = get_neighbors(pri, num_neighbors)
    
    # Computing normalization
    alpha = 1/sum(1/cat_spec_neighbours['DN_DIST'])  
    z_photo_knn = alpha*sum(cat_spec_neighbours['Z_SPEC']/cat_spec_neighbours['DN_DIST'])
    
    # Check if Z_KNN column exists, if not, generate it
    if 'Z_KNN' in cat_photo:
        # Inserting infered value into photometric (test) catalogue
        cat_photo['Z_KNN'].iloc[pri] = z_photo_knn
    else:
        # Initializating empty column
        cat_photo['Z_KNN'] = np.zeros(cat_photo.shape[0])
        cat_photo['Z_KNN'].iloc[pri] = z_photo_knn
    
    return(cat_photo)
    
def DNF_photoz_estimator(cat_spec_neighbours):
    # Hyperplane fit methodology
    return()
    
###############################################################################
########            Checking d_KNN photo-z estimator                    #######
###############################################################################
    
for pri in np.arange(10):
    # pri = 0
    
    num_neighbors = 20
    
    cat_photo = d_kNN_photoz_estimator(pri, num_neighbors)
    
    print(cat_photo['ZREDMAGIC'].iloc[pri], cat_photo['Z_KNN'].iloc[pri])
