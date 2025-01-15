#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:52:14 2024

@author: mariachiara
"""


import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
from glob import glob
import os 

import numpy as np
import time
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns 

from pathlib import Path
from shutil import copyfile

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.metrics import pairwise_distances
import nibabel as nib
import nilearn  
from nilearn import plotting
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.decoding import SearchLight
from nilearn import image
from nilearn.image import new_img_like
from nilearn.plotting import plot_img

import statsmodels.api as sm

import rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed

from scipy.stats import distributions
from patsy import dmatrices

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from nilearn.regions import connected_regions
from nilearn.image import threshold_img
from nilearn import reporting

from scipy import stats
from scipy.spatial.distance import euclidean 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from scipy.spatial import procrustes 
from sklearn.metrics import pairwise_distances
from scipy.spatial import ConvexHull

import math
from matplotlib import gridspec





general_path = "/home/data/"    ## behav

people = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
runs = [1, 2, 3, 4]


all_participants = pd.DataFrame()
for j in people:
    for f in runs: 

        if len(str(j)) == 1:
            participant = pd.read_excel(os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx"))
        else:
            participant = pd.read_excel(os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx"))
        participant["Participant"] = j
        all_participants = pd.concat([all_participants, participant])

single_part_perf = all_participants.groupby(["Participant"])["Performance"].mean().reset_index()
##################### placement data
##############
# divided by axis 
flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 34]
##########################
#######################################################
            # function to get the borders and the whole placement of all participants
def getting_placement(participants_set, cond): 
    all_parts_placement = pd.DataFrame()
    all_parts_borders = pd.DataFrame()

    if cond == "flipped":
        for j in participants_set: 
            placement = pd.read_csv(f"/home/mariachiara/Escritorio/fMRI/behavior/Meadows/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[2, 7, 19, 6]] = ["calabacines", "fresas", "manzana", "piña"]
            borders = placement.iloc[[2, 7, 19, 6]].reset_index()
            
            all_parts_borders = pd.concat([all_parts_borders, borders])   #flipped
            
    else: 
        for j in participants_set:
            placement = pd.read_csv(f"/home/mariachiara/Escritorio/fMRI/behavior/Meadows/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[4, 12, 3, 2]] = ["manzana", "calabacines", "fresas", "piña"]
            borders = placement.iloc[[4, 12, 3, 2]].reset_index()
            all_parts_borders = pd.concat([all_parts_borders, borders])    
            
    return all_parts_placement, all_parts_borders


placement_fl, borders_fl = getting_placement(flipped, "flipped")
placement_og, borders_og = getting_placement(unflipped, "unflipped")

all_placement= pd.concat([placement_og, placement_fl])
all_borders = pd.concat([borders_og, borders_fl])

# excluding boundaries
exclude_stimulus = ["3_fresas","manzana-removebg-preview", "1_calabacines (1) (1)", "tomate (1) (1)","Captura_de_pantalla_2023-03-14_112329-removebg-preview", "1_calabacines (1)"]
products_placement = all_placement.loc[~all_placement["stimulus"].isin(exclude_stimulus)]

landmark_placement = all_placement.loc[all_placement["stimulus"].str.contains("tomate")]

# checking whether they cluster the placement around THEIR LANDMARK// THEIR CENTROID

products_from_center = pd.DataFrame()
# t-test per participant + group level t-test
stats_df  = pd.DataFrame() 
for j in people: 
    part_products = products_placement.loc[products_placement["participant"] == j]
    #landmark = landmark_placement.loc[landmark_placement["participant"] == j][["confidence", "property"]].values[0]
    #landmark = [0.5,0.7]
    # distance from the boundaries
    part_borders = all_borders.loc[all_borders["participant"] == j]
    border_coords = part_borders[["confidence", "property"]].values
    centroid = np.mean(border_coords[:, 0]), np.mean(border_coords[:, 1])
    distances_data = []
    for border_placement in range(len(part_borders)):
        border_distances = []
        
        border_coords = part_borders[["confidence", "property"]].values[border_placement]
        for product_placement in range(len(part_products)):
            product_coords = part_products[["confidence", "property"]].values[product_placement]
            
            border_distances.append(euclidean(product_coords, border_coords))
        distances_data.append(border_distances)
    
    # getting a vector with the minimum value for each product
    array= np.array(distances_data)
    min_values = np.min(array, axis=0)
    
    
    ## distance from the center of the space
    center_distances = []

    for product_placement in range(len(part_products)):
        product_coords = part_products[["confidence", "property"]].values[product_placement]
    
        center_dist= euclidean(product_coords, centroid)
        center_distances.append(center_dist)
    
    dist_cent =pd.DataFrame({"Participant": j, "Distance": center_distances})
    products_from_center = pd.concat([products_from_center, dist_cent])

    stat = pd.DataFrame({"Participant": [j], "Stat": stats.ttest_rel(min_values, center_distances)[0]})
    stats_df = pd.concat([stats_df ,stat])


print(stats.ttest_1samp(stats_df["Stat"], 0))






