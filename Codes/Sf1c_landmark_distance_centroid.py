#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:55:59 2024

@author: mariachiara
"""

    
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

from sklearn.metrics import pairwise_distances

general_path = "/home/data/"    ## behav


# loading behavioral data
people = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
runs = [1, 2,3,4]

all_participants = pd.DataFrame()
for j in people:
    for f in runs: 

        if len(str(j)) == 1:
            participant = pd.read_excel(os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx"))
        else:
            participant = pd.read_excel(os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx"))
        participant["Participant"] = j
        all_participants = pd.concat([all_participants, participant])
        
        
        
        
# loading participants placement

# divided by axis 
flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33, 34]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32]
##########################
#######################################################
            # function to get the borders and the whole placement of all participants
def getting_placement(participants_set, cond): 
    all_parts_placement = pd.DataFrame()
    all_parts_borders = pd.DataFrame()

    if cond == "flipped":
        for j in participants_set: 
            placement = pd.read_csv(f"/home/data/drag-and-rate/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[2, 7, 19, 6]] = ["calabacines", "fresas", "manzana", "piña"]
            borders = placement.iloc[[2, 7, 19, 6]].reset_index()
            
            all_parts_borders = pd.concat([all_parts_borders, borders])   #flipped
            
    else: 
        for j in participants_set:
            placement = pd.read_csv(f"/home/data/drag-and-rate/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[4, 12, 3, 2]] = ["manzana", "calabacines", "fresas", "piña"]
            borders = placement.iloc[[4, 12, 3, 2]].reset_index()
            all_parts_borders = pd.concat([all_parts_borders, borders])    
            
    return all_parts_placement, all_parts_borders
        
        
placement_fl, borders_fl = getting_placement(flipped, "flipped")
placement_og, borders_og = getting_placement(unflipped, "unflipped")

landmark_og = placement_og.loc[placement_og["stimulus"].str.contains("tomate")]
landmark_fl = placement_fl.loc[placement_fl["stimulus"].str.contains("tomate")]

all_landmarks = pd.concat([landmark_og, landmark_fl])
all_borders = pd.concat([borders_og, borders_fl])


distance_centroid =[]
for j in people:
    
    landmark_placement = all_landmarks.loc[all_landmarks["participant"] == j][["confidence","property"]].values[0]
    
    borders_placement = all_borders.loc[all_borders["participant"] == j][["confidence", "property"]].values

    part_centroid = np.mean(borders_placement[:, 0]), np.mean(borders_placement[:, 1])
    
    distance_centroid.append(euclidean(landmark_placement, part_centroid))
    
    
perf = all_participants.groupby(["Participant"])["Performance"].mean().reset_index()


plt.figure(figsize=(5,4))
sns.regplot(x = distance_centroid, y = perf["Performance"])
plt.xlabel("Distance of landmark placement \nfrom reconstructed centroid", size=13)
plt.ylabel("Task Accuracy", size=13)
plt.yticks(np.arange(0.55, 0.95, 0.05), [f"{int(i*100)}%" for i in np.arange(0.55, 0.95, 0.05)])
plt.tight_layout()