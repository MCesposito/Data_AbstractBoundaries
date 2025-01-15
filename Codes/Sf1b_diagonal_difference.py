#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:27:52 2024

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
from scipy.spatial import ConvexHull
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

## here i compare the diagonal of participants placement with other dimension diagonal
## and slopes 

## see v2 for plots of placement and diagonals!!!!

general_path = "/home/data/"    ## behav

people = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
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

##

shape_performances = all_participants.groupby(["Participant", "Shape"])["Performance"].mean().reset_index()

perf_by_shape= pd.DataFrame() 
for j in people:
    person_perf = shape_performances.loc[shape_performances["Participant"] == j]
    difference = person_perf["Performance"].values[0] - person_perf["Performance"].values[1]
    
    data=pd.DataFrame({"Participant":[j], "Difference": difference})
    perf_by_shape = pd.concat([perf_by_shape, data])
    
overall_performance = all_participants.groupby(["Participant"])["Performance"].mean().reset_index()
##################### placement data
##############
# divided by axis 
flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 31, 32]
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


## scaling back 
price_min, price_max = 1, 29
freshness_min, freshness_max = 1, 33

# Applying to dataframe
## First: original axes where confidence is price and property is freshness
borders_og["confidence"] = price_min + borders_og["confidence"] * (price_max - price_min)
# freshness
borders_og["property"] = freshness_min + borders_og["property"] * (freshness_max - freshness_min)

## flipped locations 
borders_fl["confidence"] = freshness_min + borders_fl["confidence"] * (freshness_max - freshness_min)
borders_fl["property"] = price_min + borders_fl["property"] * (price_max - price_min)


## loading original positions
og_borders_square = pd.read_csv("/home/data/STR_square.csv", sep=";").iloc[:, 16:].T.reset_index()
og_borders_distorted = pd.read_csv("/home/data/STR_distorted.csv", sep=";").iloc[:, 16:].T.reset_index()

og_borders_square[0] = og_borders_square[0]*5
og_borders_distorted[0] = og_borders_distorted[0]*5

original_positions = (og_borders_square.iloc[:, 1:] + og_borders_distorted.iloc[:, 1:]) /2
original_positions["index"] = og_borders_square["index"]


####################################################################################################################
# calculating diagonals 


all_participants_placements = pd.DataFrame()
whole_place = pd.concat([borders_og, borders_fl])
whole_place['stimulus'] = whole_place['stimulus'].str.replace('piña', 'pina')

diagonals = pd.DataFrame()
for j in people: 
    part_placement = whole_place.loc[whole_place["participant"] == j]
    part_placement['stimulus_clean'] = part_placement['stimulus'].str.lower()
    part_placement['stimulus_clean'] = (
        part_placement['stimulus']
        .str.lower()
        .str.replace(r'-removebg-preview|\d+|_|\(.*?\)', '', regex=True)
        .str.strip()
        .str.replace(r'-+$', '', regex=True))

    original_positions['index_clean'] = original_positions['index'].str.lower().str.strip()
    
    merged_df = pd.merge(part_placement, original_positions, left_on='stimulus_clean', right_on='index_clean', how='left')

    good_df = merged_df.iloc[:, 1:8]
    good_df = good_df.dropna() # got rid of boundaries, since I mapped to just products. 
    
    
    freshness_diagonal = euclidean(good_df.loc[good_df["stimulus"] == "calabacines"][["confidence", "property"]].values.flatten(), good_df.loc[good_df["stimulus"] == "pina"][["confidence", "property"]].values.flatten())
    price_diagonal = euclidean(good_df.loc[good_df["stimulus"] == "fresas"][["confidence", "property"]].values.flatten(), good_df.loc[good_df["stimulus"] == "manzana"][["confidence", "property"]].values.flatten())

    diagonal_difference = pd.DataFrame({"Participant": [j], "Freshness_diagonal": freshness_diagonal, "Price_diagonal": price_diagonal, "Diagonal_difference": freshness_diagonal - price_diagonal})
    
    diagonals = pd.concat([diagonals, diagonal_difference])


dataframe_data = pd.DataFrame({"Diagonal_difference": diagonals["Diagonal_difference"].values, "Participant": people, "Overall performance": overall_performance["Performance"].values})


square_perf= all_participants.loc[all_participants["Shape"] == "square"].groupby(["Participant"])["Performance"].mean().reset_index()
distorted_perf = all_participants.loc[all_participants["Shape"] == "distorted"].groupby(["Participant"])["Performance"].mean().reset_index()




###
## loading original positions
og_borders_square = pd.read_csv("/home/data/STR_square.csv", sep=";").iloc[:, 16:].T.reset_index()
og_borders_distorted = pd.read_csv("/home/data/STR_distorted.csv", sep=";").iloc[:, 16:].T.reset_index()

og_borders_square[0] = og_borders_square[0]*5
og_borders_distorted[0] = og_borders_distorted[0]*5

original_positions = (og_borders_square.iloc[:, 1:] + og_borders_distorted.iloc[:, 1:]) /2
original_positions["index"] = og_borders_square["index"]


# freshness diagonal
og_freshness_diagonal = euclidean(og_borders_distorted.loc[og_borders_distorted["index"] == "calabacines "][[0, 1]].values.flatten(), og_borders_distorted.loc[og_borders_distorted["index"] == "pina"][[0, 1]].values.flatten())
og_price_diagonal = euclidean(og_borders_distorted.loc[og_borders_distorted["index"] == "manzana"][[0, 1]].values.flatten(), og_borders_distorted.loc[og_borders_distorted["index"] == "fresas"][[0, 1]].values.flatten())

diagonals["Freshness_Distance_og"] = diagonals.apply(lambda row: row["Freshness_diagonal"] - og_freshness_diagonal, axis=1)
diagonals["Price_Distance_og"] = diagonals.apply(lambda row: row["Price_diagonal"] - og_price_diagonal, axis=1)

diagonals["Overall_performance"] = overall_performance["Performance"].values

plt.figure()
sns.set_style("whitegrid")
sns.regplot(data=diagonals, x = "Diagonal_difference", y = "Overall_performance")

pr_distances = pd.read_excel("/home/data/procrustes_distances.xlsx")
pr_distances.sort_values(by="Participant", inplace=True)




dataframe_data = pd.DataFrame({"Diagonal_difference": diagonals["Diagonal_difference"].values, "Participant": people, "Overall performance": overall_performance["Performance"].values})

plt.figure()
sns.regplot(data=dataframe_data, x = "Diagonal_difference", y = "Overall performance")
plt.xlabel("Diagonal difference")
plt.ylabel("Task accuracy")
plt.yticks(np.arange(0.60, 0.95, 0.05), [f"{int(i*100)}%" for i in np.arange(0.6, 0.95, 0.05)])


