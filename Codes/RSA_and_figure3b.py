#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:39:01 2024

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
from atlasreader import create_output
from scipy import stats

from sklearn.decomposition import PCA

###########################################################################################
def func_model_rdm(row): 
    if row["Type"] == "novel":
        return 1
    else:
        return 2 

def preparing_regressors(data_be, reg_label):

    if reg_label == "Type": 
        data_be["dummy"] = data_be.apply(lambda row: func_model_rdm(row), axis=1)
        regressor = data_be.groupby(["novels", "Type"])["dummy"].mean().reset_index()
        reg = abs(regressor["dummy"].values - regressor["dummy"].values[:, np.newaxis])
    else: 
        regressor = data_be.groupby(["novels", "Type"])[reg_label].mean().reset_index()
        reg = abs(regressor[reg_label].values - regressor[reg_label].values[:, np.newaxis])
        
    def standardizing(reg):           
        standardizing = StandardScaler().fit_transform(reg)
        flattening =  np.concatenate([standardizing[i, i+1:] for i in range(0, standardizing.shape[0])])        
        return flattening            
    reg_fl = standardizing(reg)

    return reg_fl

def regression(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def GLM(RDM):

    
    RDM_ED_landmark = preparing_regressors(data_be, reg1_label)
    RDM_ED_boundary = preparing_regressors(data_be, reg2_label)
    RDM_trial_distance = preparing_regressors(data_be, reg3_label)
    RDM_type = preparing_regressors(data_be, reg4_label)
            

    X = np.array([RDM_trial_distance, RDM_ED_landmark, RDM_ED_boundary, RDM_type]).squeeze()
    X = X.T
    X = X - X.mean(0)
    X = sm.add_constant(X)  # intercept
    Y = RDM
    return {"Betas":regression(X, Y.T),"Participant":j , "Run":f}


####################################################################################################

general_path = "/home/data/"    ## behav
betas_path = "/home/products_betas/"
roi_path = "/home/data/ROIs/"

people = [2, 3, 4, 6, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
runs = [1, 2, 3, 4]

hc_mask = os.path.join(roi_path, "BilatHippocampalAAL.nii")

mpfc_mask = nib.load(os.path.join(roi_path, "mPFC_final.nii.gz"))

reg1_label = "ED_landmark"
reg2_label= "ED_boundary"
reg3_label = "Trial_absolute"
reg4_label= "Type"

#####################################################################################################

all_data = []

for j in people:
    if j == 1:
        runs = [2, 3, 4]
    elif j == 11:
        runs = [1, 2, 3]
    else: 
        runs = [1, 2, 3, 4]
    for f in runs:
        if len(str(j)) == 1:
            mask_image = os.path.join(general_path, f"sub-0{j}", "func", f"sub-0{j}_task-main_run-0{f}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
            participant = os.path.join(general_path, f"sub-0{j}", "func", f"sub-0{j}_run-0{f}_events.xlsx")
        else:
            mask_image = os.path.join(general_path, f"sub-{j}", "func", f"sub-{j}_task-main_run-0{f}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
            participant = os.path.join(general_path, f"sub-{j}", "func", f"sub-{j}_run-0{f}_events.xlsx")

    # working on images
        beta_images_folder = os.path.join(betas_path, f"betas_{j}_run{f}")
        image_paths = list(glob(f"{beta_images_folder}/beta_*.nii"))
        image_paths.sort()
        
        products_images = [nib.load(path) for path in image_paths]
        participant_mask= nib.load(mask_image)
         
        reaffined_mask = nilearn.image.resample_img(mpfc_mask, target_affine=participant_mask.affine, target_shape = participant_mask.get_fdata().shape, interpolation='nearest', fill_value=0)
        masker = NiftiMasker(mask_img=reaffined_mask)
        masked_arrays = [masker.fit_transform(image) for image in products_images]
        
        reshaped_array = np.array(masked_arrays).reshape(16, len(masked_arrays[0][0]))
        
        
        ##
        #correlation, p_value = stats.spearmanr(principal_components, axis=1)
        correlation = np.corrcoef(reshaped_array)
        
        diss = 1 - correlation# Transforming correlation value into dissimilarity 0-similarity 2-dissimilarity ( same direction to models RDM)
        neural_RDM = np.concatenate([diss[i, i + 1:] for i in range(0, diss.shape[0])]) # Flattern RDM and take just one part of the matrix
    # getting flattened matrices + running the GLM   
        data_be = pd.read_excel(participant)
        
        model_params = GLM(neural_RDM)
        #GLM_betas = getting_betas(model_params, j, f)
        
        all_data.append(model_params)
        
##
data_df = pd.DataFrame()
for index in range(len(all_data)): 
    betas = pd.DataFrame({"Participant": [all_data[index]["Participant"]], "Run": all_data[index]["Run"], "Trial_distance": all_data[index]["Betas"][1], "ED_landmark": all_data[index]["Betas"][2], "ED_boundary": all_data[index]["Betas"][3], "Type": all_data[index]["Betas"][4]})
    data_df = pd.concat([data_df, betas])


parti_df = data_df.groupby(["Participant"]).mean().reset_index()
variables = ["Trial_distance", "ED_landmark", "ED_boundary", "Type"]


for var in variables: 
    t_stat, p_value = stats.ttest_1samp(parti_df[var], 0)
    print(f"{var}: T_stat: {t_stat} - p_value: {p_value}" )

# plotting...

melted_parts = pd.melt(parti_df, id_vars="Participant")


plt.figure()
plt.subplot(2,2,1)
sns.set_style("whitegrid")
sns.barplot(data=melted_parts, x = "variable", y = "value")
sns.stripplot(data=melted_parts, x = "variable", y = "value", color="grey")
plt.xlabel("")
plt.ylabel("Beta coefficients")


