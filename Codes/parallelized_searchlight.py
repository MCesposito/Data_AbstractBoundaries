#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:32:34 2023

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

general_path = "/home/data/"    ## behav
betas_path = "/home/data/product_betas/"


people = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
run = [1, 2, 3, 4]


reg1_label = "Trial_absolute"
reg2_label = "ED_landmark"
reg3_label = "ED_boundary"
reg4_label = "Type"

def func_model_rdm(row): 
    if row["Type"] == "novel":
        return 1
    else:
        return 2 
## define a function to get behavioral RDMs

def preparing_regressors(data_be, reg_label):


    if reg_label == "Type": 
        regressor = data_be.groupby(["novels", "Type"]).mean().reset_index()
        regressor["dummy"] = regressor.apply(lambda row: func_model_rdm(row), axis=1)
        reg = abs(regressor["dummy"].values - regressor["dummy"].values[:, np.newaxis])
    else: 
        regressor = data_be.groupby(["novels", "Type"])[reg_label].mean().reset_index()
        reg = abs(regressor[reg_label].values - regressor[reg_label].values[:, np.newaxis])
        
    def standardizing(reg):           
        standardizing = StandardScaler().fit_transform(reg)
        flattening =  np.concatenate([reg[i, i+1:] for i in range(0, reg.shape[0])])        
        return flattening            
    reg_fl = standardizing(reg)

    return reg_fl



## define the searchlight function 


def Searchlight_GLM(f, j):
    if len(str(j)) == 1:
        mask_image = os.path.join(general_path, f"sub-0{j}", "func", f"sub-0{j}_task-main_run-0{f}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
        participant = os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx")
    else:
        mask_image = os.path.join(general_path, f"sub-{j}", "func", f"sub-{j}_task-main_run-0{f}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
        participant = os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx")

    beta_images_folder = os.path.join(betas_path, f"betas_{j}_run{f}")
    image_paths = list(glob(f"{beta_images_folder}/beta_*.nii"))
    image_paths.sort()
    
    
    #### loading the mask 
    mask_brain = nib.load(mask_image)
    
    ## Get mask data
    mask = (mask_brain.get_fdata())
    x, y, z = mask_brain.get_fdata().shape
    
    ## 
    ## Loading event files
    data_be = pd.read_excel(participant)
    

    
    data = np.zeros((len(image_paths), x, y, z)) 
    for x, im in enumerate(image_paths):
        data[x] = nib.load(im).get_fdata()[:, :, :, 0]
    
    image_value = np.arange(len(image_paths))
    
    data_2d = data.reshape([data.shape[0], -1])   ## NB. Functional data has to be 2d! 
    data_2d = np.nan_to_num(data_2d)
     
    
    radius = 4   ## NB. rsatoolbox measures radius in VOXELS IN THE SEARCHLIGHT SPHERE, not millimeters! **** 
                 #      A 4 voxel radius means a 10 mm radius (dim: 2.5 x 2.5 x 2.5) 
    center, neighbor = get_volume_searchlight(mask, radius = radius, threshold = 0.8)  ## At least 80% of the neighboring voxels 
                                                                                  # need to be within the brain mask

    SL_RDM = get_searchlight_RDMs(data_2d, center, neighbor, image_value, method='correlation') #  Voxel ( correlation con 80000 voxels) RDM [0,4,0.5,0.7,0..  8]


    ## getting flattened matrices of the behavioral regressors 
    
    RDM_trial_distance= preparing_regressors(data_be, reg1_label)
    RDM_ed_landmark = preparing_regressors(data_be, reg2_label)
    RDM_ed_boundary = preparing_regressors(data_be, reg3_label)
    RDM_type = preparing_regressors(data_be, reg4_label)
    

            

    # Get value of the regressions between RDM_behaoural and RDM_neural
    def regression(X, Y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    X = np.array([RDM_trial_distance, RDM_ed_boundary, RDM_ed_landmark, RDM_type]).squeeze()
    X = X.T
    X = X - X.mean(0)
    X = sm.add_constant(X)  # intercept
    Y = SL_RDM.dissimilarities

    return {"Betas":regression(X, Y.T),"Voxel_index":SL_RDM.rdm_descriptors['voxel_index'],"Participant":j , "Run":f}




from joblib import delayed,Parallel
def runRDMGLM(j):
    runs = []
    
    if j == 1:
        run = [2, 3, 4]
    elif j == 11:
        run = [1, 2, 3]
    else:
        run = [1, 2, 3, 4]
    for f in run:
        runs += [Searchlight_GLM(f, j)]
    return runs

participantRes = Parallel(n_jobs=15)(delayed(runRDMGLM)(j) for j in people)

def getting_betas(df, participant, run): 

    betas_run = pd.DataFrame({"Trial_distance_betas": df[participant][run]["Betas"][1], "ED_boundary_Betas": df[participant][run]["Betas"][2], "ED_landmark_Betas": df[participant][run]["Betas"][3], "Type_Betas": df[participant][run]["Betas"][4], "voxel_index": df[participant][run]["Voxel_index"],
                              "Participant": df[participant][run]["Participant"], "Run": df[participant][run]["Run"]})
    sub = df[participant][run]["Participant"]
    r = df[participant][run]["Run"] 
    betas_run.to_excel(f"/home/data/coefs/GLM2_Ver3.3_sub-{sub}_run{r}.xlsx", index=False)
    
    return betas_run

for j in range(len(people)):
    if j == 0:
        run = [2, 3, 4]
    elif j == 8:
        run = [1, 2, 3]
    else: 
        run = [1, 2, 3, 4]
    for f in range(len(run)): 
        part_betas = getting_betas(participantRes, j, f)
        
        
        
        
        
        
        
        
        
        
        
        
   