#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:09:19 2024

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

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf


general_path = "/home/mariachiara/Escritorio/fMRI/fmriprep+matlab/" 

##

people = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
runs = [1, 2,3,4]


all_participants = pd.DataFrame()
betas_data = pd.DataFrame()

for j in people:
    
    participant_df = pd.DataFrame()
    
    for f in runs: 

        if len(str(j)) == 1:
            participant = pd.read_excel(os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx"))
        else:
            participant = pd.read_excel(os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx"))
        participant["Participant"] = j
        all_participants = pd.concat([all_participants, participant])
        participant_df = pd.concat([participant_df, participant])
      
    formula = 'Performance ~ Trial_absolute + ED_landmark + ED_boundary + Trial_absolute:Run + ED_landmark:Run + ED_boundary:Run'
    model = smf.ols(formula=formula, data=participant_df).fit()
    
    saving_data = pd.DataFrame({"Participant": [j], "Trial_distance_betas": model.params[1], "Trial_run": model.params[4], "ED_boundary_Betas": model.params[3], "ED_landmark_Betas": model.params[2], "Boundary_runs": model.params[6], "Landmark_runs": model.params[5] })
    betas_data = pd.concat([betas_data, saving_data])
    
t_tests = pd.DataFrame()
for column in betas_data.columns[1:]:
    t_stat, p_value = stats.ttest_1samp(betas_data[column], 0)
    t_df = pd.DataFrame({"variable": [column], "T_stat": t_stat, "p_value": p_value}) 
    t_tests= pd.concat([t_tests, t_df])    

t_tests = t_tests.reset_index()

  
hc_betas = pd.read_excel("/home/data/RSA/HC.xlsx")
mpfc_betas= pd.read_excel("/home/data/RSA/mpfc.xlsx")

betas_parts = betas_data.loc[betas_data["Participant"].isin(hc_betas["Participant"])]
      
## Inter subject RSA  for behavioral and ED boundary betas
def rdm(data, variable):
    data_rdm = abs(data[variable].values - data[variable].values[:, np.newaxis])
    return np.concatenate([data_rdm[i, i+1:] for i in range(0, data_rdm.shape[0])])    


correlations = []
for column in hc_betas.columns[1:4]: 
    behavior = rdm(betas_parts, column)
    hc = rdm(hc_betas, column)
    mpfc = rdm(mpfc_betas, column)
    
    hc_correlation = stats.spearmanr(behavior, hc)
    mpfc_correlation = stats.spearmanr(behavior, mpfc)
    
    data = {"hc": hc_correlation, "mpfc": mpfc_correlation, "variable": column}
    correlations.append(data)
    

print(correlations)