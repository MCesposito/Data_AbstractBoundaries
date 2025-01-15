#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:45:35 2024

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

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

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



all_participants["Pres"] = all_participants.apply(lambda row: "First_presentation" if (row["Run"] == 1) or (row["Run"] == 2) else "Second_presentation", axis=1)

he = all_participants.groupby(["Participant", "Shape", "Pres"])["Performance"].mean().reset_index()
anova_stats = AnovaRM(he, "Performance", "Participant", within=["Shape", "Pres"]).fit()

performances = all_participants.groupby(["Participant", "Pres", "Shape"])["Performance"].mean().reset_index()

sns.set_style("whitegrid")
plt.figure(12121)
silvers = ["#C0C0C0", "#C0C0C0"]
colors = ["#D5E8D4","#FFE6CC"]
ax1 = sns.stripplot(data=performances, x="Pres", y="Performance", hue="Shape", palette=silvers, dodge=True, linewidth=0.4, edgecolor="gray", legend=False, s=6)
ax = sns.barplot(data=all_participants, x = "Pres", y ="Performance", hue = "Shape", palette= colors)
#ax.get_legend().remove()
plt.legend(loc='lower right') 
plt.xlabel("")
plt.xticks(range(0,2), ["First presentation", "Second presentation"], size=14)
plt.ylabel("Task Accuracy", size=14)
plt.yticks(np.arange(0, 1.2, 0.2), [f"{int(i*100)}%" for i in np.arange(0, 1.2, 0.2)])



#####