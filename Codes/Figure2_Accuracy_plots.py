#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:59:27 2024

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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score

from sklearn.manifold import MDS
import rsatoolbox
from scipy import stats
import random
import scikit_posthocs as sp
from sklearn.metrics import pairwise_distances
from scipy.stats import sem

#####################################
people = [2, 3, 4, 6, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
runs = [1, 2, 3, 4]


general_path = "/home/data/" 


## getting behavioral data
all_p = pd.DataFrame()
for j in people: 
    for f in runs:
        if len(str(j)) == 1:
            participant = pd.read_excel(os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx"))
        else:
            participant = pd.read_excel(os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx"))
        
        participant["Participant"] = j
        
        participant.groupby(["Presentazione_novel"])[["ED_boundary", "Trial_absolute", "Performance", "Duration"]].mean().reset_index()
        participant["Presentazione_novel"] = [path.split('_')[1].split('.')[0] for path in participant["Presentazione_novel"]]
        participant.sort_values(by="Presentazione_novel", inplace=True)
        participant = participant.reset_index()
        
        
        all_p = pd.concat([all_p, participant])
        

sns.set_style("whitegrid")


## distance bins
all_p["CD_bins"] = pd.cut(all_p["Trial_absolute"], 3, labels= ["Close", "Medium", "Far"])
all_p["ED_bins"] = pd.cut(all_p["ED_absolute"], 3, labels= ["Hard", "Medium", "Easy"])

# accuracy
plt.figure(2, figsize=(5,3.5))   ## just choice distance
ax = sns.pointplot(data=all_p, x = "CD_bins", y = "Performance", color="silver") 

plt.xlabel("Choice Distance", size=13)
plt.yticks(np.arange(0.70,  0.95, 0.05), ["70%", "75%", "80%", "85%" , "90%"], size=13)#
#plt.yticks(np.arange(3, 3.9, 0.2))
plt.ylabel("Accuracy", size=13)
plt.xticks(size=13)
plt.tight_layout()

print(stats.spearmanr(all_p["CD_bins"], all_p["Performance"]))

##########

participants = all_p.groupby(["Participant", "Shape", "Type"])["Performance"].mean().reset_index()



plt.figure(3333)
silvers = ["#C0C0C0", "#C0C0C0"]
ax1 = sns.stripplot(data=participants, x="Shape", y="Performance", hue="Type", palette=silvers, dodge=True, linewidth=0.4, edgecolor="gray", legend=False, s=6)
ax = sns.barplot(data=all_p, x = "Shape", y ="Performance", hue = "Type", palette= "rocket")
plt.yticks(np.arange(0, 1.2, 0.2), [f"{int(i*100)}%" for i in np.arange(0, 1.2, 0.2)])

plt.ylabel("Accuracy")

handles, labels = ax1.get_legend_handles_labels()
new_labels = ["Context Dependent", "Context Independent"] 
ax1.legend(handles=handles, labels=new_labels, title="Type", loc="lower right")

plt.show()
