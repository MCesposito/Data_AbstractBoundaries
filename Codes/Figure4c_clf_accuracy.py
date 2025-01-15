#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:28:31 2025

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

import statsmodels.api as sm 
import statsmodels.formula.api as smf

from scipy import stats
from sklearn.linear_model import LinearRegression



general_path = "/home/data/" 

##########################
people = [2, 3, 6, 4, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
runs = [1, 2, 3, 4]
##########################

hc_accuracy = pd.read_excel("/home/data/accuracies_HC.xlsx")

all_parts = pd.DataFrame()


for j in people:
    for f in runs: 
            if len(str(j)) == 1:
                participant = pd.read_excel(os.path.join(general_path, f"sub-0{j}", "func", f"3.4_sub-0{j}_run-0{f}_events.xlsx"))
            else:
                participant = pd.read_excel(os.path.join(general_path, f"sub-{j}", "func", f"3.4_sub-{j}_run-0{f}_events.xlsx"))
            
            participant["Participant"] = j 
            
            
         
            all_parts = pd.concat([all_parts, participant])
            
by_shape = all_parts.groupby(["Participant", "Shape"])["Performance"].mean().reset_index()
square_accuracy = by_shape.loc[by_shape["Shape"] == "square"]["Performance"].reset_index()
distorted_accuracy = by_shape.loc[by_shape["Shape"] == "distorted"]["Performance"].reset_index()

all_df = pd.DataFrame({"Participant": people, "Square_accuracy": square_accuracy["Performance"], "Distorted_accuracy": distorted_accuracy["Performance"], "Clf_accuracy": hc_accuracy["Accuracy"]})
all_df.dropna(inplace=True)
whole_performance = all_parts.groupby(["Participant"])["Performance"].mean().reset_index()

conditions= all_parts.groupby(["Participant"])["Condition"].mean().reset_index()
########

plt.figure(3242, figsize=(6,4))
sns.set_style("whitegrid")
ax  = sns.regplot(data=all_df, x = "Clf_accuracy", y = "Square_accuracy", color="#009900", label="Square", scatter=True)
ax2 = sns.regplot(data=all_df, x = "Clf_accuracy", y = "Distorted_accuracy", color="#E37200", label="Distorted", scatter=True)
plt.xlabel("Classifier accuracy", size=13)
plt.ylabel("Task accuracy", size=13)
plt.xticks(np.arange(0.25, 1.2, 0.25), ["25%", "50%", "75%", "100%"], size=13)
plt.yticks(np.arange(0.60, 1.0, 0.05), ["60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%"], size=13)
handles, labels = ax.get_legend_handles_labels()
plt.legend(title='', handles=handles[0:2], labels=['Task acc on square', 'Task acc on distorted'])
plt.tight_layout()


