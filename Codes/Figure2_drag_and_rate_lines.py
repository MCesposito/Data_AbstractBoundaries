#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:45:53 2025

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


from scipy.spatial import procrustes
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

general_path = "/home/data/" 
dd_path = "/home/data/drag-and-rate/"
##########################
people = [1,2, 3, 6, 4, 7, 8, 9, 10,11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
runs = [1, 2, 3, 4]


flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 31, 32]


def normalize_df(df):
    x = (df.iloc[0] - 1) / (5.8 - 1)
    y = (df.iloc[1] - 1) / (33 - 1)

    data = pd.DataFrame({"x": x, "y": y})

    return data
def normalized_flipped(df):
    x = (df.iloc[1] - 1) / (33 -1)
    y = (df.iloc[0] - 1) / (5.8 - 1)


    data = pd.DataFrame({"x": x, "y": y})
    return data



    
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

og_products_square = pd.read_csv("/home/data/STR_square.csv", sep=";").iloc[:, :16]
og_products_distorted = pd.read_csv("/home/data/STR_distorted.csv", sep=";").iloc[:, :16]


og_square_pr = normalize_df(og_products_square).reset_index()
og_distorted_pr = normalize_df(og_products_distorted).reset_index()

flp_square_pr= normalize_df(og_products_square).reset_index()
flp_distorted_pr = normalize_df(og_products_distorted).reset_index()

all_prs = pd.concat([og_square_pr, og_distorted_pr, flp_square_pr, flp_distorted_pr])

original_pos = all_prs.groupby(["index"]).mean().reset_index()

#mean_all_prods

mean_flp = placement_fl.groupby(["stimulus"]).mean().reset_index() 
mean_base = placement_og.groupby(["stimulus"]).mean().reset_index()

all_placement = pd.concat([placement_fl, placement_og])
mean_placement = all_placement.groupby(["stimulus"]).mean().reset_index()


indices_to_exclude = [4, 5, 13, 20, 21, 24]
mean_placement = mean_placement[~mean_placement.index.isin(indices_to_exclude)]

mean_placement['stimulus'] = mean_placement['stimulus'].str.lower()


product_mapping = {
    '4_pimiento': 'pimiento',
    'pimiento': 'pimiento',
    '2_nispero-removebg-preview': 'nispero',
    '2_nispero-removebg-preview (1)': 'nispero', 
    '5_repollo': 'repollo',
    '5_repollo-removebg-preview' : 'repollo',
    '3_brocoli-removebg-preview': 'brocoli'

}


mean_placement['stimulus'] = mean_placement['stimulus'].map(product_mapping).fillna(mean_placement['stimulus'])

products_df = mean_placement.groupby('stimulus').mean().reset_index()

products = original_pos["index"]
products_df["stimulus"] = products
products_df["stimulus"].iloc[0] = "champiñones"
products_df.sort_values(by="stimulus", inplace=True)

## adding the boundaries 
borders_fl.rename(columns={"confidence": "y", "property": "x"}, inplace=True)
borders_og.rename(columns={"confidence": "x", "property": "y"}, inplace=True)

all_boundaries = pd.concat([borders_fl, borders_og])
mean_boundaries = all_boundaries.groupby(["stimulus"]).mean().reset_index()



products_df = products_df.dropna()
sns.set_style("whitegrid")
from matplotlib.collections import LineCollection
fig, ax = plt.subplots()


lines = [[(products_df["confidence"][i], products_df["property"][i]), (original_pos["x"][i], original_pos["y"][i])] for i in range(len(products_df))]


lc = LineCollection(lines, linewidths=(1), linestyles="dotted", color="grey")
ax.add_collection(lc)
plt.legend()
plt.tight_layout()

## 
figu, ax= plt.subplots(figsize=(6,6))
sns.scatterplot(data=products_df, x = "confidence", y = "property", label="Participants' placement", color="salmon", s = 105)
sns.scatterplot(data=original_pos, x = "x", y  ="y", label = "Original positions", color = "silver", s = 105)
sns.scatterplot(data=mean_boundaries, x = "x", y = "y", color="black", s  =110, alpha = 0.6)
plt.scatter(0.5, 0.5, color="black", alpha=0.8)
plt.xticks(np.arange(0, 1.2, 0.2),  [0, 0.2, 0.4, 0.6, 0.8, 1],  size=15)
plt.yticks(np.arange(0, 1.2, 0.2),  [0, 0.2, 0.4, 0.6, 0.8, 1],  size=15)
#plt.axis("equal")
plt.xlabel("X axis",size=16)
plt.ylabel("Y axis",size=16)
ax.set_aspect('equal', adjustable='box')
lc = LineCollection(lines, linewidths=(1), linestyles="dotted", color="grey")
ax.add_collection(lc)
plt.legend()
plt.tight_layout()
