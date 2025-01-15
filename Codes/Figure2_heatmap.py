#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:24:27 2024

@author: mariachiara
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

###



general_path = "/home/data/" 
dd_path = "/home/data/drag-and-rate"
##########################
people = [1,2, 3, 6, 4, 7, 8, 9, 10,11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
runs = [1, 2, 3, 4]

flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 31, 32]

def normalize_df(df):
    x = (df.iloc[0] - 1) / (5.8 - 1)
    y = (df.iloc[1] - 1) / (33 - 1)

    data = pd.DataFrame({"x": x, "y": y})
    return data
    
    
## square and distorted original shapes

og_borders_square = pd.read_csv("/home/data/STR_square.csv", sep=";").iloc[:, 16:]
og_borders_distorted = pd.read_csv("/home/data/STR_distorted.csv", sep=";").iloc[:, 16:]

og_borders_square.iloc[0, :] = og_borders_square.iloc[0, :]*5
og_borders_distorted.iloc[0, :] = og_borders_distorted.iloc[0, :]*5

og_square = (og_borders_square / 29).T
og_dist = (og_borders_distorted / 34).T

og_square = normalize_df(og_borders_square).values
og_distorted = normalize_df(og_borders_distorted).values

plt.figure(124)
plt.scatter(og_borders_distorted.T.iloc[:, 0], og_borders_distorted.T.iloc[:, 1], color="purple")
plt.scatter(og_borders_square.T.iloc[:, 0], og_borders_square.T.iloc[:, 1])


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

borders_fl.rename(columns={"confidence": "y", "property": "x"}, inplace=True)
borders_og.rename(columns={"confidence": "x", "property": "y"}, inplace=True)


mean_borders_og = borders_og.groupby(["stimulus"]).mean().reset_index()
mean_borders_fl = borders_fl.groupby(["stimulus"]).mean().reset_index()



everything = pd.concat([borders_fl, borders_og])

both_mean_borders = pd.concat([mean_borders_fl, mean_borders_og])
mean_borders = both_mean_borders.groupby(["stimulus"])[["y", "x"]].mean().reset_index()[["y", "x"]].values
std_borders = everything.groupby(["stimulus"])[["y", "x"]].std().reset_index().values

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

P1  = [0.229637, 0.119028]
V1 = np.mean(std_borders[0, 1:])
P2 = [0.931096, 0.470297]
V2 = np.mean(std_borders[1, 1:])
P3 = [0.079171, 0.695986]
V3 = np.mean(std_borders[2, 1:])
P4 = [0.777474, 0.964301]
V4 = np.mean(std_borders[3, 1:])


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def gradiente(X, Y, P1, P2, P3, P4, V1, V2, V3, V4):
    Z = np.zeros_like(X)

    for i, x in enumerate(X[0, :]):
        for j, y in enumerate(Y[:, 0]):
            d1 = np.linalg.norm(np.array([x, y]) - P1) / V1
            d2 = np.linalg.norm(np.array([x, y]) - P2) / V2
            d3 = np.linalg.norm(np.array([x, y]) - P3) / V3
            d4 = np.linalg.norm(np.array([x, y]) - P4) / V4
            Z[j, i] = 0.5 * (gaussian(d1, 0, 1) + gaussian(d2, 0, 1) + gaussian(d3, 0, 1) + gaussian(d4, 0, 1))

    return Z



import math

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

origin = np.array([0, 0])
rotated_points = []
for point in np.array(og_square):
    rotated = rotate(origin, point, math.radians(90))
    rotated_points.append(rotated)
rotated_points = np.array(rotated_points)

rotated_points_dist = []
for point in np.array(og_dist):
    rotated = rotate(origin, point, math.radians(162))
    rotated_points_dist.append(rotated)
rotated_points_dist = np.array(rotated_points_dist)



def normalize_points(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    range_vals = max_vals - min_vals
    normalized_points = (points - min_vals) / range_vals
    return normalized_points

# Normalize the rotated points
norm_rot_square = normalize_points(rotated_points)
norm_rot_dist = normalize_points(rotated_points_dist)

norm_rot_square = norm_rot_square + 0.02
norm_rot_dist = norm_rot_dist + 0.03

from scipy.spatial import ConvexHull

hull_square = ConvexHull(norm_rot_square)
hull_dist = ConvexHull(norm_rot_dist)



X, Y = np.meshgrid(np.linspace(0, 1.08, 250), np.linspace(0, 1.08, 250))
Z = gradiente(X, Y, P1, P2, P3, P4, V1, V2, V3, V4)
#ax.contour(X, Y, Z, levels=np.linspace(0, 1, 20), cmap=gradiente_cmap)
#PLOT ABSTRACT
fig, ax = plt.subplots()
gradiente_cmap = plt.get_cmap('coolwarm')

ax.pcolormesh(X, Y, Z, cmap=gradiente_cmap, alpha=0.5, vmin=Z.min(), vmax=Z.max())
plt.scatter(norm_rot_square[:, 0], norm_rot_square[:, 1], color='black')
plt.scatter(norm_rot_dist[:, 0], norm_rot_dist[:, 1],  color='purple')
for simplex in hull_square.simplices:
    plt.plot(norm_rot_square[simplex, 0], norm_rot_square[simplex, 1], 'k--', alpha = 0.2)
for simplex in hull_dist.simplices: 
    plt.plot(norm_rot_dist[simplex, 0], norm_rot_dist[simplex, 1], '--', alpha = 0.2, color="purple")

#plt.plot(og_square[:, 0], og_square[:, 1], linestyle=':',alpha=0.6, col`or='black')
#plt.scatter(og_distorted[:, 0], og_distorted[:, 1],  color='purple')
#plt.plot(og_distorted[:,0], og_distorted[:,1],linestyle=':',alpha=0.6, color='purple')
plt.xticks(size=10)
plt.yticks(size=10)
# Add a colorbar to the figure

sm = plt.cm.ScalarMappable(cmap=gradiente_cmap)
sm.set_array([])
cbar = fig.colorbar(sm, orientation='vertical')
plt.xlabel('X axis', size=15)
plt.ylabel('Y axis',size=15)



# Define meshgrid
X, Y = np.meshgrid(np.linspace(0, 1.08, 250), np.linspace(0, 1.08, 250))
Z = gradiente(X, Y, P1, P2, P3, P4, V1, V2, V3, V4)

# Plot the meshgrid
fig, ax = plt.subplots()
gradiente_cmap = plt.get_cmap('coolwarm')
mesh = ax.pcolormesh(X, Y, Z, cmap=gradiente_cmap, alpha=0.5, vmin=Z.min(), vmax=Z.max())

plt.scatter(norm_rot_square[:, 0], norm_rot_square[:, 1], color='black')
plt.scatter(norm_rot_dist[:, 0], norm_rot_dist[:, 1], color='purple')
for simplex in hull_square.simplices:
    plt.plot(norm_rot_square[simplex, 0], norm_rot_square[simplex, 1], 'k--', alpha=0.2)
for simplex in hull_dist.simplices: 
    plt.plot(norm_rot_dist[simplex, 0], norm_rot_dist[simplex, 1], '--', alpha=0.2, color="purple")

plt.xticks(size=10)
plt.yticks(size=10)
plt.xlabel('X axis', size=15)
plt.ylabel('Y axis', size=15)

plt.show()
