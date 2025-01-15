#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:02:08 2024

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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import random 
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.metrics import RocCurveDisplay,  roc_curve
from scipy.stats import spearmanr
from scipy import stats


######################### 
people = [2, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
runs = [1, 2, 3, 4]
########################

# defining clf pipeline
p0 = [("StandardScaler", StandardScaler()),
      ("PCA", PCA(n_components=0.9)),
("SVC", SVC( kernel="linear", class_weight="balanced"))]    

clf = Pipeline(p0)    


# loading and preparing data 
neural_data = pd.read_excel("/home/path/data.xlsx")



def zeroing(row, mean, std): 
    return (row["Data"] - mean)/std


y_trues = []
y_preds = []

conf_matrix_list_of_arrays = []

all_accuracies = pd.DataFrame()

for j in people: 
    X_test = []
    y_test = []
    
    whole_test_df = neural_data.loc[neural_data["Participant"] == j].copy()

    for f in runs: 
        test_run = whole_test_df.loc[whole_test_df["Run"] == f]
        runX_test = list(test_run["Data"])
        runy_test = list(test_run["Shape"])[0]
        
        X_test.append(runX_test)
        y_test.append(runy_test)
    
    leave_one_out = [p for p in people if p != j]
    
    X2_train = []
    y2_train = []
    
    
    for j2 in leave_one_out:
        
        pX_train = []
        py_train = []
        
        train_df = neural_data.loc[neural_data["Participant"] == j2].copy()
                    

        
        for f in runs: 
            run_train_df = train_df.loc[train_df["Run"] == f]    
            runX_train = list(run_train_df["Data"])
            runy_train = list(run_train_df["Shape"])[0]
   
            pX_train.append(runX_train)           
            py_train.append(runy_train)
        
        X2_train.append(pX_train)
        y2_train.append(py_train)
        
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X2_train = np.array(X2_train)
    y2_train = np.array(y2_train)

    # reshaping train 

    
    new_shape2 = (X2_train.shape[0] * X2_train.shape[1], X2_train.shape[2])
    X_train = X2_train.reshape(new_shape2)
    
    y2_train= y2_train.flatten()
    y_test = y_test.flatten() 
    
    
    ## 
    zipped_list = list(zip(X_train, y2_train))
    random.shuffle(zipped_list)
    
    X_train_shuffled = [arr for arr, label in zipped_list]
    y2_train_shuffled = [label for arr, label in zipped_list]
    
    clf.fit(X=X_train, y=y2_train)


    y_pred = clf.predict(X=X_test)
    
    y_trues.append(y_test)
    y_preds.append(y_pred)
    
    print(f"... Done classifying {j}... ")
    accuracy = accuracy_score(y_test, y_pred)
    
    df_acc = pd.DataFrame({"Accuracy": [accuracy], "Participant": j})
    all_accuracies = pd.concat([all_accuracies, df_acc])
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_list_of_arrays.append(conf_matrix)

print(f"Classifier accuracy : {np.mean(all_accuracies['Accuracy'])}")

mean_confusion_matrix = np.sum(conf_matrix_list_of_arrays, axis=0)

ppv_square = mean_confusion_matrix[0][0] / (mean_confusion_matrix[0][0] + mean_confusion_matrix[1][0]) 
ppv_distorted = mean_confusion_matrix[1][1] / (mean_confusion_matrix[0][1] + mean_confusion_matrix[1][1]) 


########################################################################################
# PERMUTATIONS
permuted_classifiers = pd.DataFrame()

for permutation in range(1000):
    permuted_accuracies = pd.DataFrame()
    
    for j in people: 
    
        X_test = []
        y_test = []
        
        whole_test_df = neural_data.loc[neural_data["Participant"] == j]
        for f in runs: 
            test_run = whole_test_df.loc[whole_test_df["Run"] == f]
            runX_test = list(test_run["Data"])
            runy_test = list(test_run["Shape"])[0]
            
            X_test.append(runX_test)
            y_test.append(runy_test)
        
        leave_one_out = [p for p in people if p != j]
        
        X2_train = []
        y2_train = []
        
        
        for j2 in leave_one_out:
            
            pX_train = []
            py_train = []
            
            train_df = neural_data.loc[neural_data["Participant"] == j2]
            for f in runs: 
                run_train_df = train_df.loc[train_df["Run"] == f]    
                runX_train = list(run_train_df["Data"])
                runy_train = list(run_train_df["Shape"])[0]
       
                pX_train.append(runX_train)           
                py_train.append(runy_train)
            
            X2_train.append(pX_train)
            y2_train.append(py_train)
            
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        X2_train = np.array(X2_train)
        y2_train = np.array(y2_train)
    
        # reshaping train 
    
        
        new_shape2 = (X2_train.shape[0] * X2_train.shape[1], X2_train.shape[2])
        X_train = X2_train.reshape(new_shape2)
        
        y2_train= y2_train.flatten()
        y_test = y_test.flatten() 
        
        
        ######### permutation / randomization
        
        permuted_samples = np.random.permutation(X_train)
        permuted_labels = np.random.permutation(y2_train)

        
        clf.fit(X=permuted_samples, y=y2_train)
    
        y_pred = clf.predict(X=X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        df_perm_acc = pd.DataFrame({"Accuracy": [accuracy], "Participant": j})
        permuted_accuracies = pd.concat([permuted_accuracies, df_perm_acc])
    
    permutations_dataframe = pd.DataFrame({"Mean_accuracy": [np.mean(permuted_accuracies["Accuracy"])], "Permutation": permutation})
    n_permutation = permutation+ 1
    print(f"... Done with permutation n. {n_permutation} ...")
    permuted_classifiers = pd.concat([permuted_classifiers, permutations_dataframe])


observed_acc = np.mean(all_accuracies["Accuracy"])

# one sided...
greater_values = len(permuted_classifiers.loc[permuted_classifiers["Mean_accuracy"] >= observed_acc])
p_value = greater_values / 1000 

print(f"p-value with 1000 permutations (permuting samples on train dataset) BEFORE Leave-One-Participant-Out cross-validation: {p_value}")

