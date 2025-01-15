#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:42:10 2024

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

import plotly.graph_objects as go
import plotly.io as pio

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import classification_report
######################### 
people = [2, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33]
runs = [1, 2, 3, 4]
########################

p0 = [("StandardScaler", StandardScaler()),
("SVC", SVC( kernel="linear", class_weight="balanced"))]    

clf = Pipeline(p0)    
# loading and preparing data 
neural_data = pd.read_excel("/home/All_parts_data_HC_mask.xlsx")
vector_length = 200

conf_matrix_list_of_arrays = []


def zeroing(row, mean, std): 
    return (row["Data"] - mean)/std
    

all_accuracies = pd.DataFrame()
for perm in range(1000):
    for j in people: 
        X_test = []
        y_test = []
        
        whole_test_df = neural_data.loc[neural_data["Participant"] == j].copy()
        mean = np.mean(whole_test_df["Data"])
        std = np.std(whole_test_df["Data"])
        
        whole_test_df["Neural"] = whole_test_df.apply(lambda row: zeroing(row, mean, std), axis=1)
    
        for f in runs: 
            test_run = whole_test_df.loc[whole_test_df["Run"] == f].copy()
            runX_test = test_run["Neural"]
            runy_test = list(test_run["Shape"])[0]
            
            X_test.append(runX_test)
            y_test.append(runy_test)
        
        new_x =  SelectKBest(f_classif, k=vector_length).fit_transform(X_test, y_test)
        new_x = [ (array - np.mean(array)) / np.std(array) for array in new_x ]
        
                
        noise_one =np.random.normal(loc=0, scale=1, size=vector_length)
        noise_two =np.random.normal(loc=0, scale=1, size=vector_length)
        
                
        new_x.append(noise_one)
        new_x.append(noise_two)
        
        new_x = np.array(new_x)
        
        
        y_test.append("Noise")
        y_test.append("Noise")
    
        leave_one_out = [p for p in people if p != j]
        
        X2_train = []
        y2_train = []
        
        
        for j2 in leave_one_out:
            
            pX_train = []
            py_train = []
            
            train_df = neural_data.loc[neural_data["Participant"] == j2].copy()
            
            mean = np.mean(train_df["Data"])
            std = np.std(train_df["Data"])
            
            train_df["Neural"]= train_df.apply(lambda row: zeroing(row, mean, std), axis=1)
            
            for f in runs: 
                run_train_df = train_df.loc[train_df["Run"] == f].copy()    
                runX_train =  run_train_df["Neural"]
                runy_train = list(run_train_df["Shape"])[0]
       
                pX_train.append(runX_train)           
                py_train.append(runy_train)
            
            train_x = SelectKBest(f_classif, k=vector_length).fit_transform(pX_train, py_train)
            train_x = [ (array - np.mean(array)) / np.std(array) for array in train_x ]
            noise_one =np.random.normal(loc=0, scale=1, size=vector_length)
            noise_two = np.random.normal(loc=0, scale=1, size=vector_length)
    
            
            train_x.append(noise_one)
            train_x.append(noise_two)
            
            train_x = np.array(train_x)
            
            py_train.append("Noise")
            py_train.append("Noise")
                
            
            X2_train.append(train_x)
            y2_train.append(py_train)  
    
        
        
        X_test = np.array(new_x)
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
    
        clf.fit(X=X_train, y=y2_train)
    
    
        y_pred = clf.predict(X=X_test)
        
    
        accuracy = accuracy_score(y_test, y_pred)
        
        df_acc = pd.DataFrame({"Accuracy": [accuracy], "Participant": j})
        all_accuracies = pd.concat([all_accuracies, df_acc])
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)



print(f"Classifier accuracy : {np.mean(all_accuracies['Accuracy'])}")

mean_confusion_matrix = np.sum(conf_matrix_list_of_arrays, axis=0)

disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix,display_labels=clf.classes_)

plt.figure(241)
cm = disp.plot()
plt.grid(False)
plt.ylabel("True label", size=14)
plt.xlabel("Predicted label", size=14)
plt.xticks(size=13)
plt.yticks(size=13)
### calculating ppv 

  
noise_ratio = (mean_confusion_matrix[0, 0])/ (mean_confusion_matrix[0, 0] + mean_confusion_matrix[1, 0] + mean_confusion_matrix[2, 0])
distorted_ratio = (mean_confusion_matrix[1, 1])/ (mean_confusion_matrix[0, 1] + mean_confusion_matrix[1, 1] + mean_confusion_matrix[2, 1])
square_ratio = (mean_confusion_matrix[2,2])/ (mean_confusion_matrix[0, 2] + mean_confusion_matrix[1, 2] + mean_confusion_matrix[2, 2])




####################################################### 
####################################################### running the permutation 
####################################################### 
all_ppvs = pd.DataFrame()
for perm_2 in range(1000):
    conf_matrix_list_of_arrays = []

    for perm in range(1000):
        for j in people: 
            X_test = []
            y_test = []
            
            whole_test_df = neural_data.loc[neural_data["Participant"] == j].copy()
            mean = np.mean(whole_test_df["Data"])
            std = np.std(whole_test_df["Data"])
            
            #whole_test_df["Neural"] = whole_test_df.apply(lambda row: zeroing(row, mean, std), axis=1)
        
            for f in runs: 
                test_run = whole_test_df.loc[whole_test_df["Run"] == f].copy()
                runX_test = test_run["Data"]
                runy_test = list(test_run["Shape"])[0]
                
                X_test.append(runX_test)
                y_test.append(runy_test)
            
            new_x =  SelectKBest(f_classif, k=vector_length).fit_transform(X_test, y_test)
            new_x = [(array - np.mean(array)) / np.std(array) for array in new_x]
    
                
            noise_one =np.random.normal(loc=0, scale=1, size=vector_length)
            noise_two =np.random.normal(loc=0, scale=1, size=vector_length)
            noise_one = np.array(noise_one)
            noise_two =np.array(noise_two)
            
                    
            new_x.append(noise_one)
            new_x.append(noise_two)
            
            new_x = np.array(new_x)
            
            
            y_test.append("Noise")
            y_test.append("Noise")
        
            leave_one_out = [p for p in people if p != j]
            
            X2_train = []
            y2_train = []
            
            
            for j2 in leave_one_out:
                
                pX_train = []
                py_train = []
                
                train_df = neural_data.loc[neural_data["Participant"] == j2].copy()
                
                mean = np.mean(train_df["Data"])
                std = np.std(train_df["Data"])
                
                #train_df["Neural"]= train_df.apply(lambda row: zeroing(row, mean, std), axis=1)
                
                for f in runs: 
                    run_train_df = train_df.loc[train_df["Run"] == f].copy()    
                    runX_train =  run_train_df["Data"]
                    runy_train = list(run_train_df["Shape"])[0]
           
                    pX_train.append(runX_train)           
                    py_train.append(runy_train)
                
                train_x = SelectKBest(f_classif, k=vector_length).fit_transform(pX_train, py_train)
                train_x = [ (array - np.mean(array)) / np.std(array) for array in train_x ]
    
                            
                noise_one =np.random.normal(loc=0, scale=1, size=vector_length)
                noise_two =np.random.normal(loc=0, scale=1, size=vector_length)
                        
                noise_one = np.array(noise_one)
                noise_two =np.array(noise_two)
                
                train_x.append(noise_one)
                train_x.append(noise_two)
                
                train_x = np.array(train_x)
                
                py_train.append("Noise")
                py_train.append("Noise")
                    
                
                X2_train.append(train_x)
                y2_train.append(py_train)  
        
            
            
            X_test = np.array(new_x)
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
        
            clf.fit(X=X_train, y=y2_train)
        
        
            y_pred = clf.predict(X=X_test)
            
        
            accuracy = accuracy_score(y_test, y_pred)
            
            df_acc = pd.DataFrame({"Accuracy": [accuracy], "Participant": j})
            all_accuracies = pd.concat([all_accuracies, df_acc])
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            conf_matrix_list_of_arrays.append(conf_matrix)
        
        mean_confusion_matrix = np.sum(conf_matrix_list_of_arrays, axis=0)
        
        noise_ratio = (mean_confusion_matrix[0, 0])/ (mean_confusion_matrix[0, 0] + mean_confusion_matrix[1, 0] + mean_confusion_matrix[2, 0])
        distorted_ratio = (mean_confusion_matrix[1, 1])/ (mean_confusion_matrix[0, 1] + mean_confusion_matrix[1, 1] + mean_confusion_matrix[2, 1])
        square_ratio = (mean_confusion_matrix[2,2])/ (mean_confusion_matrix[0, 2] + mean_confusion_matrix[1, 2] + mean_confusion_matrix[2, 2])
    
        ppv = pd.DataFrame({"Noise": [noise_ratio], "Square": square_ratio, "Distorted": distorted_ratio})
        all_ppvs = pd.concat([all_ppvs, ppv])