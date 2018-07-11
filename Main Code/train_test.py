# =============================================================================
# AntiCP
# BDMH Project 2018
# 
# @Author Himanshu Aggarwal
# @Author Rhythm Nagpal
# @Author Rohit Verma
# 
# This is the main python script which containes the code to train and test various machine learning models
# on the extracted features fromt the data. 
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from hpelm import ELM
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, recall_score, confusion_matrix 

from pandas import DataFrame as df, read_pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV

import numpy as np

# =============================================================================
# Classifiers (SKLEARN)
# =============================================================================

def evaluate(truth, pred):
    accuracy = accuracy_score(truth, pred)
    conf = confusion_matrix(truth, pred)
    TN = conf[0, 0]
    FP = conf[0, 1]
    spec = TN / float(TN + FP)
    mcc = matthews_corrcoef(truth, pred)
    auc = roc_auc_score(truth, pred)
    sensitivity = recall_score(truth, pred)
    
    print accuracy, mcc, auc, sensitivity, spec

def run(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    X_train=X_train.values
    X_test=X_test.values
    y_train=y_train.values
    y_test=y_test.values
    
    #Grid Search
    MLPParams={
            'learning_rate': ["constant", "invscaling", "adaptive"],
            'hidden_layer_sizes': [(100,1), (50,2), (10,3)],
            'activation': ["relu", "tanh"]
            }
    SVMParams={'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
    ABCParams={
            "n_estimators": [50, 75],
            "learning_rate": [0.01, 0.1, 0.2, 0.5]
            }
    RFCParams={
            'n_estimators': [200, 700],
            'max_features': ['auto', 'sqrt', 'log2']
            }
    QDAParams={
            'reg_param': [0.0, 0.1, 0.5]
            }
    clf1 = GridSearchCV(MLPClassifier(), MLPParams, cv=5)
    clf2 = GridSearchCV(SVC(), SVMParams, cv=5)
    clf3 = GridSearchCV(AdaBoostClassifier(), ABCParams, cv=5)
    clf4 = GridSearchCV(RandomForestClassifier(), RFCParams, cv=5)
    clf5 = GridSearchCV(QuadraticDiscriminantAnalysis(), QDAParams, cv=5)
        
    clf1.fit(X_train,y_train)
    pred = clf1.predict(X_test)
    evaluate(y_test, pred)
    
    clf2.fit(X_train,y_train)
    pred = clf2.predict(X_test)
    evaluate(y_test, pred)
    
    clf3.fit(X_train,y_train)
    pred = clf3.predict(X_test)
    evaluate(y_test, pred)
    
    clf4.fit(X_train,y_train)
    pred = clf4.predict(X_test)
    evaluate(y_test, pred)
    
    clf5.fit(X_train,y_train)
    pred = clf5.predict(X_test)
    evaluate(y_test, pred)

    
# =============================================================================
# Load Data Sets and run
# =============================================================================

path = "/Users/himanshuaggarwal/Git Repositories/Data/AntiCP/"

for root, dirs, files in os.walk(path, topdown=False):
    for di in dirs:        
        compositionFeatureX = read_pickle("data/"+di+"/compositionFeatureX.pkl")
        compositionFeatureY = read_pickle("data/"+di+"/compositionFeatureY.pkl")
        diCompositionFeatureX = read_pickle("data/"+di+"/diCompositionFeatureX.pkl")
        diCompositionFeatureY = read_pickle("data/"+di+"/diCompositionFeatureY.pkl")
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureX.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureY.pkl")
        print di
        run(compositionFeatureX, compositionFeatureY)
        run(diCompositionFeatureX, diCompositionFeatureY)
        
        print "C10"
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureXC10.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureYC10.pkl")
        run(biCompositionFeatureX, biCompositionFeatureY) 
#        
        print "N10"
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureXN10.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureYN10.pkl")
        run(biCompositionFeatureX, biCompositionFeatureY) 
        
        print "N10C10"
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureXNC10.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureYNC10.pkl")
        run(biCompositionFeatureX, biCompositionFeatureY) 
        
        print "N5C5"
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureXNC5.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureYNC5.pkl")
        run(biCompositionFeatureX, biCompositionFeatureY) 
        
        print '\n'
