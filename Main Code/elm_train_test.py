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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from hpelm import ELM, mss_cv
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, recall_score, confusion_matrix , roc_curve,

from pandas import DataFrame as df, read_pickle
import pandas as pd

import numpy as np
# =============================================================================
# ELM Classifier
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
    #ELM model
    X_train=X_train.values
    X_test=X_test.values
    y_train=y_train.values
    y_test=y_test.values
    print 'ELM tanh'
    for x in range(50, 500, 50):
        elm = ELM(X_train.shape[1], 1, classification='c')
        elm.add_neurons(x, 'tanh')
        elm.train(X_train, y_train)
        pred = elm.predict(X_test)
        temp = []
#        print 'Error(TANH, ', x, '): ', elm.error(y_test, pred)
        for p in pred:
            if p >=0.5:
                temp.append(1)
            else:
                temp.append(0)
        pred = np.asarray(temp)
#        print 'Error(TANH, ', x, '): ', elm.error(y_test, pred)
        evaluate(y_test, pred)
    
    print 'ELM rbf_linf tanh'
    for x in range(10, 100, 10):
        elm = ELM(X_train.shape[1], 1)
        elm.add_neurons(x, 'rbf_linf')
        elm.add_neurons(x*2, 'tanh')
        elm.train(X_train, y_train)
        pred = elm.predict(X_test)
        temp = []
#        print 'Error(TANH, ', x, '): ', elm.error(y_test, pred)
        for p in pred:
            if p >=0.5:
                temp.append(1)
            else:
                temp.append(0)
        pred = np.asarray(temp)
#        print 'Error(RBF+TANH, ', x, ',', 2*x, '): ', elm.error(y_test, pred)
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
        print di
        print "C10"
        biCompositionFeatureX = read_pickle("data/"+di+"/biCompositionFeatureXC10.pkl")
        biCompositionFeatureY = read_pickle("data/"+di+"/biCompositionFeatureYC10.pkl")
        run(biCompositionFeatureX, biCompositionFeatureY) 
        
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
        
        
        run(compositionFeatureX, compositionFeatureY)
        run(diCompositionFeatureX, diCompositionFeatureY)
        
        
         Concatinating compositional features to produce new feature vectors
        print "\n\n\n\nNew combined(single+diComposition)\n\n\n"
        allx = pd.concat([compositionFeatureX, diCompositionFeatureX], axis=1)
        ally = diCompositionFeatureY
        run(biCompositionFeatureX, biCompositionFeatureY) 
        
        print '\n'
