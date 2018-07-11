# =============================================================================
# AntiCP
# BDMH Project 2018
# 
# @Author Himanshu Aggarwal
# @Author Rhythm Nagpal
# @Author Rohit Verma
# 
# This is the python script to extract features from the data. The features extracted are aminoacid composition, 
# dipeptide composition and binary profiles of the peptides. 
# =============================================================================

import numpy as np
import os
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
from pandas import DataFrame as df

path = "data/"

# =============================================================================
# Prerequsites for the dataprocessing
# =============================================================================
# dict of peptides
d = dict({'A':0, 'R':0, 'N':0, 'D':0, 'C':0, 'Q':0, 'E':0, 'G':0, 
          'H':0, 'I':0, 'L':0, 'K':0, 'M':0, 'F':0, 'P':0, 'S':0, 'T':0, 'W':0, 
          'Y':0, 'V':0})  

# Non eligible amino acids
na = ['B','J','O','X','Z','U']

# dict of di peptides
dp = {}
for k in d.keys():
    for k2 in d.keys():
        dp[k+k2] = 0
        
# =============================================================================
# Single Composition Festures
# =============================================================================
        
def getComposition(sequences):
    # single peptide count
    composition = []
    for s in sequences:
        if (any(i in s for i in na)):
            continue;
        dtemp = {}
        dtemp.update(d)
        dtemp.update(Counter(s))
        l = len(s)
        dtemp = {k: v/float(l) for k,v in dtemp.iteritems()}
        composition.append(dtemp.values())
        
    return composition
    
def getCompositionFeatures(sequencesPos, sequenceNeg):
    compositionPos = getComposition(sequencesPos)
    compositionNeg = getComposition(sequencesNeg)
    
    X = compositionNeg + compositionPos
    Y = [1]*len(compositionNeg) + [0]*len(compositionPos)
    
    xx = pd.DataFrame(X)
    xx.columns = d.keys()
    y = pd.DataFrame(Y)
    y.columns = ['Classes']
    
    return xx, y
    
# =============================================================================
# Dipeptide Features
# =============================================================================

def getDipeptideComposition(sequences):
    dicomposition = []
    for s in sequences:
        if (any(i in s for i in na)):
            continue;
        dtemp = {}
        dtemp.update(dp)
        ds = []
        for i in xrange(0,len(s)-1):
            ds.append(s[i:i+2])
        dtemp.update(Counter(ds))
        l = len(s)-1
        dtemp = {k: v/float(l) for k,v in dtemp.iteritems()}
        dicomposition.append(dtemp.values())
        
    return dicomposition
    
def getDipeptideFeatures(sequencesPos, sequenceNeg):
    diCompositionPos = getDipeptideComposition(sequencesPos)
    diCompositionNeg = getDipeptideComposition(sequencesNeg)
    
    X = diCompositionNeg + diCompositionPos
    Y = [1]*len(diCompositionNeg) + [0]*len(diCompositionPos)
    
    xx = pd.DataFrame(X)
    xx.columns = dp.keys()
    y = pd.DataFrame(Y)
    y.columns = ['Classes']
    
    return xx, y

# =============================================================================
# Binary Profile (left as N and right as C terminus)
# =============================================================================
fet = df(d.keys())
fetEnc = pd.get_dummies(fet, prefix="amino")

def getBinary(sequences, n):    
    bicomposition = []
    for s in sequences:
        if (any(i in s for i in na)):
            continue;
        sl = list(s)
        if(len(sl)<n): continue;
        sfet = fetEnc["amino_"+sl[0]]
        for i in range(1, n):
            sfet = sfet.append(fetEnc["amino_"+sl[i]], ignore_index=True)
        bicomposition.append(sfet)
    return bicomposition

def getBinaryProfileFeatures(sequencesPos, sequencesNeg, n, terminal):
    
    if terminal == 'N':
        biCompositionPos = getBinary(sequencesPos, n)
        biCompositionNeg = getBinary(sequencesNeg, n)
    elif terminal == 'C':
        sequencesPos = np.asarray([s[::-1] for s in sequencesPos])
        sequencesNeg = np.asarray([s[::-1] for s in sequencesNeg])
        biCompositionPos = getBinary(sequencesPos, n)
        biCompositionNeg = getBinary(sequencesNeg, n)
    elif terminal == 'NC':       
        tempPos=[]
        for s in sequencesPos:
            temp = s[:n]+s[::-1][:n]
            tempPos.append(temp)
            
        
        tempNeg=[]
        for s in sequencesNeg:
            temp = s[:n]+s[::-1][:n]
            tempNeg.append(temp)
        
        sequencesPos = np.asarray(tempPos)
        sequencesNeg = np.asarray(tempNeg)
        biCompositionPos = getBinary(sequencesPos, n)
        biCompositionNeg = getBinary(sequencesNeg, n)
    else:
        biCompositionPos = getBinary(sequencesPos, n)
        biCompositionNeg = getBinary(sequencesNeg, n)
    
    
    X = biCompositionNeg + biCompositionPos
    Y = [1]*len(biCompositionNeg) + [0]*len(biCompositionPos)
    
    xx = pd.DataFrame(X)
    y = pd.DataFrame(Y)
    y.columns = ['Classes']
    
    return xx, y

# =============================================================================
# Normalized data
# =============================================================================

# already normalised

#min_max_scaler = preprocessing.MinMaxScaler()
#xx_scaled = min_max_scaler.fit_transform(xx)
#xx_normalized = pd.DataFrame(xx_scaled)

# =============================================================================
# Visualizing data using TSNE
# =============================================================================

def visualizeTSNE(X,Y):
    X_tsne = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000).fit_transform(X)
    colorset = ['orange', 'blue']
    
    color=[]
    for row in Y:
        color.append(colorset[row])
        
    #plot_name = args.plots_save_dir + dataset_name[-1] + '.png'
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = color)
    #plt.savefig(plot_name)
    plt.show()

# =============================================================================
# Visualizing data using pandas scatter_matrix
# =============================================================================

def visualizeScatter(xx):
    scatter_matrix(xx, alpha=0.2, figsize=(20,10), diagonal='kde')

# =============================================================================
# Visualizing data using PCA
# =============================================================================

def visualizeScatterPCA(xx, n = 2):
    pca = PCA(n_components = n)
    trans = pd.DataFrame(pca.fit_transform(xx))
    if n == 2:
        plt.scatter(trans[:][0], trans[:][1])
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter( trans[:][0], trans[:][1], trans[:][2])

# =============================================================================
# Visualizing data using Parallel coordinates
# =============================================================================
# NORMALISE DATA AND TRY THIS
def visualizeParallelCoordinates(xx,y):
#    Classes = ['G','K','C','F','I','W'] #in abundance for ACP
    Classes = d.keys()
    data_norm = pd.concat([xx[Classes], y], axis=1)
    parallel_coordinates(data_norm, 'Classes')   #not working properly

# =============================================================================
# Extracting features
# =============================================================================

for root, dirs, files in os.walk(path, topdown=False):
    for di in dirs:
        sequencesPos = np.array(open(os.path.join(path, di, "pos.txt")).read().split())
        sequencesNeg = np.array(open(os.path.join(path, di, "neg.txt")).read().split())
        
        compositionFeatureX, compositionFeatureY = getCompositionFeatures(sequencesPos, sequencesNeg)
        diCompositionFeatureX, diCompositionFeatureY = getDipeptideFeatures(sequencesPos, sequencesNeg)
        
        compositionFeatureX.to_pickle("data/"+di+"/compositionFeatureX.pkl")
        compositionFeatureY.to_pickle("data/"+di+"/compositionFeatureY.pkl")
        diCompositionFeatureX.to_pickle("data/"+di+"/diCompositionFeatureX.pkl")
        diCompositionFeatureY.to_pickle("data/"+di+"/diCompositionFeatureY.pkl")
        
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 10, 'C')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXC10.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYC10.pkl")
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 10, 'N')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXN10.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYN10.pkl")
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 5, 'C')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXC5.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYC5.pkl")
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 5, 'N')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXN5.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYN5.pkl")
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 10, 'NC')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXNC10.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYNC10.pkl")
        biCompositionFeatureX, biCompositionFeatureY = getBinaryProfileFeatures(sequencesPos, sequencesNeg, 5, 'NC')
        biCompositionFeatureX.to_pickle("data/"+di+"/biCompositionFeatureXNC5.pkl")
        biCompositionFeatureY.to_pickle("data/"+di+"/biCompositionFeatureYNC5.pkl")