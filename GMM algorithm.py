# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:47:17 2020

@author: Lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from scipy import stats
import math
import copy as cp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler   
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

##K-means clustering
def clustering(data,K):
    dim = data.shape[1];
    n = data.shape[0];
    means = [np.zeros(dim) for k in range(K)];
    means = np.array(means);
    kmeans=KMeans(n_clusters=K).fit(data);
    label=kmeans.labels_;
    for i in range(n):
        means[label[i]]=means[label[i]]+data[i];
    for j in range(K):
        means[j]=means[j]/sum(label==j);
    return(means)
    
    
##Fit the GMM model based on EM algorithm
def GMM(data,K):
    maxi=1000;
    Q=np.zeros(maxi);
    N = data.shape[0];
    dim = data.shape[1];
    mu= clustering(data,K);
    sigma=[0]*K;
    for i in range(K):
        sigma[i]=np.cov(data.T);
    pi = [1.0/K] * K;
    post_prob = [np.zeros(K) for i in range(N)];
    count=0;
    for j in range(maxi):
        for i in range(N):
            temp = [pi[k] * stats.multivariate_normal.pdf(data[i],mu[k],sigma[k]+0.00001*np.identity(dim)) for k in range(K)];
            temp1 = np.sum(temp);
            for k in range(K):           
                post_prob[i][k] = temp[k]/temp1;
        for k in range(K):
            Nk = np.sum([post_prob[n][k] for n in range(N)]);
            pi[k] = 1.0 * Nk/N;
            mu[k] = (1.0/Nk)*np.sum([post_prob[n][k] * data[n] for n in range(N)],axis=0);
            xdiffs = data - mu[k];
            sigma[k] = (1.0/ Nk)*np.sum([post_prob[n][k]* xdiffs[n].reshape(dim,1) * xdiffs[n] for  n in range(N)],axis=0)
        Q[count] = np.sum(
            [np.log(np.sum([pi[k] * stats.multivariate_normal.pdf(data[n],mu[k],sigma[k]+0.00001*np.identity(dim)) for k in range(K)])) for n in range(N)]);
        print(count)
        if count>50:
            break;
        count=count+1;
    return(pi,mu,sigma,Q)