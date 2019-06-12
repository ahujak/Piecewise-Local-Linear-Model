#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:59:58 2019

@author: kartik
"""

import numpy as np
import sklearn
#import tensorflow as tf
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, IsolationForest, VotingClassifier   
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
# from lifelines import CoxPHFitter
from sklearn.cluster import KMeans
import random
import copy
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_california_housing

import lime
import lime.lime_tabular
import warnings
from lime import submodular_pick

class piecewise_model_new:
    def __init__(self,  K, model_type, delta,  black_box, n_components, n_clusters, project, min_clus, fit_type):
        self.K   = K
        self.model_type = model_type
        self.sequence_indices = []
        self.sequence_indices_predns = []
        self.Ys = []
        self.Xs = []
        self.Xs_transform = []
        self.cluster_index  = []
        self.black_box = black_box
        self.model_list = {}
        self.n_components = n_components
        self.n_clusters   = n_clusters
        self.project      = project
        self.cluster_centers = []
        self.number_clusters = {}
        self.min_clus  = min_clus
        self.model_transform = []
        self.fit_type = fit_type
        
    
    def fit(self, X,Y):  
        self.Ys = np.sort(Y)
        self.Xs = X[np.argsort(Y)] 
        N      = self.Xs.shape[0]
        K      = self.K
        self.cluster_index = [0.0]*(N)
        self.cluster_centers = [0.0]*(K)       
        
        if(self.project == 'true'):
          self.Xs_transform = self.data_projection()
        if(self.project == 'false'):
          self.Xs_transform = self.Xs
          
        if(self.fit_type == 'optimal'): 
          self.Ys = np.sort(Y)
          self.Xs = X[np.argsort(Y)]   
#           self.Xs_transform =  X[np.argsort(Y)] 




          self.V   = np.zeros((N,K))
          self.Ind = np.zeros((N,K))


          for n in range(2,N+1):
              for k in range(1,K+1):
                  if (k==1):
                      self.V[(n-1),(0)] = self.G(0,(n)) 
                      self.Ind[(n-1),(0)] =0
                  if (k>1):
                      Value_pos = np.zeros(n-1)
                      current_min  = 1000000

                      for n_p in range(1,n):
                              Value_pos[n_p-1] = self.V[(n_p-1),(k-2)] + self.G(n_p,(n)) 

                      self.V[(n-1),(k-1)]     = np.min(Value_pos)
                      self.Ind[(n-1),(k-1)]   = np.argmin(Value_pos)
  #                     print (Value_pos)
          self.generate_model_list_extension()
        
        if (self.fit_type == 'equal_quantile'):
          self.Ys = np.sort(Y)
          self.Xs = X[np.argsort(Y)]   
          if(self.project == 'true'):
            self.Xs_transform = self.data_projection()     
          if(self.project == 'false'):  
            self.Xs_transform = self.Xs
          self.generate_model_list_extension()           
      

    def G(self, i,j):
        Ys = self.Ys
        Xs = self.Xs
        min_clus = self.min_clus
        Y  = Ys[i:j]
        X  = Xs[i:j]

        n_clusters = self.n_clusters
        if(self.model_type == 'constant'):
            Y_ls = np.sum(np.square(Y-np.mean(Y)))
            return Y_ls
        
        
        if(self.model_type == 'linear'):
            if(n_clusters == 1 or j<i+min_clus):
              reg = linear_model.Lasso(alpha=0.001,fit_intercept=True)   
              reg.fit(X, Y)
              Yp   = reg.predict(X)
              Y_ls = np.sum(np.square(Y-Yp))     
            else:
              self.data_cluster(i,j)
              Y_ls = 0
              for k in range(n_clusters):
                index_cluster = np.where(np.array(self.cluster_index[i:j])==k)[0]

                if(len(index_cluster)>=1):
#                   print (len(index_cluster))
#                   print (index_cluster)
                  reg = linear_model.Lasso(alpha=0.001, fit_intercept=True) 
#                   print(X[index_cluster].shape[0])
                  reg.fit(X[index_cluster], Y[index_cluster])
                  Yp   = reg.predict(X[index_cluster])
                  Y_ls += np.sum(np.square(Y[index_cluster]-Yp))  
            
            return Y_ls
    def data_projection(self):
        Xs = self.Xs
        pca = PCA(n_components=self.n_components)
        data = pca.fit_transform(Xs)
        plt.scatter(data[:,0], data[:,1])
#         plt.title('Scatter plot pythonspot.com')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        self.model_transform = pca

        return data
    
    def data_cluster(self,i,j):
        Xs_transform = self.Xs_transform
        X_t = Xs_transform[i:j]
        n_clusters = self.n_clusters     
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.cluster_index[i:j] = kmeans.fit_predict(X_t)
#         print (self.cluster_index[i:j])
        
    def data_cluster_final(self,i,j,k):
        Xs_transform = self.Xs_transform
        X_t = Xs_transform[i:j]
        n_clusters = self.n_clusters     
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.cluster_index[i:j] = kmeans.fit_predict(X_t)        
        self.cluster_centers[k] = kmeans.cluster_centers_
        
        
    def Ind_sequence(self):
        if(self.fit_type == 'optimal'):
          ind_sequence = []
          N      = self.Xs.shape[0] 
          K_rem  = self.K
          N_rem  = N
          while (N_rem>1):
              index = int(self.Ind[N_rem-1,K_rem-1])
              ind_sequence.append(index)
              N_rem = index - 1
              K_rem = K_rem - 1
          ind_sequence.reverse()
          ind_sequence.append(N-1)
          self.sequence_indices = ind_sequence

        if(self.fit_type == 'equal_quantile'):
          data_size = self.Xs.shape[0]
          K=self.K
          q = np.linspace(0,1,K+1)
#           print (q)
          self.sequence_indices= np.quantile(range(data_size), q=q).astype(int) 
  
  
        
    def generate_model_list_extension(self):
#         print (self.sequence_indices)
        min_clus = self.min_clus
        n_clusters = self.n_clusters
        self.Ind_sequence()
#         print (self.sequence_indices)
        Ys = self.Ys
        Xs = self.Xs
        ind_sequence= self.sequence_indices 
        L = len(ind_sequence)
#         print (L)
        Xb = Xs[ind_sequence]
        self.sequence_indices_predns = self.black_box.predict(Xb)
        for i in range(L-1):
            a = ind_sequence[i]
            b = ind_sequence[i+1]
            Y  = Ys[a:b]
            X  = Xs[a:b]            
            if(self.model_type == 'linear'):
                if(n_clusters == 1 or b<a+min_clus):
                  reg = linear_model.Lasso(alpha=0.01,fit_intercept=True)   
                  reg.fit(X, Y)
                  Yp   = reg.predict(X)
                  Y_ls = np.sum(np.square(Y-Yp))  
                  if(self.project == 'true'):
                    Xt   = self.model_transform.transform(X)   
                  if(self.project == 'false'):
                    Xt  = X
                  self.cluster_centers[i] = Xt.mean(0)  
                  self.model_list[i,0] = reg
                  self.number_clusters[i] = 1 
                else:                
                  self.data_cluster_final(a,b,i)
                  Y_ls = 0
                  self.number_clusters[i] = 0
                  for k in range(n_clusters):
                    index_cluster = np.where(np.array(self.cluster_index[a:b])==k)[0]

                    if(len(index_cluster)>=1):
    #                   print (len(index_cluster))
    #                   print (index_cluster)
                      reg = linear_model.Lasso(alpha=0.01,fit_intercept=True) 
    #                   print(X[index_cluster].shape[0])
                      reg.fit(X[index_cluster], Y[index_cluster])    
                      self.model_list[i,k] = reg  
                      self.number_clusters[i] += 1


        
                
                
    def predict_extension(self, x):

        
        pred_val = self.black_box.predict(x)
        model_list = self.model_list
        K=self.K
        b_predns = self.sequence_indices_predns
        
        len_bpred = len(b_predns)
        
        for j in range(len_bpred):
            if(j==0):
              if(pred_val <= b_predns[j]):
                ind_min = j
            if(j < len_bpred-1):    
                if (pred_val <= b_predns[j+1] and pred_val> b_predns[j]):
                    ind_min = j
            if(j== len_bpred- 1):
                if(pred_val > b_predns[j]):
                    ind_min = j-1
            
      
        cluster_centers = np.array(self.cluster_centers[ind_min])
        if(self.project=='true'):
          x_trans = self.model_transform.transform((x.reshape(-1,1)).T) 
        if(self.project == 'false'):
          x_trans = (x.reshape(-1,1)).T
        
        ind_clus = np.argmin(np.sum(np.square(x_trans-cluster_centers), axis=1))
        L = len(cluster_centers)
        
        
        
        
        
        if(pred_val>b_predns[ind_min]):
            if(ind_min < (len(b_predns)-1)):
                approx_pred = model_list[ind_min,ind_clus].predict(x)
#                 print ("error is:" + str((approx_pred- pred_val)**2))
                return np.maximum(np.minimum(approx_pred,b_predns[ind_min+1]), b_predns[ind_min])
            if(ind_min == len(b_predns)-1):
                approx_pred = model_list[ind_min,ind_clus].predict(x)
#                 print ("error is:" + str((approx_pred- pred_val)**2))
                return np.minimum(approx_pred,b_predns[ind_min])
            
        if(pred_val<=b_predns[ind_min]):
            if(ind_min >0):
                approx_pred = model_list[ind_min-1, ind_clus].predict(x)
#                 print ("error is:" + str((approx_pred- pred_val)**2))
                return np.maximum(np.minimum(approx_pred,b_predns[ind_min]), b_predns[ind_min-1])
            if(ind_min == 0):
                approx_pred = model_list[ind_min, ind_clus].predict(x)
#                 print ("error is:" + str((approx_pred- pred_val)**2))
                return np.maximum(approx_pred, b_predns[ind_min])               
        
     
    def model_interpretations_extension(self):
        Ys = self.Ys
        Xs = self.Xs
        ind_sequence= self.sequence_indices 
        L = len(ind_sequence)      
        ind_vector = {}
        coef_vector = {}
        for i in range(L-1):
          k = self.number_clusters[i]
          for j in range(k):
            if (self.model_type == 'linear'):
              model = self.model_list[i,j]
              coeff = np.abs(model.coef_)
              f_ind = np.argsort(-coeff)
              ind_vector[i,j]= f_ind
              coef_vector[i,j] = coeff[f_ind]
              
        
        return ind_vector, coef_vector
        
        
        
    def model_interpretations(self):
        Ys = self.Ys
        Xs = self.Xs
        ind_sequence= self.sequence_indices 
        L = len(ind_sequence)
#         print (L)
        ind_vector  = {}
#         print (Xs)
        for i in range(L-1):
            a = ind_sequence[i]
            b = ind_sequence[i+1]
            if(self.model_type == 'linear'):
                reg = linear_model.LinearRegression(fit_intercept=True)   
                reg.fit(Xs[a:b], Ys[a:b])
                coeff = np.abs(reg.coef_)
                f_ind = np.argsort(-coeff)
                ind_vector[i] = f_ind 

        return ind_vector    
      
    def compute_cluster_centers(self):
      Xs           = self.Xs
      Xs_transform = self.Xs_transform
      cluster_centers = self.cluster_centers
      n_clusters = self.n_clusters
      
      K          = self.K
      ind_sequence= self.sequence_indices 

      cluster_centers_original = {}
      index_centers_original = {}
      
      for i in range(K):
        a = ind_sequence[i]
        b = ind_sequence[i+1]
        ind_sequence_cont = range(a,b)
        Xs_transform_ab = Xs_transform[a:b]
        Xs_ab           = Xs[a:b]  
        for j in range(n_clusters):
          c = cluster_centers[i][j]
          index_min = np.argmin(np.sum(np.abs(Xs_transform_ab-c)**2, axis=1))
          
          cluster_centers_original[i,j] = Xs_ab[index_min]
          index_centers_original[i,j] = ind_sequence_cont[index_min]
      return cluster_centers_original, index_centers_original
