# Piecewise Local Linear Model
In this project, we provide a method to construct piecewise local linear approximations of a black box model. The main idea behind building piecewise approximations is to divide the feature space into homogeneous regions and explain the model behavior locally in terms of a linear model. This work is based on https://arxiv.org/pdf/1806.10270.pdf
## piecewise_model_new class

### Parameters:
K: integer, number of intervals to partition the f's range 
model_type = "linear" for a piecewise linear approximation and model_type = "constant" for a piecewise constant approximation
delta = 1 (Default)

black_box = black box model (here model's from sklearn are permitted)

n_components = number of components in PCA, 

n_clusters= number of regions in which we divide the inverse image of each interval

project = 'true' then project the data before clustering, project = 'false' then don't project the data  

min_clus = minimum size of the cluster

fit_type = 'equal_quantile' divide the f's range into equal quantiles, and fit_type = 'optimal' divide the f's range using the Algorithm 1 and 2 in https://arxiv.org/pdf/1806.10270.pdf.

### Methods
fit: fit(self, X, Y) Build a piecewise model from the training set (X, Y), X is an array of features size [n_samples, n_features] and Y  is the target data of size [n_samples,] 

predict_extension: predict_extension(self, test), test is an array with one test data point of size [1,n_features]

model_interpretations_extension: model_interpretations_extension(self) returns ind_vector: index of the vector of the features in the decreasing order of the absolute value of their coefficients in the linear model, f_vec: absolute value of the coefficients of the corresponding features in the decreasing order

Ind_sequence: Ind_sequence(self) function to compute the indices in the sorted data (Ys defined below) at which the data is partitioned along function f's range 



### Attributes
Ys: sorted values of Y

Xs: features sorted in the same order as Y is sorted

Xs_transform: transformed data Xs (if project = 'true' we use PCA to transform the data, else Xs_transform= Xs)
ind_sequence

sequence_indices: sequence_indices is the indices of the sorted data at which the data is partitioned along function f's range

cluster_centers: cluster_centers is the array of cluster centers of shape [K, n_clus]

cluster_index: cluster_index is the array of cluster indices of the different data points in Xs and its shape is [n_samples,] 




