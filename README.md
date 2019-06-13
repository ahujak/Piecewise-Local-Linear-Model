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
fit: fit(self, X, y) Build a piecewise model from the training set (X, y)
model_interpretations_extension: model_interpretations_extension(self) Generates 
