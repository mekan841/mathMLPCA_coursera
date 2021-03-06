# import packages and data
import numpy as np
import scipy
import scipy.stats

import matplotlib.pyplot as plt
from ipywidgets import interact

plt.style.use('fivethirtyeight')
%matplotlib inline

from load_data import load_mnist

MNIST = load_mnist('./')
images, labels = MNIST['data'], MNIST['target']


# normalisation function
def normalize(X):
    mu = X.mean(0)  
    Xbar = X - mu   
    return Xbar, mu

  
  
# compute sorted eigenvalues and eigenvectors
def eig(S):
    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]
    
    return eigvals[sort_indices], eigvecs[:, sort_indices]
  
  
  
# projection matrix function

def projection_matrix(B):
    P = B @ np.linalg.inv(B.T @ B) @ B.T
    return P

# PCA function

def PCA(X, num_components):
#     # first perform normalization on the digits so that they have zero mean and unit variance
    X_normalized, mean = normalize(X)
  
#     # Then compute the data covariance matrix S
    S = np.cov(X, bias = True, rowvar=False)    

#     # Next find eigenvalues and corresponding eigenvectors for S
    eig_vals, eig_vecs = eig(S)

#     # Take the top `num_components` of eig_vals and eig_vecs,
#     # This will be the corresponding principal values and components
    principal_vals = eig_vals[0:num_components]
    principal_components =  eig_vecs[:,0:num_components]


#     # reconstruct the data from the using the basis spanned by the principal components
    P = projection_matrix(eig_vecs[:,:num_components])
        
    reconst = (P @ X.T).T

    return reconst, mean, principal_vals, principal_components
