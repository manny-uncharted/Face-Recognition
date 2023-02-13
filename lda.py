import numpy as np
from numpy import linalg as LA
from scipy.sparse import linalg as SLA
import os

class LDA:
    def __init__(self):
        self._proj_mat = None

    def train(self, X, y, noOfClasses, r, load_path=None, verbose=False):

        """
            LDA function used for dimensionality reduction
            X_train is the data matrix
            y_train is the label column vector
            r is the dimensions of the subspace
        """

        if load_path is not None and os.path.isfile(load_path):
            if verbose:
                print(f'loading from {load_path}')
            self._proj_mat = np.real(np.load(load_path))
            return self._proj_mat

        dims = X.shape[1]
        #calculating the mean of the classes
        mean_vectors = np.empty((noOfClasses, dims))
    
        for cl in range(0, noOfClasses):
            mean_vectors[cl] = np.mean(X[y==cl+1], axis=0, keepdims=True)
                
        #within-class matrix
        S = np.zeros((dims, dims))                                         #initializing the matrix
        for cl, mean in zip(range(0, noOfClasses), mean_vectors):
            #class_sc_mat = np.zeros((dims, dims))                         # scatter matrix for every class

            mat = X[y == cl+1]
            z = mat - mean
            if verbose:
                print(f'calculating S {cl+1}')
            S += z.T.dot(z)                                                # sum class scatter matrices
        
        #between-class matrix
        overall_mean = np.mean(X, axis=0, keepdims=True)
        centered_means = mean_vectors - overall_mean
        B = np.zeros((dims, dims))
        for i, _ in enumerate(mean_vectors):  
            ni = X[y==i+1,:].shape[0] # cardinality of ith class
            if verbose:
                print(f'calculating B {i+1}')
            B += ni * (centered_means).T.dot(centered_means)
        
        if verbose:
            print('Calculating inverse')
        s_inv = LA.pinv(S)
        if verbose:
            print('Done inverse calculation')
            #calcualting eigenvalues and eigenvectors                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            print('Calculating eigenvectors')
        eig_vals, eig_vecs = LA.eig(np.dot(s_inv, B))
        if verbose:
            print('Done eigenvectors calculation')
        print(eig_vals,eig_vecs, sep='\n')
        index = (-eig_vals).argsort()[:r]                #sorting the first r eigenvals 
        sortedEigenvecs = eig_vecs[:,index]
        self._proj_mat = np.real(sortedEigenvecs)
        if verbose:
            print('computing projected data')
      
        if verbose:
            print(f'Mean vector class {cl} is {mean_vectors}.')
            print('within-class Scatter Matrix:\n', S)
            print('between-class Scatter Matrix:\n', S)
            print(f'sorted Eigenvectors\n{sortedEigenvecs}')
        if load_path is not None:
            with open(load_path,'wb+') as f:
                np.save(f,self._proj_mat)

        return sortedEigenvecs


    def project(self, X):
        return X.dot(self._proj_mat.T)