import numpy as np
import sklearn.preprocessing
import time
import pickle
import os, sys
from multiprocessing import Pool
from nmf_with_missing_values import nmf_with_missing_values

class instability:
    def __init__(self, X, folder_name = '.', random_state = 42, n_trials = 100):
        self.X = X
        self.folder_name = folder_name
        self.random_state = random_state
        self.n_trials = n_trials
    def fit_transform(self, Ks, parallel = True):
        if parallel:
            self.fit_parallel(Ks)
        else:
            self.fit(Ks)
        return self.transform(Ks)
    def fit_parallel(self, Ks, processes = 10):
        p = Pool(processes = processes)
        p.map(self.fit, Ks)
    def fit(self, Ks):
        if isinstance(Ks, int):
            Ks = [Ks]
        for k in Ks:
            for i in range(self.n_trials):
                seed = self.random_state + i + 10000 * k
                nmf = nmf_with_missing_values(n_outer_loops = 4, 
                                              save_space = True,
                                              n_components = k, 
                                              init = 'random', 
                                              random_state = seed)
                nmf.fit(self.X)
                filename = self.folder_name + '/k=' + str(k) + '/nmf_' + str(seed) + '.pickle'
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(nmf, f)
    def amariMaxError_(self, correlation):
        '''
        Computes what Wu et al. (2016) described as a 'amari-type error'
        based on average distance between factorization solutions

        Return:
        Amari distance distM

        Arguments:
        :param: correlation: k by k matrix of pearson correlations

        Usage: Called by self.transform(Ks)
        '''

        n, m = correlation.shape
        assert n == m, 'correlation matrix must be square.'
        maxCol = np.absolute(correlation).max(0)
        colTemp = np.mean((1-maxCol))
        maxRow = np.absolute(correlation).max(1)
        rowTemp = np.mean((1-maxRow))
        distM = (rowTemp + colTemp)/(2)

        return distM
    def findcorrelation_(self, A, B):
        '''
        Construct k by k matrix of Pearson product-moment correlation
        coefficients for every combination of two columns in A and B

        :param: A : first NMF solution matrix
        :param: B : second NMF solution matrix, of same dimensions as A

        Return: numpy array of dimensions k by k, where array[a][b] is the
        correlation between column 'a' of X and column 'b'

        Usage:
        Called by self.transform(Ks)

        '''
        #corrmatrix = []
        #for a in range(k):
        #    for b in range(k):
        #        c = np.corrcoef(A[:, a], B[:, b])
        #        corrmatrix.append(c[0][1])
        #return np.asarray(corrmatrix).reshape(k, k)
        A_std = sklearn.preprocessing.scale(A)
        B_std = sklearn.preprocessing.scale(B)
        return A_std.T @ B_std / A.shape[0]

    def transform(self, Ks):
        ins_mean, ins_SE = [], []
        for k in Ks:
            folder = self.folder_name + '/k=' + str(k)
            if not os.path.exists(folder):
                raise ValueError('folder %s not found, have you run fit first?'%folder)
            Dhat = []
            for filename in os.listdir(folder):
                with open(os.path.join(folder, filename), 'rb') as f:
                    nmf = pickle.load(f)
                Dhat.append(nmf.components_.T)
            num = len(Dhat)
            if num == 0:
                raise ValueError('folder %s is empty, have you run fit first?'%folder)
            distMat = np.zeros(shape=(num, num))
            for i in range(num):
                for j in range(i, num):
                    x = Dhat[i]
                    y = Dhat[j]
                    CORR = self.findcorrelation_(x, y)
                    distMat[i][j] = self.amariMaxError_(CORR)
                    distMat[j][i] = distMat[i][j]
            ins_mean.append(np.sum(distMat) / (num *(num-1)))
            ins_SE.append((np.sum(distMat**2) / (num*(num - 1)) - ins_mean[-1]**2)**.5 * (2 / distMat.shape[0])**.5)
        output = np.array([ins_mean, ins_SE]).T
        return output
                        

                
        
                    
            
            
