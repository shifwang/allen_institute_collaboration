import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
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
    def fit_transform(self, Ks, parallel = True, processes = 4):
        self.fit(Ks, parallel = parallel, processes = processes)
        return self.transform(Ks)
    def fit_single_trial(self, k, seed):
        nmf = nmf_with_missing_values(n_outer_loops = 4, 
                                      save_space = True,
                                      n_components = k, 
                                      init = 'random', 
                                      random_state = seed)
        if self.X is None:
            try:
                X = np.load('X_for_parallel.npz')['X']
                nmf.fit(X)
            except Exception as e:
                print(e)
                raise ValueError('self.X is None and cannot find it in disk.')
        else:
            nmf.fit(self.X)
        filename = self.folder_name + '/k=' + str(k) + '/nmf_' + str(seed) + '.pickle'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as f:
            pickle.dump(nmf, f)
        
    def fit(self, Ks, parallel = True, processes = 4):
        if isinstance(Ks, int):
            Ks = [Ks]
        # store X as a file
        if parallel:
            np.savez('X_for_parallel.npz', X = self.X)
            self.X = None
        for k in Ks:
            if parallel:
                args = [(k, self.random_state + i + 10000 * k) for i in range(self.n_trials)]
                p = Pool(processes = processes)
                p.starmap(self.fit_single_trial, args)
            else:
                for i in range(self.n_trials):
                    seed = self.random_state + i + 10000 * k
                    self.fit_single_trial(k, seed)
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
    def transform_cv(self, Ks, nfolds = 2, num_samples = 200, error_type = 'MSE_excluding_missing_values', use_training_error = False):
        cross_val_mean, cross_val_SE = [], []
        # define score
        if error_type == 'MSE_excluding_missing_values':
            scoring = metrics.make_scorer(lambda y, y_pred: np.mean(((y - y_pred)[y >= 0])**2) if max(y) >= 0 else np.mean(y**2), greater_is_better = False)
        elif error_type == 'RMSE_excluding_missing_values':
            scoring = metrics.make_scorer(lambda y, y_pred: np.mean(((y - y_pred)[y >= 0])**2) / np.mean(y[y >=0]**2) if max(y) >= 0 else 0, greater_is_better = False)
        elif error_type == 'MSE':
            scoring = 'neg_mean_squared_error'
        elif error_type == 'RMSE':
            scoring = 'explained_variance'
        else:
            raise ValueError('error_type is not found.')
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
            if self.X is None:
                try:
                    self.X = np.load('X_for_parallel.npz')['X']
                except Exception as e:
                    print(e)
                    raise ValueError('self.X is None and cannot find it in disk.')
            print("Calculating prediction instability for " + str(k))
            scores = np.zeros(shape=(num, ))
            total_sample = self.X.shape[0]
            if num_samples > total_sample:
                num_samples = total_sample
                print('num_samples larger than total_sample, force it to be smaller.')
            for i in range(num):
                x = Dhat[i]
                lm = LinearRegression(copy_X = False, fit_intercept = False)
                targets = np.random.choice(list(range(total_sample)), num_samples, replace=False)
                #print(cross_val_score(lm, x, data[:,targets], cv=cv, scoring = 'explained_variance'))
                for j in targets:
                    if not use_training_error:
                        scores[i] += - np.mean(cross_val_score(lm, x, self.X[j,:], cv=nfolds, scoring = scoring))
                    else:
                        lm.fit(x, self.X[j,:])
                        scores[i] += - scoring(y_true = self.X[j,:], estimator = lm, X = x)
                scores[i] /= num_samples

            cross_val_mean.append(np.mean(scores))
            # The standard error
            cross_val_SE.append(np.std(scores) / len(scores) ** .5)

        output = np.array([cross_val_mean, cross_val_SE]).T
        return output
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
