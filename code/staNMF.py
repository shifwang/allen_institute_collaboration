import numpy as np
import sklearn 
import time
import pickle
import os, sys
from nmf_with_missing_values import nmf_with_missing_values

class instability:
    def __init__(self, X, folder_name = '.', random_state = 42, n_trials = 100):
        self.X = X
        self.folder_name = folder_name
        self.random_state = random_state
        self.n_trials = n_trials
    def fit_transform(self, Ks):
        self.fit(Ks)
        output = self.transform(Ks)
        self.ins_mean, self.ins_SE = output[:,0], output[:,1]
    def fit(self, Ks):
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
    def transform(self, Ks):
        # TODO
        pass
        
                    
            
            