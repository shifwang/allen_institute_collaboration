import numpy as np
import sklearn 
import sklearn.decomposition
import time
class nmf_with_missing_values(sklearn.decomposition.NMF):
    def __init__(self, **kargs):
        if 'n_outer_loops' in kargs:
            self.n_outer_loops_ = kargs['n_outer_loops']
            del kargs['n_outer_loops']
        else:
            self.n_outer_loops_ = 1
        if 'save_space' in kargs:
            self.save_space = True
            del kargs['save_space'] 
        else:
            self.save_space = False
        super(nmf_with_missing_values, self).__init__(**kargs)
    def fit_transform(self, X, y = None, W = None, H = None):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        X_guess = np.maximum(X, 0)
        for iter in range(self.n_outer_loops_):
            # if the initialization is given, set self.init to custom
            if W is not None and H is not None:
                nmf_with_missing_values.init = 'custom'
            W = super(nmf_with_missing_values, self).fit_transform(X_guess, y, W, H)
            H = self.components_
            # update X_guess

            X_guess[X < 0] = (W @ H)[X < 0]
        if not self.save_space:
            self.X_guess = X_guess
        else:
            self.X_guess = None
        return W
