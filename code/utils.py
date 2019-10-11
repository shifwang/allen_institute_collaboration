import numpy as np
import scipy as sp

def shrink(PPs, shrink_ratio = 0.1):
    '''
    Shrink the Principle Patterns (PPs) to a smaller area.
    
    Input:
        PPs : n * d matrix or n * d1 * d2 * .. tensor, stores n principle patterns.
                The loadings should be non-negative.
        shrink_ratio : numeric, the remaining non-zero / the total number of non-zeros
                for each PP. Should be between 0 and 1.
    
    Output:
        shrink_PPs: array that has the same dimension as PPs, 0-1 matrix/tensor that 
                stores the PPs after shrinkage.
    Example:
        >>> n = 10 # number of patterns
        >>> d = 1000 # dimension
        >>> PPs = np.maximum(np.random.normal((n, d)), 0)
        >>> shrink_PPs = shrink(PPs, shrink_ratio = 0.1)
    '''
    # sanity check
    assert shrink_ratio >= 0 and shrink_ratio < 1, \
    "shrink_ratio should be between 0 and 1, but got {}".format(shrink_ratio)
    assert type(PPs) == np.ndarray, "PPs should be numpy array."
    assert len(PPs.shape) >= 2, "PPs.shape should have at least two dimensions."
    assert np.min(PPs) >= 0, "the minimum of PPs should be non-negative."
    
    # main body
    shrink_PPs = PPs.copy()
    for pp_ind in range(PPs.shape[0]):
        loadings = PPs[pp_ind].flatten()
        loadings = loadings[loadings > 0]
        threshold = np.quantile(loadings, 1 - shrink_ratio)
        shrink_PPs[pp_ind] = PPs[pp_ind] > threshold
    return shrink_PPs

def loadings_to_vector_sets(PPs):
    '''
    Turn an array to a set of vectors, each vector indicating the position of a non-zero element
    '''