import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def correlation_map_with_CCF(PPs, original_shape, plot=True, order_type = 1):
    ''' Compare PPs with the standard ABA CCF.
    '''
    # transform PPs to 4d tensor
    PPs_3d = np.zeros([PPs.shape[0]] + original_shape[1:].tolist())
    for i in range(PPs.shape[0]):
        p2 = np.reshape(PPs[i,:], original_shape[1:])
        PPs_3d[i,:,:,:] = p2
    # load ABA CCF coarse
    areas_atlas = np.load('mouse_coarse_structure_atlas.npy')
    mouse_coarse_df = pd.read_pickle('mouse_coarse_df')
    cor_mat = np.corrcoef(np.vstack([areas_atlas.reshape(12, -1), PPs_3d[:, :-1,:-1,:-1].reshape(18,-1)]))[:areas_atlas.shape[0], areas_atlas.shape[0]:]
    
    if order_type == 1:
        rows, cols = linear_sum_assignment(-np.abs(cor_mat))
        factor_order = list(cols) + [x for x in range(18) if x not in cols]
    elif order_type == 2:
        cols = np.argmax(np.abs(cor_mat), 0)
        factor_order = np.argsort(cols.tolist())
    if plot:
        plt.imshow(np.abs(cor_mat[:,factor_order]).tolist())
        plt.yticks(np.arange(12),(mouse_coarse_df.iloc[:]['name'].tolist()))
        plt.ylim([-0.5, 11.5])
        plt.gca().invert_yaxis()
        plt.xticks(range(18),factor_order)
        plt.title('Correlation Coefficient')
        plt.xlabel('Principle Patterns')
        plt.colorbar()
        plt.show()
    return np.abs(cor_mat[:,factor_order])
    
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