import numpy as np
import sklearn
from pca_with_missing_values import pca_with_missing_values
from multiprocessing import Pool
def f(i):
    tmp = np.load('../data/mouse_brain_ISH_float32.npz', allow_pickle=True)
    data = tmp['data']
    sections = tmp['sections']
    original_shape = data.shape
    d = data.shape[1] * data.shape[2] * data.shape[3]
    data = np.reshape(data, (data.shape[0], d))
    pca = pca_with_missing_values(n_outer_loops = 4, n_components = 18, random_state = None)
    coefs = pca.fit_transform(sklearn.utils.resample(data, random_state=i))
    PPs = pca.components_
    print('result {} fingerprint: {}'.format(i, np.mean(PPs)))
    np.savez('stability_pca_result_comp_18_{}.npz'.format(i), PPs = PPs, coefs = coefs, X_guess = None, data = None, original_shape = original_shape)
a = Pool(20)
a.map(f, range(20))
