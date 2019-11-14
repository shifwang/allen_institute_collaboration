import numpy as np
import sklearn
from nmf_with_missing_values import nmf_with_missing_values
from multiprocessing import Pool
def f(i):
    tmp = np.load('../data/mouse_brain_ISH_float32.npz', allow_pickle=True)
    data = tmp['data']
    sections = tmp['sections']
    original_shape = data.shape
    d = data.shape[1] * data.shape[2] * data.shape[3]
    data = np.reshape(data, (data.shape[0], d))
    nmf = nmf_with_missing_values(n_outer_loops = 4, n_components = 18, init = 'nndsvd', random_state = None)
    D = nmf.fit_transform(sklearn.utils.resample(data, random_state = i))
    print('result {} fingerprint: {}'.format(i, np.mean(D)))
    A = nmf.components_
    np.savez('stability_result_comp_18_{}.npz'.format(i), D = D, A = A, X_guess = None, data = None, original_shape = original_shape)
a = Pool(20)
a.map(f, range(20))
