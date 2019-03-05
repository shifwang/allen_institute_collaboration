
# coding: utf-8

# # Demo of using staNMF for Mouse Brain ISH data

# In[2]:


import numpy as np
import sklearn
from staNMF import instability


# ## load the data

# In[3]:


tmp = np.load('../data/mouse_brain_ISH_float32.npz')
data = tmp['data']
sections = tmp['sections']
original_shape = data.shape
d = data.shape[1] * data.shape[2] * data.shape[3]
data = np.reshape(data, (data.shape[0], d))


# ## calculate staNMF (takes a long time)

# In[3]:


ins = instability(data, n_trials=10, folder_name='mouse_brain_ISH')


# In[ ]:


Ks = list(reversed(range(23, 26)))
output = ins.fit_transform(Ks, parallel = True, processes = 5)
