#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from fancyimpute import SoftImpute
from sklearn.decomposition import NMF

class Algorithms:
    
    def __init__(self, sparse):
        self.sparse = sparse

    def isvt(self):
        """
        Matrix completion is done using the softimpute function in the fancyimpute library.
        """
        #the fancyimpute library requires that all the sparse elements in the sparse matrix be NaN. 
        #so, the zeroes are converted accordingly
        rating = self.sparse.copy()
        rating[np.where(rating==0)] = np.nan

        #the sparse matrix is then filled
        filled = SoftImpute(max_iters=100).fit_transform(rating)

        return (rating, filled)

    def nmf(self):
        model = NMF(init='random', random_state=0)
        W = model.fit_transform(self.sparse)
        H = model.components_
        filled = np.matmul(W,H)
        return filled


# In[ ]:




