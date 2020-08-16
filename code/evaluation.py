
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from algorithms import Algorithms
import matplotlib.pyplot as plt
import random
import time

class Evaluate:
    
    def __init__(self, n_iter, holdout_percent):
        self.n_iter = n_iter
        self.holdout_percent = holdout_percent

    def performance_eval_isvt(self, sparse, filled, plot=True):
        """
        hold out a certain percent of the ratings and observe how well the algorithm approximates those ratings
        """
        mse = []
        for k in range(self.n_iter):
            test = sparse.copy()
            n_ratings = np.size(sparse) - len(np.where(np.isnan(sparse))[0])
            holdout_count = int(self.holdout_percent*n_ratings/100)
            for subs in range(holdout_count):
                m = 0
                while m == 0:
                    i = random.randint(0,len(sparse)-1)
                    j = random.randint(0,len(sparse[0])-1)
                    if np.isnan(test[i][j]) == False:
                        temp = test[i][j]
                        test[i][j] = np.nan
                        if np.all(np.isnan(test[i])) or np.all(np.isnan(test[:j])):
                            test[i][j] = temp
                        else:
                            m = 1    
            completed = SoftImpute(max_iters=100).fit_transform(test)
            mse.append(mean_squared_error(filled, completed)) 

        if plot:

            # plot mean squared error per iteration
            x = [i for i in range(1,self.n_iter+1)]
    
            plt.plot(x, mse)
            plt.xlabel("Iterations")
            plt.ylabel("Mean Squared Error")
            plt.title("Iterative Singular Value Thresholding")
            plt.show()
        
        return np.average(mse)
    
    def performance_eval_nmf(self, sparse, filled, plot=True):
        """
        hold out a certain percent of the ratings and observe how well the algorithm approximates those ratings
        """
        mse = []
        for k in range(self.n_iter):
            test = sparse.copy()
            a = Algorithms(test)
            n_ratings = np.size(sparse) - len(np.where(sparse==0)[0])
            holdout_count = int(self.holdout_percent*n_ratings/100)
            for subs in range(holdout_count):
                m = 0
                while m == 0:
                    i = random.randint(0,len(sparse)-1)
                    j = random.randint(0,len(sparse[0])-1)
                    if test[i][j] != 0:
                        temp = test[i][j]
                        test[i][j] = 0
                        if np.all(test[i]==0) or np.all(test[:][j]==0):
                            test[i][j] = temp
                        else:
                            m = 1        
            completed = a.nmf()
            mse.append(mean_squared_error(filled, completed)) 

        if plot:

            # plot mean squared error per iteration
            x = [i for i in range(1,self.n_iter+1)]
            plt.plot(x, mse)
            plt.xlabel("Iterations")
            plt.ylabel("Mean Squared Error")
            plt.title("Non-Negative Matrix Factorization")
            plt.show()
        
        return np.average(mse)


# In[2]:




