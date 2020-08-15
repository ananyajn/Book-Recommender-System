#!/usr/bin/env python
# coding: utf-8

# In[2]:


from dataprocessing import ReadData, VisualizeData
from algorithms import Algorithms
from recommender import recommend_n_books, UsingNorm, UsingKMeans
from evaluate import Evaluate 

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
import matplotlib.pyplot as plt
import time

def reco_sys():
    # Load the user-book ratings
    rd = ReadData(500000, 1000, 100)
    (sparse_ratings, books_used, deleted_users) = rd.load_ratings_data()

    # Obtain the filled matrix using iterative singular value thresholding
    a = Algorithms()
    (ratings_with_nan, filled_ratings) = isvt(sparse_ratings)

    # Recommend books to a user who exists in the system: User #59
    recommend_n_books(ratings, filled_ratings, books_used, 59)

    #Recommend books to a user who does not exist in the system, using norm calculations: User #1004
    rec_n = UsingNorm(sparse_ratings, filled_ratings, books_used, 1004)
    books_norm = rec_n.rec_new_user_norm()
    vd1 = VisualizeData(books_norm)
    vd1.display_book_info()

    #Recommend books to a user who does not exist in the system, using kmeans clustering: User #1004
    rec_km = UsingKMeans(sparse_ratings, filled_ratings, books_used, 1004, deleted_users)
    books_km = rec_km.rec_new_user_kmeans()
    vd2 = VisualizeData(books_km)
    vd2.display_book_info()
    
def ev():
    
    rd = ReadData(500000, 1000, 100)
    (sparse_ratings, books_used, deleted_users) = rd.load_ratings_data()

    # Obtain the filled matrix using iterative singular value thresholding
    a = Algorithms(sparse_ratings)
    (ratings_with_nan, filled_ratings_isvt) = a.isvt()
    e = Evaluate(5, 10)
    mse_isvt = e.performance_eval_isvt(formatted_r, filled_ratings_isvt)
    
    # Obtain the filled matrix using non-negative matrix factorization
    filled_ratings_nmf = a.nmf()
    mse_nmf = e.performance_eval_nmf(sparse_ratings, filled_ratings_nmf)
    
    
    
    
    
if __name__ == '__main__':
    reco_sys()
    ev()




# In[ ]:




