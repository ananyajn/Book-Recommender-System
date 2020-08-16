
from dataprocessing import ReadData, VisualizeData
from algorithms import Algorithms
from recommender import recommend_n_books, UsingNorm, UsingKMeans
from evaluation import Evaluate 

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
import matplotlib.pyplot as plt
import time

def display_recommend():
    # Load the user-book ratings
    rd = ReadData(500000, 1000, 100)
    (sparse_ratings, books_used, deleted_users) = rd.load_ratings_data()

    # Obtain the filled matrix using iterative singular value thresholding
    a = Algorithms(sparse_ratings)
    (ratings_with_nan, filled_ratings) = a.isvt()

    # Recommend books to a user who exists in the system: User #59. Changing the user ID will result in error.
    books_user = recommend_n_books(sparse_ratings, filled_ratings, books_used, 59)
    vd1 = VisualizeData(books_user)
    vd1.display_book_info()


    # Recommend books to a user who does not exist in the system, using norm calculations: User #1004. 
    # Changing the user ID will result in error.
    rec_n = UsingNorm(sparse_ratings, filled_ratings, books_used, 1004)
    books_norm = rec_n.rec_new_user_norm()
    vd2 = VisualizeData(books_norm[1])
    vd2.display_book_info()

    # Recommend books to a user who does not exist in the system, using kmeans clustering: User #1004.
    # Changing the user ID will result in error.
    rec_km = UsingKMeans(sparse_ratings, filled_ratings, books_used, 1004, deleted_users)
    books_km = rec_km.rec_new_user_kmeans()
    vd3 = VisualizeData(books_km)
    vd3.display_book_info()
    
def algorithm_evaluation():
    
    rd = ReadData(500000, 1000, 100)
    (sparse_ratings, books_used, deleted_users) = rd.load_ratings_data()

    # Obtain the filled matrix using iterative singular value thresholding and view mse per iteration
    a = Algorithms(sparse_ratings)
    (ratings_with_nan, filled_ratings_isvt) = a.isvt()
    e = Evaluate(5, 10)
    mse_isvt = e.performance_eval_isvt(ratings_with_nan, filled_ratings_isvt)
    
    # Obtain the filled matrix using non-negative matrix factorization and view mse per iteration
    filled_ratings_nmf = a.nmf()
    mse_nmf = e.performance_eval_nmf(sparse_ratings, filled_ratings_nmf)

    # Vary hold-out set and find average mse for each algorithm
    hos = [5, 10, 15, 20]
    e_isvt = []
    e_nmf = []
    for i in hos:
        e2 = Evaluate(5, i)
        mse_isvt = e2.performance_eval_isvt(ratings_with_nan, filled_ratings_isvt, plot=False)
        e_isvt.append(mse_isvt)

        mse_nmf = e2.performance_eval_nmf(sparse_ratings, filled_ratings_nmf, plot=False)
        e_nmf.append(mse_nmf)

    plt.plot(hos, e_isvt, label="Soft Impute")
    plt.plot(hos, e_nmf, label="NMF")
    plt.xlabel("Hold-Out Set %")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':

    # run the following function to view a working example of the recommender system
    display_recommend()

    # run the following function to view evaluation and comparison of the two matrix completion algorithms 
    algorithm_evaluation()

