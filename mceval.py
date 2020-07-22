#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from fancyimpute import SoftImpute, BiScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import operator
import random

def load_ratings_data(n_rows, n_users, n_books):
    """
    loads a subset of user ratings data from the data file and restructures it into a user-book ratings matrix with 
    n_users rows and n_books columns
    """
    
    #load n_rows from the data file
    df=pd.read_csv('some_goodreads_interactions.csv', sep=',', nrows=n_rows)
    data_array = df.to_numpy()
    
    #create dictionaries for book count and user-book ratings
    books = {}
    user_ratings = {}
    for row in data_array:
        books[row[1]] = 0
        if row[0] < n_users:
            user_ratings[row[0]] = {}
    for row in data_array:
        if row[0] < n_users:
            if row[1] in books.keys():
                books[row[1]] += 1
                user_ratings[row[0]][row[1]] = row[3]
                
    #select the 100 most frequently rated books
    sorted_books = dict(sorted(books.items(), key=operator.itemgetter(1),reverse=True))
    top_books = np.fromiter(sorted_books.keys(), dtype=int)[:n_books]
    
    #create the user-book ratings matrix
    ratings = np.zeros((n_users, n_books))
    for user in range(n_users):
        for book in user_ratings[user]:
            if book in top_books:
                i = np.where(top_books==book)
                ratings[user][i[0][0]] = user_ratings[user][book]
                
    return ratings

def matrix_completion(ratings):
    """
    Matrix completion is done using the softimpute function in the fancyimpute library.
    """
    #the fancyimpute library requires that all the sparse elements in the ratings matrix be NaN. 
    #so, the zeroes are converted accordingly
    rating = ratings.copy()
    rating[np.where(rating==0)] = np.nan

    #empty rows and columns are deleted
    temp = []
    for i in range(len(rating)):
        if np.all(np.isnan(rating[i])):
            temp.append(i)
    rating = np.delete(rating,temp,axis=0)

    rating = rating[:,~np.all(np.isnan(rating), axis=0)]
    
    #the ratings matrix is first normalized and then filled
    ratings_normalized = BiScaler(max_iters=100).fit_transform(rating)
    ratings_softimpute = SoftImpute(max_iters=100).fit_transform(ratings_normalized)
    
    return (rating, ratings_softimpute)

def performance_eval(ratings, filled_ratings, n_iter, holdout_percent):
    """
    hold out a certain percent of the ratings and observe how well the algorithm approximates those ratings
    """
    mse = []
    for k in range(n_iter):
        test = ratings.copy()
        n_ratings = np.size(ratings) - len(np.where(np.isnan(ratings))[0])
        holdout_count = int(holdout_percent*n_ratings/100)
        for subs in range(holdout_count):
            m = 0
            while m == 0:
                i = random.randint(0,len(ratings)-1)
                j = random.randint(0,len(ratings[0])-1)
                if np.isnan(test[i][j]) == False:
                    temp = test[i][j]
                    test[i][j] = np.nan
                    if np.all(np.isnan(test[i])) or np.all(np.isnan(test[:][j])):
                        test[i][j] = temp
                    else:
                        m = 1    
        test_normalized = BiScaler(max_iters=100).fit_transform(test)
        test_softimpute = SoftImpute(max_iters=100).fit_transform(test_normalized)
        mse.append(mean_squared_error(filled_ratings, test_softimpute)) 
    
    #plot mean squared error per iteration
    x = [i for i in range(1,n_iter+1)]
    plt.plot(x, mse)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    

if __name__ == '__main__':
    
    ratings = load_ratings_data(500000, 1000, 100)
    (formatted_r, filled_ratings) = matrix_completion(ratings)
    performance_eval(formatted_r, filled_ratings, 10, 10)
    
    


# In[ ]:




