#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import operator
import ast

class ReadData:
    
    def __init__(self, n_rows, n_users, n_books):
        self.n_rows = n_rows
        self.n_users = n_users
        self.n_books = n_books
        
    def load_ratings_data(self):
        """
        loads a subset of user ratings data from the data file and restructures it into a user-book ratings matrix with 
        n_users rows and n_books columns
        """
    
        #load n_rows from the data file
        df=pd.read_csv('../data/some_goodreads_interactions.csv', sep=',', nrows=self.n_rows)
        data_array = df.to_numpy()

        #create dictionaries for book count and user-book ratings
        books = {}
        user_ratings = {}
        for row in data_array:
            books[row[1]] = 0
            if row[0] < self.n_users:
                user_ratings[row[0]] = {}
        for row in data_array:
            if row[0] < self.n_users:
                if row[1] in books.keys():
                    books[row[1]] += 1
                    user_ratings[row[0]][row[1]] = row[3]

        #select the 100 most frequently rated books
        sorted_books = dict(sorted(books.items(), key=operator.itemgetter(1),reverse=True))
        top_books = np.fromiter(sorted_books.keys(), dtype=int)[:self.n_books]

        #create the user-book ratings matrix
        sparse = np.zeros((self.n_users, self.n_books))
        for user in range(self.n_users):
            for book in user_ratings[user]:
                if book in top_books:
                    i = np.where(top_books==book)
                    sparse[user][i[0][0]] = user_ratings[user][book]

        del_users = []
        for i in range(len(sparse)):
            if np.all(sparse[i]==0):
                del_users.append(i)
        ratings = np.delete(sparse,temp,axis=0)

        sparse = sparse[:,~np.all(ratings==0, axis=0)]

        return (sparse, top_books, del_users)

class VisualizeData:
    
    def __init__(self, book_ids):
        self.book_ids = book_ids
        # self.author_id = author_id
      #   self.mapped_books = books

    def map_books(self):
        df=pd.read_csv('../data/book_id_map.csv', sep=',')
        data_array = df.to_numpy()

        real_ids = []
        for book in book_ids:
          real_ids.append(data_array[book][1])

        return real_ids

    def map_author(self, author_id):
        df=pd.read_csv('../data/goodreads_book_authors.csv', sep=',')
        data_array = df.to_numpy()
        i = np.where(data_array[0:]==author_id)[0][0]
        name = data_array[i][1]

        return name

    def display_book_info(self):

        mapped_books = self.map_books(books)

        df=pd.read_csv('../data/goodreads_books.csv', sep=',')
        data_array = df.to_numpy()

        for book in mapped_books:
            i = np.where(data_array[0:]==book)[0][0]
            l = data_array[i]
            temp = ast.literal_eval(l[2]) 
            a_id = int(temp[0]['author_id'])
            print("Title:", l[1])
            print("Author:", self.map_author(a_id))
            print("Description:", l[3])
            print('\n')


    


# In[ ]:




