import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

def recommend_n_books(sparse, filled, used_books, user_id, n=5):
    """
    Uses a user's row of filled ratings to recommend n books to the user, by removing the common elements between
    the sparse and filled matrix rows and finding the top n rated books.
    """
    user_row = filled[user_id]
    already_rated = sparse[user_id]
    new_books = user_row - already_rated
    ind = np.argpartition(new_books, -n)[-n:]
    temp_top = ind[np.argsort(new_books[ind])[::-1]]
    top = [used_books[i] for i in temp_top]

    return top


class UsingNorm:
    
    def __init__(self, sparse, filled, used_books, user_id):
        self.sparse = sparse
        self.filled = filled
        self.used_books = used_books
        self.user_id = user_id
        
    def rec_new_user_norm(self):
        """
        Takes a row of ratings belonging to a new user and recommend n books by finding the existing user closest
        to them, by calculating the norm between the new ratings row and the ratings matrix.
        """
        df=pd.read_csv('../data/some_goodreads_interactions.csv', sep=',', nrows=600000)
        data_array = df.to_numpy()
        
        u_rats = np.zeros((len(self.used_books)))
        for row in data_array:
            if row[0] == self.user_id:
                if row[1] in self.used_books:
                    i = np.where(self.used_books==row[1])
                    u_rats[i[0][0]] = row[3]
                    
        distance = np.linalg.norm(self.sparse-u_rats, axis=1)
        closest = np.where(distance == np.amin(distance))[0][0]
        recs = recommend_n_books(self.sparse, self.filled, self.used_books, closest)

        return (closest,recs)  

class UsingKMeans:
    
    def __init__(self, sparse, filled, used_books, user_id, del_users):
        self.sparse = sparse
        self.filled = filled
        self.used_books = used_books
        self.user_id = user_id
        self.del_users = del_users
        
    def clustering(self):
        """
        Performs k-means clustering
        """
        kmeans = KMeans(n_clusters=8, max_iter=100).fit(self.sparse)
        labels = kmeans.labels_
        i = len(labels)
        while i < 1000:
            labels = np.append(labels,0)
            i += 1
        all_users = labels
        for i in range(len(labels)):
            np.insert(all_users,i,labels[i])
        for i in self.del_users:
            np.insert(all_users,i,9)
        clusters = {}
        for i in range(8):
            clusters[i] = []
        for i in range(len(all_users)):
            if all_users[i] in clusters:
                clusters[all_users[i]].append(i)
      
        return (kmeans, clusters)
    
    def rec_new_user_kmeans(self):
        """
        Recommends n books to a new user using their ratings row by finding the cluster that they belong to
        and selecting the most reecommended books to the users in that cluster
        """
    
        (k,clusters) = self.clustering()

        df=pd.read_csv('../data/some_goodreads_interactions.csv', sep=',', nrows=600000)
        data_array = df.to_numpy()

        u_rats = np.zeros((len(self.used_books)))
        for row in data_array:
            if row[0] == self.user_id:
                if row[1] in self.used_books:
                    i = np.where(self.used_books==row[1])
                    u_rats[i[0][0]] = row[3]

        c = k.predict([u_rats])
        bs = np.array([recommend_n_books(self.sparse, self.filled, self.used_books, user) for user in clusters[c[0]]])
        books = np.ndarray.flatten(bs)
        b = Counter(books)
        m = []
        for i in b.most_common(5):
            m.append(i[0])
        return np.array(m)



