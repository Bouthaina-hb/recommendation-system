# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:40:12 2020

@author: dell
"""

# importing libraries
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from matplotlib import pyplot

ratings = pd.read_csv("ratings_new.csv")

after_pca = pd.read_csv("after_pca500.csv")

after_autoencoder = pd.read_csv("after_autoencoder.csv")
#movies = pd.read_csv("movies_new.csv")

"""
Building collaborative filtering model from scratch
We will recommend movies based on user-user similarity. 

Now, we will create a user-item matrix which can be used to calculate the similarity 
between users and items.
"""

after_pca.set_index('userId', inplace=True)
after_autoencoder.set_index('userId', inplace=True)

rating_matrix = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)


# ----------------- Split data : train,test -----------------------

from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(after_autoencoder, test_size = 0.25)

rating_matrix = rating_matrix.loc[train_data.index, :]


# --------------------- Similarité ------------------------------

"""
Write a function to find the most similar users to the current_user using cosine 
similarity. We’ve arbitrarily decided to find the 3 most similar users.
And picked “294” as our current user.
"""

from sklearn.metrics.pairwise import cosine_similarity
import operator

 
def similar_u(matrix, k=40):
    dictionary = {}
    for user_id in matrix.index:
    # create a df of just the current user
        user = matrix[matrix.index == user_id]   
    # and a df of all other users
        other_users = matrix[matrix.index != user_id]    
    # calc cosine similarity between user and each other user
        similarities = cosine_similarity(user,other_users)[0].tolist()    
    # create list of indices of these users
        indices = other_users.index.tolist()   
    # create key/values pairs of user index and their similarity
        index_similarity = dict(zip(indices, similarities))
    # sort by similarity
        index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
        index_similarity_sorted.reverse()  

    # grab k users off the top
        top_users_similarities = index_similarity_sorted[:k]
        #print(top_users_similarities)        
        dictionary[user_id]= top_users_similarities            
        #users = [u[0] for u in top_users_similarities]
    #print(type(users))
    return dictionary  


# ----------------------Prédiction -----------------------------------


"""
Now write a function to make the recommendation. We’ve set the function to return 
the 5 top recommended movies.
"""

def recommend_items(train_data, matrix):
    similar_user_dict = similar_u(train_data)
    liste = []
    # load vectors for similar users
    #print(matrix)
    for p_id, p_info in similar_user_dict.items():
        for key in p_info:
            liste.append(key[0])
        similar_users = matrix[matrix.index.isin(liste)]
        liste.clear()  
   
        # calc avg ratings across the similar users
        similar_users = similar_users.mean(axis=0)
    
        # convert to dataframe so its easy to sort and filter
        similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
        # load vector for the current user
        user_df = matrix[matrix.index == key[0]]
        # transpose it so its easier to filter
        user_df_transposed = user_df.transpose()

        # rename the column as 'rating'
        user_df_transposed.columns = ['rating']
        # remove any rows without a 0 value. We keep only the movies not watched yet
        #print(user_df_transposed)        
        user_df_transposed0 = user_df_transposed[user_df_transposed['rating']==0]
        # generate a list of movies the user has not seen
        movies_unseen = user_df_transposed0.index.tolist()
    
        # filter avg ratings of similar users for only movie the current user has not seen
        similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(movies_unseen)]
        similar_users_df_filtered.columns = ['rating']
        #print(similar_users_df_filtered)
        
        #final_user_rating = similar_users_df_filtered.loc[similar_users_df_filtered.index.isin(user_df_transposed.index), ['rating']] = user_df_transposed[['rating']]
        final_user_rating = similar_users_df_filtered.combine_first(user_df_transposed).reindex(user_df_transposed.index)
        #print(final_user_rating)
        matrix.loc[key[0]] = final_user_rating['rating']

    return matrix 


def precision (similar_users_bestmovies_ordered, k=40):
    relevance = 0
    top_k_movies = similar_users_bestmovies_ordered.head(k)
    liste = top_k_movies['rating_pr'].tolist()
    for i in liste :
        if i>=3:
            relevance = relevance +1
    prec = k /(k + relevance)
    if relevance == 0:
        recall = 0
    else: 
        recall = relevance/(k + relevance)
    return prec, recall

"""
def rmse(self, pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(np.mean(np.power(pred - actual, 2)))

def mae(self, pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.mean(np.abs(pred - actual))
"""    
def recommend_item_pr(train_data, test_data, matrix, items = 40):
    similar_user_dict = similar_u(train_data)
    liste = []
    prec = []
    recal = []
    # load vectors for similar users
    #print(matrix)
    for p_id, p_info in similar_user_dict.items():
        for key in p_info:
            liste.append(key[0])
        similar_users = matrix[matrix.index.isin(liste)]
        liste.clear()  
   
        # calc avg ratings across the similar users
        similar_users = similar_users.mean(axis=0)
    
        # convert to dataframe so its easy to sort and filter
        similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
        # load vector for the current user
        user_df = matrix[matrix.index == key[0]]
        # transpose it so its easier to filter
        user_df_transposed = user_df.transpose()

        # rename the column as 'rating'
        user_df_transposed.columns = ['rating']
        # remove any rows without a 0 value. We keep only the movies not watched yet
        #print(user_df_transposed)        
        user_df_transposed0 = user_df_transposed[user_df_transposed['rating']==0]
        # generate a list of movies the user has not seen
        movies_unseen = user_df_transposed0.index.tolist()
    
        # filter avg ratings of similar users for only movie the current user has not seen
        similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(movies_unseen)]

        # order the dataframe
        similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)
        similar_users_df_ordered.columns = ['rating_pr']

        p, r = precision (similar_users_df_ordered)
        prec.append(p)
        recal.append(r)
    liste = []
    for i in matrix.index:
        liste.append(i)
    #print(liste)
    #print(prec)
    """
    fig = plt.figure(figsize =(3, 10)) 
    plt.boxplot(prec) 
    plt.show() 
    """

    performance = []
    #return movie_information #items
    precision_ = sum(prec) / len(prec) 
    recall = sum(recal) / len(recal)
    performance.append(precision_)
    performance.append(recall)
    obj = ('précision','recall')
    y_pos = np.arange(len(obj))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, obj)

    plt.ylabel('')
    plt.title("La précision et le recall de l'ACP")
    plt.show()
    print("the precision is :" + str(precision_))
    print("the recall is :" + str(recall))

train_data, test_data = train_test_split(after_pca, test_size = 0.25)
rating_matrix = rating_matrix.loc[train_data.index, :]
print("Evaluation de la recommandation après l'application de l'ACP :")
rating_predictions = recommend_item_pr(train_data,test_data, rating_matrix)

train_data, test_data = train_test_split(after_autoencoder, test_size = 0.25)
rating_matrix = rating_matrix.loc[train_data.index, :]
print("Evaluation de la recommandation après l'application de l'autoencoder :")
rating_predictions = recommend_item_pr(train_data,test_data, rating_matrix)




