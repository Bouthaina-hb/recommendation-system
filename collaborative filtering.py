# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:52:13 2020

@author: dell
"""

# importing libraries
import pandas as pd
import numpy as np

ratings = pd.read_csv("ratings_new.csv")

after_pca = pd.read_csv("after_pca.csv")

after_autoencoder = pd.read_csv("after_autoencoder.csv")

movies = pd.read_csv("movies_new.csv")

"""
Building collaborative filtering model from scratch
We will recommend movies based on user-user similarity. 

Now, we will create a user-item matrix which can be used to calculate the similarity 
between users and items.
"""
after_pca.set_index('userId', inplace=True)
#print(after_pca)
rating_matrix = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
after_autoencoder.set_index('userId', inplace=True)
#print(rating_matrix)
#print(after_pca)
"""
Write a function to find the most similar users to the current_user using cosine 
similarity. We’ve arbitrarily decided to find the 3 most similar users.
And picked “294” as our current user.
"""
from sklearn.metrics.pairwise import cosine_similarity
import operator
def similar_users(user_id, matrix, k=3):
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
    users = [u[0] for u in top_users_similarities]
    
    return users


#print(similar_user_indices)
#=> [51902, 15513, 92616]

"""
Now write a function to make the recommendation. We’ve set the function to return 
the 3 top recommended movies.
"""
def recommend_item(user_index, similar_user_indices, matrix, items=3):
    
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    # remove any rows without a 0 value. Movie not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
    # generate a list of movies the user has not seen
    movies_unseen = user_df_transposed.index.tolist()
    
    # filter avg ratings of similar users for only movie the current user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(movies_unseen)]
    # order the dataframe
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)
    # grab the top n movie   
    top_n_movies = similar_users_df_ordered.head(items)
    top_n_movie_indices = top_n_movies.index.tolist()
    # lookup these movies in the other dataframe to find names
    movie_information = movies[movies['movieId'].isin(top_n_movie_indices)]
    
    return movie_information #items
# try it out
user_ids = rating_matrix.index
print(rating_matrix.info())
print(user_ids)
n = int(input("Les filmes recommandé à l'utilisateur : "))
similar_user_indices = similar_users(n, after_pca)
recommendation = recommend_item(n, similar_user_indices, rating_matrix)
print("Après l'application de l'ACP, Les utilisateurs similaire à l'utilisateur "+ str(n)+" ont aimé :")
print(recommendation)

similar_user_indices = similar_users(n, after_autoencoder)
recommendation = recommend_item(n, similar_user_indices, rating_matrix)
print("\n131Après l'application de l'Autoencoder, Les utilisateurs similaire à l'utilisateur "+ str(n)+" ont aimé :")
print(recommendation)


