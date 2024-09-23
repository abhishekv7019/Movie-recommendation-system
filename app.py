from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies1.csv')


def find_similar_movies(movie_id,  k, metric='cosine'):
    
    df=ratings
  
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
        
   
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
        
    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
 
    X = X.T
    neighbour_ids = []
    
    if movie_id not in movie_mapper:
        print("Movie ID not found in the dataset.")
        return neighbour_ids
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


while True:
    try:
        movie_id = int(input("Enter movie id for movie recommendation , please refer movies1.csv file to enter a valid movie_id:"))
        break  
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

    

similar_movies = find_similar_movies(movie_id,  10, metric='cosine')

if(len(similar_movies)>0):
    movie_title = movies[movies['movieId']==movie_id]['title'].values[0]
    print(f"Because you watched {movie_title}:")
    for i in similar_movies:
        print(movies[movies['movieId']==i]['title'].values[0])