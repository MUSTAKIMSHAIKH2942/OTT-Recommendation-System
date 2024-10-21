import logging
import pandas as pd
from src.data.data_loader import load_data
from src.data.data_preprocessing import preprocess_ratings, preprocess_movies
from src.models.hybrid_model import HybridRecommender
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_id, num_recommendations=5):
    logging.info("Loading datasets...")
    
    ratings_df = load_data('ratings.csv')
    movies_df = load_data('movies.csv')
    
    user_movie_ratings = preprocess_ratings(ratings_df)
    movies_df, unique_genres = preprocess_movies(movies_df)

    genre_matrix = movies_df[list(unique_genres)].values
    movie_similarity = cosine_similarity(genre_matrix)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=movies_df['movieId'], columns=movies_df['movieId'])

    hybrid_recommender = HybridRecommender(user_movie_ratings, movie_similarity_df)
    
    user_ratings = user_movie_ratings.loc[user_id]
    recommendations = hybrid_recommender.recommend(user_id, user_ratings, num_recommendations)
    
    recommended_movie_titles = movies_df[movies_df['movieId'].isin(recommendations.index)]['title']
    
    return recommended_movie_titles

