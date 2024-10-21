import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    
    def __init__(self, movie_similarity_df):
        self.movie_similarity_df = movie_similarity_df

    def recommend(self, user_high_rated_movies, num_recommendations=5):
        content_scores = self.movie_similarity_df.loc[user_high_rated_movies].mean(axis=0)
        return content_scores.sort_values(ascending=False).head(num_recommendations)
 
