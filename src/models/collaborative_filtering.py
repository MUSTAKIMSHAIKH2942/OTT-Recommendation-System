import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class CollaborativeRecommender:
    
    def __init__(self, user_movie_ratings):
        self.user_movie_ratings = user_movie_ratings
        self.user_similarity_df = cosine_similarity(user_movie_ratings)

    def recommend(self, user_id, num_recommendations=5):
        logging.info(f"Generating collaborative recommendations for user {user_id}")
        similar_users = self.user_similarity_df[user_id].argsort()[-num_recommendations-1:-1]
        recommendations = self.user_movie_ratings.iloc[similar_users].mean(axis=0)
        return recommendations.sort_values(ascending=False).head(num_recommendations)

