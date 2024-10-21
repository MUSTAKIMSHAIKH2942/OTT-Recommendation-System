from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
import logging

class HybridRecommender:
    
    def __init__(self, user_movie_ratings, movie_similarity_df):
        self.collaborative_recommender = CollaborativeRecommender(user_movie_ratings)
        self.content_based_recommender = ContentBasedRecommender(movie_similarity_df)

    def recommend(self, user_id, user_ratings, num_recommendations=5, collaborative_weight=0.5, content_weight=0.5):
        logging.info(f"Generating hybrid recommendations for user {user_id}")

        # Get collaborative recommendations
        collaborative_recommendations = self.collaborative_recommender.recommend(user_id, num_recommendations)

        # Get content-based recommendations
        user_high_rated_movies = user_ratings[user_ratings > 4].index
        content_recommendations = self.content_based_recommender.recommend(user_high_rated_movies, num_recommendations)

        # Combine scores
        hybrid_scores = (collaborative_weight * collaborative_recommendations) + (content_weight * content_recommendations)

        return hybrid_scores.sort_values(ascending=False).head(num_recommendations)
 
