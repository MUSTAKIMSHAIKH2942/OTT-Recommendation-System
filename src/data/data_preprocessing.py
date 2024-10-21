import pandas as pd

def preprocess_ratings(ratings_df):
    """Preprocess the ratings DataFrame."""
    return ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

def preprocess_movies(movies_df):
    """Preprocess the movies DataFrame."""
    movies_df['genres'] = movies_df['genres'].str.split('|')
    unique_genres = set([genre for sublist in movies_df['genres'] for genre in sublist])

    for genre in unique_genres:
        movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)

    movies_df.drop(columns=['genres'], inplace=True)
    return movies_df, unique_genres
 
