# Collaborative-based-movie-recommendation-system
Movie Recommendation System part 2
# Collaborative Filtering Recommendation System

## Overview

This project implements a collaborative filtering recommendation system using item-based similarity. The system recommends movies based on the similarity of items (movies) using cosine similarity.

## Libraries Used

- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning library for calculating cosine similarity.
- `numpy`: Numerical operations library.

## Files

- `ratings.csv`: Contains user ratings for movies with columns `userId`, `movieId`, and `rating`.
- `movies.csv`: Contains movie details with columns `movieId` and `title`.

## Code

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the ratings and movies datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Create the user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compute cosine similarity between items (movies)
item_similarity = cosine_similarity(user_item_matrix_filled.T)

# Convert into a DataFrame for easier manipulation
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_collaborative_recommendations(movie_id, user_item_matrix, item_similarity_df, movies_df, top_n=10):
    """
    Get movie recommendations based on item similarity.
    
    Args:
    - movie_id (int): The ID of the movie for which recommendations are to be made.
    - user_item_matrix (pd.DataFrame): User-item matrix with ratings.
    - item_similarity_df (pd.DataFrame): DataFrame containing item similarities.
    - movies_df (pd.DataFrame): DataFrame containing movie details.
    - top_n (int): Number of top similar movies to recommend.
    
    Returns:
    - list: List of recommended movie titles.
    """
    if movie_id not in item_similarity_df.columns:
        return "Movie not found in the dataset."
    
    # Get the similarity scores for the given movie
    similar_scores = item_similarity_df[movie_id]
    
    # Drop the input movie to avoid recommending it
    similar_scores = similar_scores.drop(movie_id)
    
    # Check if there are enough movies to recommend
    if similar_scores.empty:
        return "Not enough similar movies found."
    
    # Sort movies by similarity score
    similar_movies_ids = similar_scores.sort_values(ascending=False).index[:top_n]
    
    # Map movie IDs to movie names
    similar_movies = movies_df[movies_df['movieId'].isin(similar_movies_ids)]['title'].values
    
    return similar_movies

# Example movieId (change to an actual movieId from your dataset)
example_movie_id = 2

# Test the function
recommended_movies = get_collaborative_recommendations(example_movie_id, user_item_matrix_filled, item_similarity_df, movies)
print("Recommended Movies:")
print(recommended_movies)
```
## Advantages
- **Simple Implementation: Easy to understand and implement.
- **Item-Based Recommendations: Effective for recommending similar items.
- **Scalability: Handles large datasets reasonably well with optimization.

## Disadvantages
- **Cold Start Problem: New movies with no ratings won't be recommended.
- **Popularity Bias: Popular movies might overshadow less popular ones.
- **Scalability Issues: Computing similarity for very large datasets can be resource-intensive.
## Usage
- **Ensure you have ratings.csv and movies.csv in your working directory.
- **Run the script to load data, compute similarities, and get recommendations.
- **Feel free to modify the example movie ID to test different recommendations.

  ## License

This project is licensed under the NIT Sikkim License. For more details, see the [LICENSE](LICENSE) file.
