# Music Popularity Prediction

This project predicts song popularity based on audio features using machine learning. It uses a Random Forest Regressor to analyze various musical attributes and predict a popularity score.

## Features

- Loads and preprocesses music data
- Trains a Random Forest model
- Evaluates model performance
- Visualizes feature importance
- Handles missing data and outliers

## Prerequisites

- Python 3.8+
- Required Python packages (install using `pip install -r requirements.txt`)

## Dataset

This project expects a CSV file named `spotify_songs.csv` containing music data with the following features:
- danceability
- energy
- key
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration_ms
- popularity (target variable)

## Usage

1. Place your dataset as `spotify_songs.csv` in the project directory
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the prediction script:
   ```
   python music_popularity_prediction.py
   ```

## Output

- Model evaluation metrics (MSE and R² score)
- Feature importance plot saved as `feature_importance.png`

## Model Performance

The model's performance can be evaluated using:
- Mean Squared Error (MSE): Lower is better
- R² Score: Closer to 1 is better

## Feature Importance

The script generates a bar plot showing the importance of each feature in predicting song popularity.
