import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class MusicPopularityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
        self.target = 'popularity'
        
    def load_data(self, filepath):
        """Load and preprocess the dataset."""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded with {len(df)} tracks")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        # Handle missing values
        df = df.dropna(subset=self.features + [self.target])
        
        # Convert duration from ms to minutes
        df['duration_min'] = df['duration_ms'] / 60000
        
        # Drop unnecessary columns
        df = df[self.features + [self.target]]
        
        return df
    
    def train(self, X, y):
        """Train the model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        print("\nModel Evaluation:")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        
        return self.model
    
    def predict(self, features):
        """Predict popularity for given features."""
        if not hasattr(self, 'model'):
            raise Exception("Model not trained. Please train the model first.")
        return self.model.predict([features])[0]
    
    def save_model(self, filename='music_popularity_model.pkl'):
        """Save the trained model to disk."""
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='music_popularity_model.pkl'):
        """Load a trained model from disk."""
        if os.path.exists(filename):
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"No model found at {filename}")
            return False

def main():
    # Example usage
    predictor = MusicPopularityPredictor()
    
    # For demonstration, we'll use a sample dataset
    # In practice, you would load your actual dataset here
    print("Please load your dataset (CSV file) with Spotify audio features.")
    print("Required features:", predictor.features)
    print("Target variable: popularity (0-100)")
    
    # Here you would typically load your dataset
    # df = predictor.load_data('your_dataset.csv')
    # if df is not None:
    #     df = predictor.preprocess_data(df)
    #     X = df[predictor.features]
    #     y = df[predictor.target]
    #     predictor.train(X, y)
    #     predictor.save_model()

if __name__ == "__main__":
    main()
