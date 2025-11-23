import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    # Basic preprocessing
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def preprocess_data(df):
    """Preprocess the data for modeling."""
    # Select features and target
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
               'speechiness', 'acousticness', 'instrumentalness',
               'liveness', 'valence', 'tempo', 'duration_ms']
    
    X = df[features]
    y = df['popularity']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    return y_pred

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('spotify_songs.csv')
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    print("\nFeature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main()
