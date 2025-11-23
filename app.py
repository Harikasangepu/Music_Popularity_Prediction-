from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from music_popularity_predictor import MusicPopularityPredictor
from sklearn.exceptions import NotFittedError
import os
import json
from datetime import datetime

app = Flask(__name__)
predictor = MusicPopularityPredictor()
_model_loaded_once = predictor.load_model()
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'

def _load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def _load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

@app.route('/')
def home():
    return render_template('index.html', user=session.get('user'))

@app.route('/health')
def health():
    """Simple health check endpoint."""
    status = {
        "status": "ok",
        "model_file_exists": os.path.exists('music_popularity_model.pkl')
    }
    return jsonify(status)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        if not username:
            return render_template('login.html', error='Username is required')
        if not password:
            return render_template('login.html', error='Password is required')
        users = _load_users()
        # Strict validation: must exist and match
        if username not in users or users[username].get('password') != password:
            return render_template('login.html', error='Invalid credentials')
        session['user'] = username
        return redirect(url_for('home'))
    return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm = request.form.get('confirm') or ''
        if not username:
            return render_template('signup.html', error='Username is required')
        if not password:
            return render_template('signup.html', error='Password is required')
        if password != confirm:
            return render_template('signup.html', error='Passwords do not match')
        users = _load_users()
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = { 'password': password }
        _save_users(users)
        session['user'] = username
        return redirect(url_for('home'))
    return render_template('signup.html', error=None)
@app.route('/history')
def history():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))
    history = _load_history()
    items = history.get(user, [])
    # sort newest first
    items = sorted(items, key=lambda x: x.get('ts', ''), reverse=True)
    return render_template('history.html', user=user, items=items)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log raw form for debugging
        print('Incoming form data:', dict(request.form))

        # Extract and validate inputs
        try:
            features = [
                float(request.form.get('danceability', 0)),
                float(request.form.get('energy', 0)),
                int(request.form.get('key', 0)),
                float(request.form.get('loudness', 0)),
                int(request.form.get('mode', 0)),
                float(request.form.get('speechiness', 0)),
                float(request.form.get('acousticness', 0)),
                float(request.form.get('instrumentalness', 0)),
                float(request.form.get('liveness', 0)),
                float(request.form.get('valence', 0)),
                float(request.form.get('tempo', 0)),
                float(request.form.get('duration_ms', 0)),
                int(request.form.get('time_signature', 4))
            ]
        except (TypeError, ValueError) as conv_err:
            return jsonify({"error": f"Invalid input values: {conv_err}"}), 400
        
        # Make prediction, auto-load model if not fitted
        try:
            popularity = predictor.predict(features)
        except NotFittedError:
            # Try to load trained model and retry once
            if predictor.load_model():
                popularity = predictor.predict(features)
            else:
                return jsonify({"error": "Model not fitted and no saved model found. Please train the model first."}), 400
        
        result = {
            "predicted_popularity": round(popularity, 2),
            "features": {
                "danceability": features[0],
                "energy": features[1],
                "key": features[2],
                "loudness": features[3],
                "mode": features[4],
                "speechiness": features[5],
                "acousticness": features[6],
                "instrumentalness": features[7],
                "liveness": features[8],
                "valence": features[9],
                "tempo": features[10],
                "duration_ms": features[11],
                "time_signature": features[12]
            }
        }

        # Save to history if logged in
        user = session.get('user')
        if user:
            entry = {
                'ts': datetime.utcnow().isoformat() + 'Z',
                'prediction': result['predicted_popularity'],
                'features': result['features']
            }
            history = _load_history()
            history.setdefault(user, []).append(entry)
            _save_history(history)

        return jsonify(result)
    except Exception as e:
        # Log unexpected errors for debugging
        print('Prediction error:', repr(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
