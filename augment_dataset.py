import csv
import random
import os
from datetime import datetime

# Configuration
DATASET_PATH = 'spotify_songs.csv'
BACKUP_DIR = 'backups'
NEW_ROWS = 300  # how many synthetic rows to add

FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'popularity'
]

# Simple helper distributions to keep values realistic

def rand_float(a, b, dp=2):
    return round(random.uniform(a, b), dp)


def synth_row():
    danceability = rand_float(0.3, 0.9, 2)
    energy = rand_float(0.3, 0.95, 2)
    key = random.randint(0, 11)
    loudness = round(random.uniform(-13.0, -4.0), 1)
    mode = random.randint(0, 1)
    speechiness = rand_float(0.02, 0.12, 2)
    acousticness = rand_float(0.02, 0.7, 2)
    # 90% of tracks non-instrumental, small chance of higher instrumentalness
    instrumentalness = 0.0 if random.random() < 0.9 else rand_float(0.1, 0.8, 2)
    liveness = rand_float(0.05, 0.35, 2)
    valence = rand_float(0.2, 0.85, 2)
    tempo = round(random.uniform(85, 145))
    duration_ms = random.randint(150_000, 270_000)
    time_signature = random.choice([3, 4, 5, 6, 7])

    # Popularity loosely correlated with some features
    base = 50
    base += int((danceability - 0.5) * 40)
    base += int((energy - 0.5) * 35)
    base += int((0.6 - abs(valence - 0.6)) * 30)
    base += int((120 - abs(tempo - 120)) / 8)
    base -= int((abs(loudness + 7.5)) * 2)
    base -= int(instrumentalness * 30)
    popularity = max(0, min(100, base + random.randint(-10, 10)))

    return [
        f"{danceability:.2f}", f"{energy:.2f}", str(key), f"{loudness:.1f}", str(mode),
        f"{speechiness:.2f}", f"{acousticness:.2f}", f"{instrumentalness:.2f}",
        f"{liveness:.2f}", f"{valence:.2f}", str(tempo), str(duration_ms), str(time_signature), str(popularity)
    ]


def ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_file(path):
    ensure_backup_dir()
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    base = os.path.basename(path)
    dst = os.path.join(BACKUP_DIR, f"{base}.{ts}.bak")
    with open(path, 'rb') as src, open(dst, 'wb') as out:
        out.write(src.read())
    return dst


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return

    # Read header and validate
    with open(DATASET_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Dataset is empty.")
        return

    header = rows[0]
    if header != FEATURES:
        print("Header does not match expected features order.")
        print("Expected:", FEATURES)
        print("Found   :", header)
        # Continue anyway but warn

    # Backup original file
    backup = backup_file(DATASET_PATH)
    print(f"Backup created: {backup}")

    # Append synthetic rows
    new_data = [synth_row() for _ in range(NEW_ROWS)]

    with open(DATASET_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(new_data)

    print(f"Appended {NEW_ROWS} synthetic rows to {DATASET_PATH}")


if __name__ == '__main__':
    main()
