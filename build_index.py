"""
build_index.py
Run this once to fingerprint every song in songs/ and save the database.
Usage: python build_index.py
"""
import os
import pickle
from fingerprint import build_database

SONGS_DIR = "songs"
OUTPUT_FILE = "song_database.pkl"

if __name__ == "__main__":
    song_paths = [os.path.join(SONGS_DIR, f) for f in os.listdir(SONGS_DIR)
                  if f.lower().endswith(('.mp3', '.wav'))]
    print(f"Found {len(song_paths)} songs:")
    for p in song_paths:
        print(" -", p)

    print("\nIndexing... (this can take a while for large databases)")
    db = build_database(song_paths)
    print(f"Done. {len(db)} unique hashes.")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(db, f)
    print(f"Saved to {OUTPUT_FILE}")
