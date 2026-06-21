# Song Identifier (EE200 Q3B)

Spectrogram peak-fingerprinting song identifier, in the style of Shazam.

## Files
- fingerprint.py : core engine (spectrogram, peak picking, hashing, matching)
- build_index.py : run once to fingerprint all songs in songs/ and save song_database.pkl
- app.py : Streamlit app (single-clip mode + batch mode)
- songs/ : put all song files here before indexing
- requirements.txt : dependencies

## Run locally
1. Put your song files in songs/
2. python build_index.py
3. streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Push this whole folder to a new GitHub repo (include song_database.pkl
   after running build_index.py, and the songs/ folder, so the deployed
   app works immediately without re-indexing).
2. Go to streamlit.io/cloud, sign in with GitHub.
3. Click "New app", select the repo, set main file to app.py.
4. Deploy. You'll get a URL like yourapp.streamlit.app
