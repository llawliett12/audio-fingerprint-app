# streamlit app for Q3B - wraps the fingerprinting engine from fingerprint.py
# two modes: single clip (shows the visuals) and batch (writes results.csv)

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fingerprint import fingerprint_song, match_query

DB_FILE = "song_database.pkl"

st.set_page_config(page_title="Song Identifier", layout="wide")


@st.cache_resource
def load_database():
    if not os.path.exists(DB_FILE):
        return None
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp_path = f"_tmp_query{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


def plot_spectrogram_and_peaks(extras):
    f, t, Sxx, peaks = extras
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].pcolormesh(t, f, 20 * np.log10(Sxx + 1e-10), shading="auto", cmap="magma")
    axes[0].set_ylim(0, 3000)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Spectrogram")

    peak_t = [t[p[1]] for p in peaks]
    peak_f = [f[p[0]] for p in peaks]
    axes[1].scatter(peak_t, peak_f, s=6, color="black")
    axes[1].set_ylim(0, 3000)
    if len(t) > 0:
        axes[1].set_xlim(t[0], t[-1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"Constellation ({len(peaks)} peaks)")

    plt.tight_layout()
    return fig


def plot_offset_histogram(results):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = [r[0] for r in results[:8]]
    counts = [r[1] for r in results[:8]]
    ax.bar(names, counts, color="steelblue")
    ax.set_ylabel("Matching hashes at best offset")
    ax.set_title("Offset histogram (top candidates)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


st.title("Song Identifier")
st.caption("Spectrogram peak fingerprinting - EE200 Q3B")

db = load_database()
if db is None:
    st.error("song_database.pkl not found. Run build_index.py first.")
    st.stop()

st.sidebar.write(f"Database loaded: {len(db)} unique hashes")
mode = st.sidebar.radio("Mode", ["Single clip", "Batch"])

if mode == "Single clip":
    uploaded = st.file_uploader("Upload a query clip (mp3/wav)", type=["mp3", "wav"])

    if uploaded is not None:
        tmp_path = save_uploaded_file(uploaded)
        with st.spinner("Matching against the database..."):
            hashes, extras = fingerprint_song(tmp_path)
            results = match_query(hashes, db)

        if not results:
            st.warning("No match found.")
        else:
            best_name, best_count, best_offset = results[0]
            st.success(f"Identified as: **{best_name}**")
            st.write(f"Matching hashes at best offset: {best_count}")

            st.subheader("Spectrogram and constellation")
            st.pyplot(plot_spectrogram_and_peaks(extras))

            st.subheader("Offset histogram")
            st.pyplot(plot_offset_histogram(results))

        os.remove(tmp_path)

else:
    st.write("Upload several clips at once. Produces results.csv with "
             "columns: filename, prediction")
    uploaded_files = st.file_uploader("Upload query clips", type=["mp3", "wav"],
                                       accept_multiple_files=True)

    if uploaded_files:
        rows = []
        progress = st.progress(0)
        for i, uploaded in enumerate(uploaded_files):
            tmp_path = save_uploaded_file(uploaded)
            hashes, _ = fingerprint_song(tmp_path)
            results = match_query(hashes, db)
            prediction = results[0][0] if results else ""
            rows.append({"filename": uploaded.name, "prediction": prediction})
            os.remove(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(df)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", data=csv_bytes,
                            file_name="results.csv", mime="text/csv")
