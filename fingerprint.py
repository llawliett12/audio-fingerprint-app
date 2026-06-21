# core fingerprinting logic - spectrogram -> peaks -> hashes -> matching
# used by both the report experiments (Q3A) and the streamlit app (Q3B)

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
import librosa


def compute_spectrogram(y, sr, nperseg=2048, noverlap=1536):
    # returns f (Hz), t (s), Sxx (magnitude, linear scale)
    f, t, Sxx = spectrogram(y, fs=sr, window='hann', nperseg=nperseg,
                             noverlap=noverlap, mode='magnitude')
    return f, t, Sxx


def find_peaks(Sxx, amp_min_db=-60, neighborhood=(20, 20)):
    # local maxima that stand out from their time-frequency neighborhood
    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    above_thresh = Sxx_db > amp_min_db
    peak_mask = local_max & above_thresh
    freq_idx, time_idx = np.where(peak_mask)
    return list(zip(freq_idx, time_idx))


def generate_hashes(peaks, fan_out=5, max_time_delta_bins=200):
    """Pair each peak with up to fan_out nearby future peaks.
    hash = (f1, f2, delta_t) mapped to the anchor time bin."""
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda p: p[1])
    n = len(peaks_sorted)
    for i in range(n):
        f1, t1 = peaks_sorted[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = peaks_sorted[j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_time_delta_bins:
                break
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


def generate_single_peak_hashes(peaks):
    # used only for the single-peak vs paired-hash comparison in the report
    return [((f, 0, 0), t) for f, t in peaks]


def fingerprint_song(path, sr_target=11025, nperseg=2048, noverlap=1536,
                      amp_min_db=-60, fan_out=5):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    f, t, Sxx = compute_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)
    peaks = find_peaks(Sxx, amp_min_db=amp_min_db)
    hashes = generate_hashes(peaks, fan_out=fan_out)
    return hashes, (f, t, Sxx, peaks)


def build_database(song_paths, **kwargs):
    db = {}
    for path in song_paths:
        # handle both / and \ since this also needs to run on windows
        base = path.replace('\\', '/').split('/')[-1]
        name = base.rsplit('.', 1)[0] if '.' in base else base

        hashes, _ = fingerprint_song(path, **kwargs)
        for h, t_anchor in hashes:
            db.setdefault(h, []).append((name, t_anchor))
    return db


def match_query(query_hashes, db):
    """For each song, count hash matches at every offset and keep the best
    offset. A real match piles up at one offset, wrong songs scatter."""
    offset_counts = {}
    for h, t_query in query_hashes:
        if h not in db:
            continue
        for song_name, t_db in db[h]:
            offset = t_db - t_query
            offset_counts.setdefault(song_name, {})
            offset_counts[song_name][offset] = offset_counts[song_name].get(offset, 0) + 1

    results = []
    for song_name, offsets in offset_counts.items():
        best_offset = max(offsets, key=offsets.get)
        results.append((song_name, offsets[best_offset], best_offset))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def identify(query_path, db, **kwargs):
    hashes, extras = fingerprint_song(query_path, **kwargs)
    results = match_query(hashes, db)
    if not results:
        return None, [], extras
    return results[0][0], results, extras
