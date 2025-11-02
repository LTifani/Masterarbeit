import math
import numpy as np
import pandas as pd
from icecream import ic
from typing import List, Tuple
from utils import *

# ---------------------------------------------------------
# Load anomalies from CSV
# ---------------------------------------------------------
def load_anomalies(csv_filepath: str) -> List[Tuple[float, float]]:
    """
    Loads anomaly timestamps from the specified CSV file.
    Assumes the relevant columns are 'start' and 'end' (in seconds).
    It sorts the anomalies by start time, which is required for
    the optimized labeling function.

    Args:
        csv_filepath: The path to your CSV file (e.g., "anomalies.csv").

    Returns:
        A sorted list of tuples, where each tuple is (start_time, end_time)
        of the anomaly.
    """
    try:
        df = pd.read_csv(csv_filepath)
        df["start"] = pd.to_numeric(df["Start"])
        df["end"] = pd.to_numeric(df["End"])

        if "start" not in df.columns or "end" not in df.columns:
            raise KeyError("Columns 'start' or 'end' not found.")

    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        print(
            f"Warning: Could not load {csv_filepath} with headers. "
            "Trying to load first two columns (0 and 1) without header."
        )
        try:
            df = pd.read_csv(csv_filepath, header=None, usecols=[0, 1])
            df.columns = ["start", "end"]
            df["start"] = pd.to_numeric(df["start"])
            df["end"] = pd.to_numeric(df["end"])
        except Exception as e:
            print(
                f"FATAL ERROR: Could not read CSV file {csv_filepath} "
                f"with or without headers. Error: {e}"
            )
            return []

    anomaly_list = list(zip(df["start"], df["end"]))
    anomaly_list.sort(key=lambda x: x[0])
    print(f"Successfully loaded and sorted {len(anomaly_list)} anomalies from {csv_filepath}.")
    return anomaly_list


# ---------------------------------------------------------
# Generate time segments
# ---------------------------------------------------------
def generate_segments(
    total_duration_sec: float,
    segment_duration_sec: float,
    overlap_percentage: float,
) -> List[Tuple[float, float]]:
    """
    Generates a list of (start, end) time segments for a given duration.
    """
    if not (0.0 <= overlap_percentage < 1.0):
        raise ValueError("overlap_percentage must be between 0.0 and 1.0")

    segments = []
    step_size_sec = segment_duration_sec * (1.0 - overlap_percentage)

    if step_size_sec <= 0:
        print("Warning: Step size is 0. Only one segment will be generated.")
        step_size_sec = segment_duration_sec

    current_start_sec = 0.0
    while current_start_sec + segment_duration_sec <= total_duration_sec:
        segments.append((current_start_sec, current_start_sec + segment_duration_sec))
        current_start_sec += step_size_sec

    print(f"Generated {len(segments)} segments.")
    return segments


# ---------------------------------------------------------
# Segment labeling
# ---------------------------------------------------------
def label_segments(
    audio_segments: List[Tuple[float, float]],
    anomalies: List[Tuple[float, float]],
    min_overlap_threshold: float,
) -> List[int]:
    """
    Labels audio segments based on overlap using an optimized O(1)
    range-based calculation.
    """
    num_segments = len(audio_segments)
    if num_segments == 0 or len(anomalies) == 0:
        return [0] * num_segments

    labels = [0] * num_segments
    segment_duration_sec = audio_segments[0][1] - audio_segments[0][0]

    if num_segments > 1:
        step_size_sec = audio_segments[1][0] - audio_segments[0][0]
    else:
        step_size_sec = segment_duration_sec

    if step_size_sec <= 0:
        print("Error: Segment step size is 0 or negative. Aborting.")
        return labels

    for anom_start, anom_end in anomalies:
        i_start = int(np.floor((anom_start - segment_duration_sec) / step_size_sec)) + 1
        i_start = max(0, i_start)

        i_end_check = int(np.ceil(anom_end / step_size_sec)) - 1
        i_end_check = min(num_segments - 1, i_end_check)

        for i in range(i_start, i_end_check + 1):
            if labels[i] == 1:
                continue

            seg_start, seg_end = audio_segments[i]
            segment_duration = seg_end - seg_start
            if segment_duration <= 0:
                continue

            intersection_start = max(seg_start, anom_start)
            intersection_end = min(seg_end, anom_end)
            intersection_duration = max(0, intersection_end - intersection_start)
            overlap_ratio = intersection_duration / segment_duration

            if overlap_ratio >= min_overlap_threshold:
                labels[i] = 1

    return labels



def label_segments_analytic(
    audio_segments: List[Tuple[float, float]],
    anomalies: List[Tuple[float, float]],
    min_overlap_threshold: float,
    eps: float = 1e-9
) -> List[int]:
    """
    Label segments using analytic mapping from anomaly time -> affected segment index range.

    Parameters
    ----------
    audio_segments : List[Tuple[float,float]]
        Sorted list of (start, end) segment times.
    anomalies : List[Tuple[float,float]]
        Sorted list of (start, end) anomaly times.
    min_overlap_threshold : float
        Fraction in [0,1] meaning minimal overlap ratio relative to segment duration.
    eps : float
        Tiny epsilon used to stabilize floating comparisons.

    Returns
    -------
    labels : List[int]
        0/1 labels in same order as audio_segments.
    """
    n_seg = len(audio_segments)
    if n_seg == 0 or len(anomalies) == 0:
        return [0] * n_seg

    # Precompute arrays for speed
    seg_starts = np.array([s for s, e in audio_segments], dtype=np.float64)
    seg_ends   = np.array([e for s, e in audio_segments], dtype=np.float64)
    seg_durations = seg_ends - seg_starts
    if np.any(seg_durations <= 0.0):
        raise ValueError("All segments must have positive duration.")

    # infer step size
    if n_seg > 1:
        step = seg_starts[1] - seg_starts[0]
    else:
        step = seg_durations[0]
    if step <= 0:
        raise ValueError("Segment step size must be positive.")

    labels = np.zeros(n_seg, dtype=np.uint8)

    for anom_start, anom_end in anomalies:
        # compute candidate index range
        # i_start = floor((anom_start - seg_duration) / step) + 1
        i_start = int(np.floor((anom_start - seg_durations[0]) / step)) + 1
        if i_start < 0:
            i_start = 0
        # i_end_check = ceil(anom_end / step) - 1
        i_end = int(np.ceil((anom_end) / step)) - 1
        if i_end >= n_seg:
            i_end = n_seg - 1
        if i_start > i_end:
            continue  # no candidate segments

        # vectorized slice for candidate segments
        idxs = np.arange(i_start, i_end + 1)
        # skip those already labeled
        unlabeled_mask = labels[idxs] == 0
        if not np.any(unlabeled_mask):
            continue
        idxs = idxs[unlabeled_mask]
        if idxs.size == 0:
            continue

        ss = seg_starts[idxs]
        se = seg_ends[idxs]
        sd = seg_durations[idxs]

        # compute intersection
        left = np.maximum(ss, anom_start)
        right = np.minimum(se, anom_end)
        inter = right - left

        ok_mask = inter > eps
        inter = np.where(inter > 0.0, inter, 0.0)

        if not np.any(ok_mask):
            continue

        inter = inter[ok_mask]
        sd_pos = sd[ok_mask]
        frac = inter / sd_pos

        hit_mask = frac >= (min_overlap_threshold - eps)
        if np.any(hit_mask):
            hit_idxs = idxs[ok_mask][hit_mask]
            labels[hit_idxs] = 1

    return labels.tolist()


# ---------------------------------------------------------
# Main script execution
# ---------------------------------------------------------
if __name__ == "__main__":
    CSV_FILEPATH = "./Dataset/second_labeling/filtered_BA0803901_segment_1_PSW_FB_labels.csv"
    AUDIO_DURATION_SECONDS = 20.0
    SEGMENT_DURATION_SECONDS = 0.01
    SEGMENT_OVERLAP_PERCENTAGE = 0.5
    MIN_ANOMALY_OVERLAP_PERCENTAGE = 20.0
    target_sample_rate = 11025
    

    min_overlap_threshold = MIN_ANOMALY_OVERLAP_PERCENTAGE / 100.0
    df = pd.read_csv(CSV_FILEPATH)
    df["Diff(ms)"] = (df["End"] - df["Start"]) * 1000
    df.to_csv("output/anomalies_with_durations.csv", index=True)
    ic(df.head(10))

    anomalie_zeiten = load_anomalies(CSV_FILEPATH)
    
    meine_audio_segmente = generate_segments(
        AUDIO_DURATION_SECONDS,
        SEGMENT_DURATION_SECONDS,
        SEGMENT_OVERLAP_PERCENTAGE,
    )

    if anomalie_zeiten and meine_audio_segmente:
        segment_labels = label_segments(
            meine_audio_segmente,
            anomalie_zeiten,
            min_overlap_threshold,
        )

        print("\n--- ERGEBNISSE DES LABELING ---")
        print(f"Mindestüberlappung eingestellt auf: {MIN_ANOMALY_OVERLAP_PERCENTAGE}%\n")

        total_anomalies = sum(segment_labels)
        df = pd.DataFrame({
            "Start": [seg[0] for seg in meine_audio_segmente],
            "End": [seg[1] for seg in meine_audio_segmente]
        })
        df["Start_sample"] = (df["Start"] * target_sample_rate).astype(int)
        df["End_sample"] = (df["End"] * target_sample_rate).astype(int)
        df["Label"] = segment_labels

        df.to_csv("output/segment_labels_output.csv", index=False)

        ic(df.head(20))

        print("\nLabeling abgeschlossen.")
        print(f"Gesamte Segmente: {len(segment_labels)}")
        print(f"Segmente mit Label 1 (Anomalie): {total_anomalies}")
    else:
        print(
            "\nSkript wurde nicht ausgeführt: "
            "Es konnten keine Anomalien oder Segmente geladen/generiert werden."
        )
