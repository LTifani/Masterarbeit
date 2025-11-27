"""
Utility Functions for EMG Signal Processing

Common helper functions used across the project.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_segment(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize audio segment to zero mean and unit standard deviation.
    
    Args:
        segment: Input audio segment array
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Normalized segment with mean ≈ 0 and std ≈ 1
    """
    mean = np.mean(segment)
    std = np.std(segment)
    
    if std < eps:
        return segment - mean
    
    normalized = (segment - mean) / std
    return normalized.astype(np.float32)


def load_anomaly_annotations(csv_filepath: str) -> List[Tuple[float, float]]:
    """
    Load anomaly time ranges from CSV file.
    
    Args:
        csv_filepath: Path to CSV file containing anomaly annotations
        
    Returns:
        List of tuples containing (start_time, end_time) for each anomaly
    """
    try:
        df = pd.read_csv(csv_filepath)
        df["Start"] = pd.to_numeric(df["Start"], errors='coerce')
        df["End"] = pd.to_numeric(df["End"], errors='coerce')
        
        if "Start" not in df.columns or "End" not in df.columns:
            raise KeyError("Columns 'Start' or 'End' not found.")
        
        df = df.dropna(subset=["Start", "End"])
        
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as e:
        logger.debug(f"Failed to read {csv_filepath} with header: {e}")
        try:
            df = pd.read_csv(csv_filepath, header=None, usecols=[0, 1])
            df.columns = ["Start", "End"]
            df["Start"] = pd.to_numeric(df["Start"], errors='coerce')
            df["End"] = pd.to_numeric(df["End"], errors='coerce')
            df = df.dropna(subset=["Start", "End"])
        except Exception as e:
            logger.error(f"Could not read CSV {csv_filepath}: {e}")
            return []
    
    df.sort_values('Start', inplace=True)
    anomaly_list = df[["Start", "End"]].values.tolist()
    
    return anomaly_list


def generate_audio_segments(total_duration_sec: float, 
                           segment_duration_sec: float, 
                           overlap_percentage: float) -> List[Tuple[float, float]]:
    """
    Generate overlapping time segments for audio processing.
    
    Args:
        total_duration_sec: Total duration of audio in seconds
        segment_duration_sec: Duration of each segment in seconds
        overlap_percentage: Overlap between segments as fraction (0.0 to 1.0)
        
    Returns:
        List of tuples containing (start_time, end_time) for each segment
    """
    if not (0.0 <= overlap_percentage < 1.0):
        raise ValueError(f"overlap_percentage must be in [0.0, 1.0), got {overlap_percentage}")
    
    segments = []
    step_size = segment_duration_sec * (1.0 - overlap_percentage)
    
    if step_size <= 0:
        step_size = segment_duration_sec
    
    start_time = 0.0
    while start_time + segment_duration_sec <= total_duration_sec:
        segments.append((start_time, start_time + segment_duration_sec))
        start_time += step_size
    
    return segments


def compute_segment_labels(audio_segments: List[Tuple[float, float]], 
                          anomalies: List[Tuple[float, float]], 
                          min_overlap_threshold: float,
                          eps: float = 1e-9) -> List[int]:
    """
    Label audio segments as anomalous (1) or normal (0) based on overlap.
    
    Args:
        audio_segments: List of (start, end) tuples for audio segments
        anomalies: List of (start, end) tuples for anomaly regions
        min_overlap_threshold: Minimum overlap fraction to label as anomaly
        eps: Small epsilon for numerical stability
        
    Returns:
        List of binary labels (0=normal, 1=anomaly) for each segment
    """
    num_segments = len(audio_segments)
    
    if num_segments == 0 or len(anomalies) == 0:
        return [0] * num_segments
    
    segment_starts = np.array([s for s, e in audio_segments], dtype=np.float64)
    segment_ends = np.array([e for s, e in audio_segments], dtype=np.float64)
    segment_durations = segment_ends - segment_starts
    
    labels = np.zeros(num_segments, dtype=np.uint8)
    step_size = segment_starts[1] - segment_starts[0] if num_segments > 1 else segment_durations[0]
    
    for anomaly_start, anomaly_end in anomalies:
        idx_start = max(0, int(np.floor((anomaly_start - segment_durations[0]) / step_size)) + 1)
        idx_end = min(num_segments - 1, int(np.ceil(anomaly_end / step_size)) - 1)
        
        if idx_start > idx_end:
            continue
        
        indices = np.arange(idx_start, idx_end + 1)
        unlabeled_mask = labels[indices] == 0
        indices = indices[unlabeled_mask]
        
        if indices.size == 0:
            continue
        
        seg_starts = segment_starts[indices]
        seg_ends = segment_ends[indices]
        seg_durs = segment_durations[indices]
        
        intersection_left = np.maximum(seg_starts, anomaly_start)
        intersection_right = np.minimum(seg_ends, anomaly_end)
        intersection_length = np.where(
            intersection_right - intersection_left > eps,
            intersection_right - intersection_left,
            0.0
        )
        
        valid_overlap_mask = intersection_length > eps
        overlap_fractions = np.zeros_like(intersection_length)
        
        if np.any(valid_overlap_mask):
            overlap_fractions[valid_overlap_mask] = (
                intersection_length[valid_overlap_mask] / seg_durs[valid_overlap_mask]
            )
            anomaly_mask = overlap_fractions >= (min_overlap_threshold - eps)
            labels[indices[anomaly_mask]] = 1
    
    return labels.tolist()


def setup_logging(log_dir: str = "output", 
                 log_filename: str = "application.log",
                 level: str = "INFO"):
    """
    Configure logging for the application.
    
    Args:
        log_dir: Directory for log files
        log_filename: Name of the log file
        level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename), mode="w"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# --- UTILITY-FUNKTION ZUM GLÄTTEN DES CONFIG DICT ---
def _flatten_config_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Glättet ein verschachteltes Konfigurations-Dictionary für MLflow."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Dict):
            items.extend(_flatten_config_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) or isinstance(v, tuple):
            items.append((new_key, str(v))) # Listen/Tupel als String loggen
        else:
            items.append((new_key, v))
    return dict(items)