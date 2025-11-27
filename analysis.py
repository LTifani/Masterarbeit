"""
Analysis script for EMG dataset.

Goals:
- Dataset size overview.
- Consistency check (sampling rate, bitrate, duration).
- Signal comparison in time and frequency domains (normal vs. spontaneous activity).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def collect_dataset_summary(base_path: Path, categories: List[str]) -> pd.DataFrame:
    """
    Collect summary statistics for each category in the dataset.

    Args:
        base_path: Root directory of the dataset.
        categories: List of category names (e.g., ["Normal", "Spontanaktivität"]).

    Returns:
        DataFrame with summary statistics.
    """
    summary_data = []

    for category in categories:
        category_path = base_path / category
        if not category_path.exists():
            continue

        wav_files = list(category_path.glob("*.wav"))
        durations = []
        sample_rates = []

        for file_path in wav_files:
            audio, sr = librosa.load(file_path, sr=None)
            durations.append(len(audio) / sr)
            sample_rates.append(sr)

        summary_data.append({
            "Category": category,
            "Number of Files": len(wav_files),
            "Average Duration (s)": np.mean(durations),
            "Min Duration (s)": np.min(durations),
            "Max Duration (s)": np.max(durations),
            "Sampling Rate": np.unique(sample_rates, return_inverse=True) if sample_rates else None
        })

    return pd.DataFrame(summary_data)


def visualize_example_signal(
    base_path: Path, category: str, sr_target: Optional[int] = None
) -> None:
    """
    Visualize an example signal in time and frequency domains.

    Args:
        base_path: Root directory of the dataset.
        category: Category to select example from.
        sr_target: Target sampling rate (optional).
    """
    category_path = base_path / category
    if not category_path.exists():
        print(f"Category path {category_path} does not exist.")
        return

    example_file = next(category_path.glob("*.wav"), None)
    if example_file is None:
        print(f"No WAV files found in {category_path}.")
        return

    audio, sr = librosa.load(example_file, sr=sr_target)

    # Time domain plot
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio)
    plt.title(f"Example Signal ({category}) - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # Frequency domain: Spectrogram
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title(f"Spectrogram ({category})")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.show()


def main() -> None:
    """Run the dataset analysis."""
    base_path = Path("../Dataset")
    categories = ["Normal", "Spontanaktivität"]

    # Print summary
    summary_df = collect_dataset_summary(base_path, categories)
    print(summary_df)

    # Visualize examples
    visualize_example_signal(base_path, "Spontanaktivität")


if __name__ == "__main__":
    main()