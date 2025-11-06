import os
import glob
import math
import numpy as np
import soundfile as sf
import csv

# ================================
# EINSTELLUNGEN
# ================================
NORMAL_DIR = r"Dataset/Normal"             # <-- Pfad zu deinem 'normal'-Ordner
OUT_DIR = os.path.join("Dataset", "Label_normal")
WINDOW_MS = 10.0                        # Segmentlänge (ms)
OVERLAP = 0.5                           # Anteil (0.5 = 50% Überlappung)
COVER_LAST = True                       # Letztes (ggf. kurzes) Segment mitnehmen
LABEL_VALUE = 0                         # Label für normale Segmente

# ================================
# HILFSFUNKTIONEN
# ================================
def read_wav_mono(path: str):
    """Liest WAV als float32 Mono + Samplingrate."""
    data, sr = sf.read(path, always_2d=True)
    x = data.astype(np.float32).mean(axis=1)
    return x, int(sr)

def seconds_to_samples(sec: float, sr: int) -> int:
    return int(round(sec * sr))

def make_segments(n_samples: int, sr: int, window_ms: float, overlap: float, cover_last: bool):
    """Berechnet (start,end)-Paare in Samples für 10ms-Fenster."""
    assert 0.0 <= overlap < 1.0
    win = seconds_to_samples(window_ms / 1000.0, sr)
    hop = max(1, int(round(win * (1.0 - overlap))))
    starts = list(range(0, max(0, n_samples - win + 1), hop))
    ends = [s + win for s in starts]
    if cover_last and (not starts or ends[-1] < n_samples):
        s = starts[-1] + hop if starts else 0
        if s < n_samples:
            e = min(s + win, n_samples)
            if e > s:
                starts.append(s)
                ends.append(e)
    return list(zip(starts, ends))

def write_csv(path: str, segments, sr: int, label_value: int):
    """Schreibt CSV im Label-Format: Start, End, Label, Start Sample, End Sample."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start", "End", "Label", "Start Sample", "End Sample"])
        for (s0, s1) in segments:
            w.writerow([s0 / sr, s1 / sr, label_value, s0, s1])

# ================================
# HAUPTLAUF
# ================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(NORMAL_DIR, "**", "*.wav"), recursive=True))
    if not wavs:
        print(f"❌ Keine WAV-Dateien gefunden in: {NORMAL_DIR}")
        return

    print(f"Gefundene WAVs: {len(wavs)}")
    print(f"→ Erstelle Labels im Ordner: {OUT_DIR}")
    print(f"Fenster: {WINDOW_MS} ms, Überlappung: {OVERLAP*100:.0f}%, cover_last={COVER_LAST}\n")

    for i, wav_path in enumerate(wavs, 1):
        try:
            x, sr = read_wav_mono(wav_path)
            segs = make_segments(len(x), sr, WINDOW_MS, OVERLAP, COVER_LAST)

            base = os.path.splitext(os.path.basename(wav_path))[0]
            csv_name = f"{base}_labels.csv"
            csv_path = os.path.join(OUT_DIR, csv_name)

            write_csv(csv_path, segs, sr, LABEL_VALUE)
            print(f"[{i:03d}/{len(wavs):03d}] {base}: {len(segs)} Segmente gespeichert.")
        except Exception as e:
            print(f"⚠️ Fehler bei {wav_path}: {e}")

    print("\n✅ Fertig! Alle CSVs im Ordner:", OUT_DIR)

# ================================
# START
# ================================
if __name__ == "__main__":
    main()
