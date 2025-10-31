# preprocessing.py — Needle-EMG (WAV) → LSTM/TCN-ready .npz
# - Ordner: Normal / Spontanaktivität / Mixed
# - Patient-aware Splits (Overlap erlaubt), Train-Normal min. K IDs
# - Filter (Bandpass/Notch), Segmentierung, Augmentation
# - Normalisierung via cfg.normalization: 'per_segment' | 'per_file' | 'none'
# - Läuft ohne CLI: Config unten anpassen und ausführen.

import numpy as np
from scipy.io import wavfile
import os, re, json, random, math
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    # Ein-/Ausgabe
    input_dir: str = "Dataset"         
    out_dir: str = "preprocess"
    out_name: str = "emg_phase1_augment"       

    # Segmentierung
    target_sr: int = 11025
    window_sec: float = 1.0
    hop_sec: float = 1

    # Normalisierung: 'per_segment' | 'per_file' | 'none'
    normalization: str = "per_segment"

    # Filter
    apply_bandpass: bool = True
    bandpass: Tuple[float, float] = (20.0, 4500.0)
    filter_order: int = 4
    apply_notch: bool = True
    notch_freq: float = 50.0
    notch_q: float = 30.0

    # Datenquellen
    normal_name: str = "Normal"
    abnormal_name: str = "Spontanaktivität"
    mixed_name: str = "Mixed"

    # Mixed-Verwendung
    use_mixed_in_train: bool = False     
    include_mixed_in_test: bool = True   

    # Splits (Segment-basiert; Patienten-Overlap erlaubt)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    min_train_unique_patients: int = 5 

    # Augmentation (auf Trainings-Segmente angewandt)
    augment: bool = True
    aug_per_segment: int = 1
    p_noise: float = 0.5
    noise_std: Tuple[float, float] = (0.002, 0.01)
    p_shift: float = 0.5
    max_shift_sec: float = 0.1
    p_gain: float = 0.5
    gain_range: Tuple[float, float] = (0.9, 1.1)
    p_stretch: float = 0.3
    stretch_range: Tuple[float, float] = (0.95, 1.05)
    p_mask: float = 0.3
    mask_frac_range: Tuple[float, float] = (0.05, 0.1)

cfg = Config()

# ----------------------------
# I/O, DSP, Utils
# ----------------------------

def _read_wav(path: str, target_sr: int) -> np.ndarray:
    samplerate, data = wavfile.read(path)
    ic(data.shape[0]/samplerate, data.shape, data.dtype, samplerate, target_sr)
    # auf float [-1,1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    if samplerate != target_sr:
        g = math.gcd(samplerate, target_sr)
        up, down = target_sr // g, samplerate // g
        data = resample_poly(data, up, down).astype(np.float32)
    return data

def _butter_bandpass(sig: np.ndarray, sr: int, lo: float, hi: float, order: int) -> np.ndarray:
    ny = 0.5 * sr
    lo = max(1.0, lo) / ny
    hi = min(ny-1.0, hi) / ny
    if not (0 < lo < hi < 1):
        return sig
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, sig)

def _notch(sig: np.ndarray, sr: int, f0: float, q: float) -> np.ndarray:
    w0 = f0 / (sr/2.0)
    if w0 <= 0 or w0 >= 1:
        return sig
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.mean(x); sd = np.std(x)
    if sd < eps: sd = eps
    return (x - mu) / sd

def patient_id_from_path(path: str, folder_kind: str) -> str:
    """
    Normal/Spontan: führendes Buchstabenpräfix (z.B. 'LV260881' -> 'LV', 'müda2912786' -> 'müda')
    Mixed (auch Temp-Namen): führende Buchstaben bis zur ersten Ziffer; Fallback vorletztes '_' Token.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    if folder_kind in ("Normal", "Spontanaktivität"):
        m = re.match(r"^([A-Za-zÄÖÜäöü]+)", name)
        if m: return m.group(1)
        return name[:1]
    # Mixed
    m = re.match(r"^([A-Za-zÄÖÜäöü]+)", name)
    if m: return m.group(1)
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[-2]
    return parts[0]

def segment_audio_signal(
    signal: np.ndarray,
    sample_rate: int,
    window_duration_sec: float,
    step_duration_sec: float
) -> np.ndarray:
    """
    Split an audio signal into overlapping segments (frames).

    Args:
        signal (np.ndarray): Input audio signal (1D array).
        sample_rate (int): Sampling rate of the signal in Hz.
        window_duration_sec (float): Duration of each window in seconds.
        step_duration_sec (float): Time step (hop) between consecutive windows in seconds.

    Returns:
        np.ndarray: 2D array of shape (n_frames, frame_length_samples) containing segmented frames.
    """
    window_size = int(round(window_duration_sec * sample_rate)) # Window size in samples
    step_size = int(round(step_duration_sec * sample_rate)) # Step size in samples

    if window_size <= 0 or step_size <= 0 or len(signal) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    start_indices = np.arange(0, len(signal) - window_size + 1, step_size)
    frames = np.stack([signal[start:start + window_size] for start in start_indices]).astype(np.float32)

    return frames

# ----------------------------
# Augmentation 
# ----------------------------

rng = np.random.default_rng(cfg.seed)

def aug_add_noise(seg: np.ndarray, std_range: Tuple[float,float]) -> np.ndarray:
    std = rng.uniform(*std_range)
    return seg + rng.normal(0.0, std, size=seg.shape).astype(np.float32)

def aug_shift(seg: np.ndarray, sr: int, max_shift_sec: float) -> np.ndarray:
    max_samp = int(round(max_shift_sec * sr))
    if max_samp <= 0: return seg
    k = rng.integers(-max_samp, max_samp+1)
    if k == 0: return seg
    if k > 0:
        return np.concatenate([seg[k:], rng.normal(0, 1e-6, size=k).astype(np.float32)])
    else:
        k = -k
        return np.concatenate([rng.normal(0, 1e-6, size=k).astype(np.float32), seg[:-k]])

def aug_gain(seg: np.ndarray, g_range: Tuple[float,float]) -> np.ndarray:
    g = rng.uniform(*g_range)
    return (seg * g).astype(np.float32)

def aug_time_stretch(seg: np.ndarray, rate: float) -> np.ndarray:
    L = len(seg)
    new_L = max(8, int(round(L / rate)))
    stretched = resample_poly(seg, new_L, L).astype(np.float32)
    if len(stretched) > L:
        stretched = stretched[:L]
    elif len(stretched) < L:
        pad = np.zeros(L - len(stretched), dtype=np.float32)
        stretched = np.concatenate([stretched, pad])
    return stretched

def aug_mask(seg: np.ndarray, frac_range: Tuple[float,float]) -> np.ndarray:
    frac = rng.uniform(*frac_range)
    L = len(seg)
    m = max(1, int(round(frac * L)))
    i = rng.integers(0, max(1, L - m + 1))
    out = seg.copy(); out[i:i+m] = 0.0
    return out

def apply_augs(seg: np.ndarray, sr: int) -> np.ndarray:
    y = seg.copy()
    if rng.random() < cfg.p_noise:   y = aug_add_noise(y, cfg.noise_std)
    if rng.random() < cfg.p_shift:   y = aug_shift(y, sr, cfg.max_shift_sec)
    if rng.random() < cfg.p_gain:    y = aug_gain(y, cfg.gain_range)
    if rng.random() < cfg.p_stretch: y = aug_time_stretch(y, rng.uniform(*cfg.stretch_range))
    if rng.random() < cfg.p_mask:    y = aug_mask(y, cfg.mask_frac_range)
    return y

# ----------------------------
# File → Segmente 
# ----------------------------

def preprocess_file(path: str, folder_kind: str) -> Dict[str, np.ndarray]:
    x = _read_wav(path, cfg.target_sr)

    # Filter
    if cfg.apply_bandpass:
        x = _butter_bandpass(x, cfg.target_sr, cfg.bandpass[0], cfg.bandpass[1], cfg.filter_order)
    if cfg.apply_notch:
        x = _notch(x, cfg.target_sr, cfg.notch_freq, cfg.notch_q)

    # Normalisierung nach Modus
    norm_mode = cfg.normalization.lower()
    if norm_mode == "per_file":
        x = zscore(x)  # einmal pro komplette Datei

    # Segmentierung
    ic()
    segs = segment_audio_signal(x, cfg.target_sr, cfg.window_sec, cfg.hop_sec)
    if segs.size == 0:
        return {"segs": segs}

    if norm_mode == "per_segment":
        segs = np.stack([zscore(s) for s in segs]).astype(np.float32)
    elif norm_mode in ("per_file", "none"):
        segs = segs.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization mode: {cfg.normalization}")

    return {"segs": segs}

def collect_files(root: str) -> Dict[str, List[str]]:
    subdirs = [cfg.normal_name, cfg.abnormal_name, cfg.mixed_name]
    files = {k: [] for k in subdirs}
    for k in subdirs:
        d = os.path.join(root, k)
        if not os.path.isdir(d): 
            continue
        for dirpath, _, fnames in os.walk(d):
            for f in fnames:
                if f.lower().endswith((".wav", ".wave")):
                    files[k].append(os.path.join(dirpath, f))
    return files

# ----------------------------
# Split-Logik (Segment-basiert)
# ----------------------------

def split_segments_by_ratio(
    segments: List[np.ndarray],
    patient_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    min_unique_train_patients: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split segment data into train, validation, and test sets.
    Ensures that the training set contains at least a minimum number of unique patient IDs.

    Args:
        segments (List[np.ndarray]): List of segment arrays.
        patient_ids (List[str]): List of patient IDs, one per segment.
        train_ratio (float): Fraction of data for training.
        val_ratio (float): Fraction of data for validation.
        min_unique_train_patients (int): Minimum number of unique patients in training set.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays for (X_train, X_val, X_test).
    """
    n_segments = len(segments)
    indices = np.arange(n_segments)
    rng.shuffle(indices)

    n_train = int(round(train_ratio * n_segments))
    n_val = int(round(val_ratio * n_segments))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Ensure sufficient unique patient IDs in training
    train_pids = np.array(patient_ids)[train_indices]
    unique_train_pids = set(train_pids)

    if len(unique_train_pids) < min_unique_train_patients:
        all_unique_pids = set(patient_ids)
        missing_pids = list(all_unique_pids - unique_train_pids)
        rng.shuffle(missing_pids)

        for pid in missing_pids:
            candidate_indices = np.flatnonzero(np.array(patient_ids) == pid)
            if len(candidate_indices) == 0:
                continue
            selected_idx = rng.choice(candidate_indices)
            if selected_idx not in train_indices:
                train_indices = np.concatenate((train_indices, [selected_idx]))
                unique_train_pids.add(pid)
                if len(unique_train_pids) >= min_unique_train_patients:
                    break

    def safe_stack(idxs: np.ndarray) -> np.ndarray:
        return np.stack([segments[i] for i in idxs]) if len(idxs) else np.empty((0, *segments[0].shape))

    X_train = safe_stack(train_indices)
    X_val = safe_stack(val_indices)
    X_test = safe_stack(test_indices)

    return X_train, X_val, X_test

# ----------------------------
# Pipeline
# ----------------------------

def run():
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    files = collect_files(cfg.input_dir)

    segs_normal, pids_normal = [], []
    segs_abn,    pids_abn    = [], []
    segs_mixed,  pids_mixed  = [], []

    def process_group(file_list: List[str], folder_kind: str, segs_acc: List[np.ndarray], pids_acc: List[str]):
        for fp in file_list:
            pid = patient_id_from_path(fp, folder_kind)
            ic(fp, pid, folder_kind)
            return
            out = preprocess_file(fp, folder_kind)
            S = out["segs"]
            if S.size == 0:
                continue
            for s in S:
                segs_acc.append(s.astype(np.float32))
                pids_acc.append(pid)

    process_group(files.get(cfg.normal_name, []), cfg.normal_name, segs_normal, pids_normal)
    return
    process_group(files.get(cfg.abnormal_name, []), cfg.abnormal_name, segs_abn, pids_abn)
    process_group(files.get(cfg.mixed_name, []), cfg.mixed_name, segs_mixed, pids_mixed)

    # Splits pro Quelle
    def to_3d(x: np.ndarray) -> np.ndarray:  # (N, T) -> (N, T, 1)
        return x[..., None].astype(np.float32)

    Xtr_n = Xv_n = Xte_n = np.empty((0, int(cfg.window_sec*cfg.target_sr)), dtype=np.float32)
    Xtr_a = Xv_a = Xte_a = np.empty_like(Xtr_n)
    Xtr_m = Xv_m = Xte_m = np.empty_like(Xtr_n)

    if len(segs_normal):
        Xtr_n, Xv_n, Xte_n = split_segments_by_ratio(segs_normal, pids_normal)
    if len(segs_abn):
        Xtr_a, Xv_a, Xte_a = split_segments_by_ratio(segs_abn, pids_abn)
    if len(segs_mixed):
        Xtr_m, Xv_m, Xte_m = split_segments_by_ratio(segs_mixed, pids_mixed)

    # Trainset: normal + optional mixed
    X_train = Xtr_n
    if cfg.use_mixed_in_train and len(Xtr_m):
        X_train = np.concatenate([X_train, Xtr_m], axis=0)

    # Augmentation nur auf Train
    if cfg.augment and len(X_train):
        sr = cfg.target_sr
        L = X_train.shape[1]
        aug_list = []
        for s in X_train:
            for _ in range(cfg.aug_per_segment):
                y = apply_augs(s, sr)
                # nur wenn per_segment-Norm aktiv, nach Augment erneut z-Norm
                if cfg.normalization.lower() == "per_segment":
                    y = zscore(y)
                aug_list.append(y.reshape(1, L))
        if aug_list:
            X_aug = np.concatenate(aug_list, axis=0).astype(np.float32)
            X_train = np.concatenate([X_train, X_aug], axis=0)
        # Shuffle
        idx = np.arange(len(X_train)); rng.shuffle(idx); X_train = X_train[idx]

    # Val/Test
    X_val_normal, X_test_normal = Xv_n, Xte_n
    X_val_abn,    X_test_abn    = Xv_a, Xte_a
    if cfg.include_mixed_in_test and (len(Xv_m) or len(Xte_m)):
        X_val_mixed, X_test_mixed = Xv_m, Xte_m
    else:
        X_val_mixed = np.empty((0, X_train.shape[1]), dtype=np.float32)
        X_test_mixed = np.empty_like(X_val_mixed)

    # Zusätzlich: abn + mixed gemerged (für „nicht-normal“-Evaluation)
    X_val_abn_plus_mixed  = X_val_abn if len(X_val_mixed)==0 else np.concatenate([X_val_abn, X_val_mixed], axis=0)
    X_test_abn_plus_mixed = X_test_abn if len(X_test_mixed)==0 else np.concatenate([X_test_abn, X_test_mixed], axis=0)

    # In (N, T, 1) bringen
    X_train = to_3d(X_train)
    X_val_normal  = to_3d(X_val_normal)
    X_test_normal = to_3d(X_test_normal)
    X_val_abn     = to_3d(X_val_abn)
    X_test_abn    = to_3d(X_test_abn)
    X_val_mixed   = to_3d(X_val_mixed)
    X_test_mixed  = to_3d(X_test_mixed)
    X_val_abn_plus_mixed  = to_3d(X_val_abn_plus_mixed)
    X_test_abn_plus_mixed = to_3d(X_test_abn_plus_mixed)

    # Meta
    meta = {
        "config": asdict(cfg),
        "counts": {
            "train": int(len(X_train)),
            "val_normal": int(len(X_val_normal)),
            "val_abnormal": int(len(X_val_abn)),
            "val_mixed": int(len(X_val_mixed)),
            "test_normal": int(len(X_test_normal)),
            "test_abnormal": int(len(X_test_abn)),
            "test_mixed": int(len(X_test_mixed)),
        },
        "normalization": cfg.normalization,
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"{cfg.out_name}.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train,
        X_val_normal=X_val_normal,
        X_val_abnormal=X_val_abn,
        X_val_mixed=X_val_mixed,
        X_test_normal=X_test_normal,
        X_test_abnormal=X_test_abn,
        X_test_mixed=X_test_mixed,
        X_val_abn_plus_mixed=X_val_abn_plus_mixed,
        X_test_abn_plus_mixed=X_test_abn_plus_mixed,
        meta=np.array(json.dumps(meta), dtype=object),
    )
    print("Saved:", out_path)
    print(json.dumps(meta, indent=2, ensure_ascii=False))

# =========================
# Loader (für Training)
# =========================
def load_npz_for_training(npz_path: str):
    """
    Lädt die vom Preprocessing erzeugte .npz-Datei.
    Gibt zurück:
      X_train  : np.ndarray (N, T, 1)
      splits   : Dict[str, Dict[str, np.ndarray]]
                 -> splits['val']['normal'], ['abnormal'], ['mixed'], ['abn_plus_mixed']
                 -> splits['test']['normal'], ...
      meta     : Dict (mit Config und Counts)
    """
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"].item()))

    splits = {
        "val": {
            "normal":          d["X_val_normal"],
            "abnormal":        d["X_val_abnormal"],
            "mixed":           d["X_val_mixed"],
            "abn_plus_mixed":  d["X_val_abn_plus_mixed"],
        },
        "test": {
            "normal":          d["X_test_normal"],
            "abnormal":        d["X_test_abnormal"],
            "mixed":           d["X_test_mixed"],
            "abn_plus_mixed":  d["X_test_abn_plus_mixed"],
        }
    }
    return d["X_train"], splits, meta


if __name__ == "__main__":
    logfile = open("output/console.log", "w")
    from icecream import ic
    def dual_output(s):
        print(s)                # Konsole
        print(s, file=logfile)  # Datei
    ic.configureOutput(prefix="DEBUG | ", includeContext=True, outputFunction=dual_output)
    # Varianten testen: cfg.normalization = "per_segment" | "per_file" | "none"
    run()
    
    example_path = "Dataset/Spontanaktivität/BA0803901.wav"
    results = preprocess_file(example_path, "Spontanaktivität")
    ic(results)
    
