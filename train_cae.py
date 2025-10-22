# 1D Convolutional Autoencoder for needle-EMG anomaly detection
# - trains on "normal" segments only
# - optional multi-view loss with log-mel spectrogram
# - computes val threshold via ROC/F1 (if y given) or 95th percentile of train errors
# - saves model + metrics/plots to ./outputs


import os, sys, json, importlib
import numpy as np
import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
NPZ_PATH = "Dataset/emg_phase1_autoencoder.npz"


#  Preprocessing  +   NPZ laden

print("Preprocessing ...")
import preprocessing

cfg = preprocessing.Config(
    input_dir="Dataset",
    output_dir="Dataset"
)
preprocessing.preprocess_dataset(cfg)
print("Preprocessing abgeschlossen, lade NPZ...")

data = np.load(NPZ_PATH, allow_pickle=True)
print(f"Geladene Keys: {list(data.keys())}")

X_train = data["X_train"]
y_train = data["y_train"] if "y_train" in data else None
X_val   = data["X_val"]   if "X_val"   in data else None
y_val   = data["y_val"]   if "y_val"   in data else None
sr      = int(data["sr"]) if "sr" in data else None

print(f"Train: {X_train.shape}, Val: {None if X_val is None else X_val.shape}, SR={sr}")


# Autoencoder definieren

def build_cae(input_len, base=32, k=7, wd=1e-4):
    #---------Encoder ----------
    reg = keras.regularizers.l2(wd)
    inp = keras.Input(shape=(input_len, 1))
    x = layers.Conv1D(base, k, padding="same", kernel_regularizer=reg)(inp)
    x = layers.ReLU()(x); x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(base*2, k, padding="same", kernel_regularizer=reg)(x)
    x = layers.ReLU()(x); x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(base*4, k, padding="same", kernel_regularizer=reg)(x)
    x = layers.ReLU()(x); x = layers.MaxPool1D(2)(x)

    z = x  # latent space

    #---------Decoder ----------
    x = layers.UpSampling1D(2)(z)
    x = layers.Conv1D(base*2, k, padding="same")(x); x = layers.ReLU()(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(base, k, padding="same")(x); x = layers.ReLU()(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(1, k, padding="same", activation="linear")(x)
    return keras.Model(inp, out, name="cae_1d")

def recon_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2])
    return tf.reduce_mean(0.7*mse + 0.3*mae)

def recon_errors(model, X):
    X = np.expand_dims(X.astype(np.float32), -1)
    Y = model.predict(X, verbose=0)
    mse = np.mean((X - Y)**2, axis=(1,2))
    mae = np.mean(np.abs(X - Y), axis=(1,2))
    return 0.7*mse + 0.3*mae

def pick_threshold(err_train, err_val=None, y_val=None):
    if y_val is not None and err_val is not None:
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(y_val, err_val)
        f1 = 2*p*r/(p+r+1e-8)
        i = np.nanargmax(f1)
        return float(t[i])
    return float(np.percentile(err_train, 95))


#  Training

T = min(X_train.shape[1], X_val.shape[1]) if X_val is not None else X_train.shape[1]
X_train = X_train[:, :T]
if X_val is not None: X_val = X_val[:, :T]

# Nur normale Segmente
if y_train is not None:
    X_train = X_train[y_train == 0]

def make_ds(X, shuffle=False):
    X = np.expand_dims(X.astype(np.float32), -1)
    ds = tf.data.Dataset.from_tensor_slices((X, X))
    if shuffle: ds = ds.shuffle(4096)
    return ds.batch(128).prefetch(tf.data.AUTOTUNE)

dstr = make_ds(X_train, shuffle=True)
dsva = make_ds(X_val, shuffle=False) if X_val is not None else None

model = build_cae(T)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=recon_loss)

cbs = [
    keras.callbacks.ModelCheckpoint(os.path.join(OUTDIR, "cae_emg.keras"),
                                    save_best_only=True, monitor="val_loss", mode="min"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
]

print("Starte Training ...")
model.fit(dstr, validation_data=dsva, epochs=40, callbacks=cbs, verbose=1)


# Evaluation & Speichern

e_tr = recon_errors(model, X_train)
e_va = recon_errors(model, X_val) if X_val is not None else None
thr = pick_threshold(e_tr, e_va, y_val if y_val is not None else None)

results = {"threshold": float(thr)}
if y_val is not None:
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    yhat = (e_va >= thr).astype(int)
    results["val_report"] = classification_report(y_val, yhat, output_dict=True)
    results["val_auroc"]  = float(roc_auc_score(y_val, e_va))
    results["val_auprc"]  = float(average_precision_score(y_val, e_va))
    print(f"Val AUROC={results['val_auroc']:.3f}, AUPRC={results['val_auprc']:.3f}")

np.save(os.path.join(OUTDIR, "errors_train.npy"), e_tr)
if e_va is not None:
    np.save(os.path.join(OUTDIR, "errors_val.npy"), e_va)
with open(os.path.join(OUTDIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n ENDE")
