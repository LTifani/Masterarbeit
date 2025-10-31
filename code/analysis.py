# Ziel der ersten Analyse
    #-Wie groß ist der Datensatz wirklich?
    #-Sind die Dateien konsistent (Samplingrate, Bitrate, Dauer)?
    #-Wie unterscheiden sich die Normal- und Spontanaktivitäts-Signale im Zeit- und Frequenzbereich?



import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Pfade
base_path = "Dataset"
categories = ["Normal", "Spontanaktivität"]

# Ergebnisse speichern
summary = []

for category in categories:
    folder = os.path.join(base_path, category)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    durations = []
    sample_rates = []

    for f in files:
        file_path = os.path.join(folder, f)
        y, sr = librosa.load(file_path, sr=None)
        durations.append(len(y) / sr)
        sample_rates.append(sr)

    summary.append({
        "Kategorie": category,
        "Anzahl Dateien": len(files),
        "Durchschnittsdauer (s)": np.mean(durations),
        "Min Dauer (s)": np.min(durations),
        "Max Dauer (s)": np.max(durations),
        "Samplingrate": np.unique(sample_rates)
    })

# Ergebnisse als DataFrame anzeigen
df_summary = pd.DataFrame(summary)
print(df_summary)

# Beispielsignal visualisieren (Zeit & Frequenz)
example_path = os.path.join(base_path, "Normal", os.listdir(os.path.join(base_path, "Normal"))[0])
example_path = os.path.join(base_path, "Spontanaktivität", os.listdir(os.path.join(base_path, "Spontanaktivität"))[0])
y, sr = librosa.load(example_path, sr=None)

plt.figure(figsize=(12,4))
# plt.title("Beispielsignal (Normal) - Zeitbereich")
plt.title("Beispielsignal (Spontanaktivität) - Zeitbereich")

plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
plt.xlabel("Zeit (s)")
plt.ylabel("Amplitude")
plt.show()

# Frequenzanalyse
plt.figure(figsize=(10,4))
plt.specgram(y, Fs=sr, NFFT=1024, noverlap=512, cmap='viridis')
plt.title("Spektrogramm (Normal)")
plt.xlabel("Zeit (s)")
plt.ylabel("Frequenz (Hz)")
plt.colorbar(label="Leistung (dB)")
plt.show()
