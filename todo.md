1. Setup & Projekt-Vorbereitung

 Projektstruktur anlegen (Ordner für Data, Notebooks, Models, Logs, Ergebnisse)

 Python-Umgebung einrichten (z. B. conda / venv) mit Bibliotheken: numpy, scipy, torch (oder tensorflow), librosa / torchaudio, matplotlib, sklearn

 Versionskontrolle (Git) initialisieren, erste Commit

2. Datenimport & Grundüberprüfung

 WAV-Dateien importieren, Metadaten (Samplingrate, Kanäle, Dauer) prüfen

 Konsistenz prüfen: sind alle Clips Mono, gleiche Samplingrate?

 Ausschluss schlechter Aufnahmen (zu laut, Verzerrungen, Störungen)

3. Preprocessing (Signalaufbereitung)

 Bandpass-Filterung (z. B. 20–5 000 Hz)

 Notch-Filter gegen Netzbrummen (z. B. 50 Hz und ggf. Harmonische)

 DC-Offset entfernen (Mittelwert subtrahieren)

 Falls nötig: Signal glätten oder rauschen (leichte Glättungsfilter)

 Signalnormierung (z. B. RMS-Norm, z-Score, Clip-Weise)

 (Optional) Artefakte erkennen und entfernen (Stopp, Knacksen)

 4. Segmentierung / Fensterung / Datenaufbau

 Fenstergröße festlegen (z. B. 256 ms, 512 ms)

 Überlappung definieren (z. B. 50 %)

 Fenster extrahieren: Rohsignal → Array mit Form:

Für 1D-CAE: [N_windows, 1, T_samples]

Für LSTM-AE: [N_windows, SeqLen, 1]

 (Optional) 2D-Darstellung erzeugen (Log-Mel, STFT) für alternative Modelle

 Split in Trainings / Validierungs / Test (patientenweise) sicherstellen

5. Datenaugmentation (optional / moderat)

 Gain Jitter (±3 dB)

 Zeit-Jitter (±5 % Länge)

 Hinzufügen von Rauschen (white / pink) bei hoher SNR

 Kein aggressives Deforming, damit „Normal“ stabil bleibt