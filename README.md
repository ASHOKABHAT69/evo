# Digital Audio Notary

A Flask-based forensic toolkit that performs a **3-layer trust audit** on uploaded audio and returns a final binary verdict: **Human** or **Not Human**.

## Important limitation

Perfect differentiation between live human voice and deepfake/synthetic voice is **not possible** in a universal sense. This tool provides a probabilistic, explainable forensic assessment (decision support), not absolute proof.

## What it does

1. **Layer 1 — Metadata & Structural Integrity (Rule-based)**
   - Detects bitrate inconsistencies and duration/size anomalies.
   - Searches metadata tags for editing software markers.
   - Detects abrupt digital silence runs (near zero-amplitude segments).

2. **Layer 2 — Biological Feature Extraction**
   - Computes and visualizes **MFCC** and **log spectrogram**.
   - Computes **pitch jitter** and **shimmer** as biological-noise cues.

3. **Layer 3 — Authenticity Score (0-100%)**
   - Combines Layer 1 penalties, Layer 2 variability checks, and a Hugging Face model signal.
   - Produces:
     - **Final Binary Verdict:** Human / Not Human
     - **Forensic Note:** e.g., `Authentic Voice, but Potentially Edited`

## Hugging Face model

By default, the app uses:

`DunnBC22/wav2vec2-base-finetuned-gtzan-music-speech`

You can swap it with any audio-classification model by setting:

```bash
export HF_MODEL_ID="your/model-id"
```

If your model has labels such as `human/real/bonafide` and `fake/synthetic/spoof`, the mapping is stronger for authenticity use-cases.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8000`.
