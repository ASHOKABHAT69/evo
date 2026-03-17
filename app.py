import base64
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mutagen
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "DunnBC22/wav2vec2-base-finetuned-gtzan-music-speech")


@dataclass
class Layer1Result:
    bitrate_inconsistent: bool
    duration_size_mismatch: bool
    editing_software_detected: list[str]
    digital_silence_events: int
    notes: list[str]


@dataclass
class Layer2Result:
    jitter: float
    shimmer: float
    jitter_flag: bool
    shimmer_flag: bool
    mfcc_image_b64: str
    spectrogram_image_b64: str


@dataclass
class ModelResult:
    human_probability: float
    top_label: str
    raw_scores: list[dict[str, Any]]


class HFHumanDetector:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipe = pipeline("audio-classification", model=model_id, top_k=None)

    @staticmethod
    def _label_to_human_score(label: str) -> float | None:
        l = label.lower()
        if any(k in l for k in ["human", "real", "bonafide", "bona fide", "live", "speech"]):
            return 1.0
        if any(k in l for k in ["fake", "spoof", "synthetic", "ai", "tts", "cloned"]):
            return 0.0
        return None

    def predict_human_probability(self, audio_path: str) -> ModelResult:
        scores = self.pipe(audio_path)
        if isinstance(scores, dict):
            scores = [scores]

        weighted = []
        known_mass = 0.0
        for row in scores:
            mapped = self._label_to_human_score(row["label"])
            score = float(row["score"])
            if mapped is None:
                continue
            weighted.append(mapped * score)
            known_mass += score

        if known_mass > 0:
            human_probability = float(np.clip(sum(weighted) / known_mass, 0, 1))
        else:
            # Fallback when model labels are not authenticity-oriented.
            top_conf = float(max(row["score"] for row in scores)) if scores else 0.5
            human_probability = float(np.clip(0.5 + (top_conf - 0.5) * 0.1, 0, 1))

        top = max(scores, key=lambda x: x["score"]) if scores else {"label": "unknown", "score": 0.0}
        return ModelResult(
            human_probability=human_probability,
            top_label=f"{top['label']} ({top['score']:.3f})",
            raw_scores=scores,
        )


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyze_layer1(audio_path: str) -> Layer1Result:
    notes: list[str] = []
    bitrate_inconsistent = False
    duration_size_mismatch = False
    editing_tags = []

    audio_meta = mutagen.File(audio_path)
    file_size_bytes = Path(audio_path).stat().st_size
    duration = getattr(getattr(audio_meta, "info", None), "length", None)
    reported_bitrate = getattr(getattr(audio_meta, "info", None), "bitrate", None)

    if duration and duration > 0:
        estimated_bitrate = (file_size_bytes * 8) / duration
        if reported_bitrate and reported_bitrate > 0:
            ratio = abs(estimated_bitrate - reported_bitrate) / reported_bitrate
            bitrate_inconsistent = ratio > 0.35
            if bitrate_inconsistent:
                notes.append(
                    f"Bitrate mismatch: estimated {estimated_bitrate:.0f}bps vs header {reported_bitrate:.0f}bps"
                )
        duration_size_mismatch = estimated_bitrate < 8_000 or estimated_bitrate > 1_500_000
        if duration_size_mismatch:
            notes.append(f"Duration-size ratio is unusual (estimated bitrate {estimated_bitrate:.0f}bps).")

    if getattr(audio_meta, "tags", None):
        tag_blob = str(audio_meta.tags).lower()
        markers = ["audacity", "adobe", "premiere", "ffmpeg", "da vinci", "garageband", "logic pro"]
        for marker in markers:
            if marker in tag_blob:
                editing_tags.append(marker)
        if editing_tags:
            notes.append(f"Editing software markers found: {', '.join(sorted(set(editing_tags)))}")

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    near_zero = np.abs(y) < 1e-7
    min_len = int(0.05 * sr) if sr else 1
    silence_events = 0
    run = 0
    for z in near_zero:
        run = run + 1 if z else 0
        if run == min_len:
            silence_events += 1

    if silence_events > 3:
        notes.append(f"Detected {silence_events} abrupt digital-silence segments (>=50ms near-zero).")

    return Layer1Result(
        bitrate_inconsistent=bitrate_inconsistent,
        duration_size_mismatch=duration_size_mismatch,
        editing_software_detected=sorted(set(editing_tags)),
        digital_silence_events=silence_events,
        notes=notes,
    )


def analyze_layer2(audio_path: str) -> Layer2Result:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0 = f0[~np.isnan(f0)]
    if len(f0) > 2 and np.mean(f0) > 0:
        jitter = float(np.mean(np.abs(np.diff(f0))) / np.mean(f0))
    else:
        jitter = 0.0

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    if len(rms) > 2 and np.mean(rms) > 0:
        shimmer = float(np.mean(np.abs(np.diff(rms))) / np.mean(rms))
    else:
        shimmer = 0.0

    jitter_flag = jitter < 0.002
    shimmer_flag = shimmer < 0.015

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    img1 = librosa.display.specshow(mfcc, x_axis="time", ax=ax1)
    ax1.set_title("MFCC")
    fig1.colorbar(img1, ax=ax1)
    mfcc_b64 = fig_to_b64(fig1)
    plt.close(fig1)

    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    img2 = librosa.display.specshow(spec, y_axis="log", x_axis="time", sr=sr, ax=ax2)
    ax2.set_title("Log Spectrogram")
    fig2.colorbar(img2, ax=ax2)
    spec_b64 = fig_to_b64(fig2)
    plt.close(fig2)

    return Layer2Result(
        jitter=jitter,
        shimmer=shimmer,
        jitter_flag=jitter_flag,
        shimmer_flag=shimmer_flag,
        mfcc_image_b64=mfcc_b64,
        spectrogram_image_b64=spec_b64,
    )


def calculate_trust_score(layer1: Layer1Result, layer2: Layer2Result, model: ModelResult) -> tuple[float, str, str]:
    score = 100.0

    if layer1.bitrate_inconsistent:
        score -= 12
    if layer1.duration_size_mismatch:
        score -= 10
    if layer1.editing_software_detected:
        score -= min(12, 4 * len(layer1.editing_software_detected))
    if layer1.digital_silence_events > 3:
        score -= min(12, layer1.digital_silence_events)

    if layer2.jitter_flag:
        score -= 15
    if layer2.shimmer_flag:
        score -= 10

    score += (model.human_probability - 0.5) * 60
    score = float(np.clip(score, 0, 100))

    binary_verdict = "Human" if (score >= 60 and model.human_probability >= 0.5) else "Not Human"

    confidence = "Low"
    if score >= 80 or score <= 20:
        confidence = "High"
    elif score >= 65 or score <= 35:
        confidence = "Medium"

    if binary_verdict == "Human" and (layer1.editing_software_detected or layer1.digital_silence_events > 3):
        detail = "Authentic Voice, but Potentially Edited"
    elif binary_verdict == "Human":
        detail = f"Likely Human ({confidence} confidence)"
    else:
        detail = f"Likely Synthetic/Playback ({confidence} confidence)"

    return score, binary_verdict, detail


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "analyzed": False,
        "error": None,
        "limitations": "No detector can guarantee perfect human-vs-deepfake separation. This is a probabilistic forensic aid, not legal proof.",
    }

    if request.method == "POST":
        audio_file = request.files.get("audio")
        if not audio_file or audio_file.filename == "":
            context["error"] = "Please upload an audio file."
            return render_template("index.html", **context)

        suffix = Path(audio_file.filename).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            sf.write(tmp_path, y, sr)

            layer1 = analyze_layer1(tmp_path)
            layer2 = analyze_layer2(tmp_path)
            detector = HFHumanDetector(HF_MODEL_ID)
            model_result = detector.predict_human_probability(tmp_path)
            trust_score, binary_verdict, detail_verdict = calculate_trust_score(layer1, layer2, model_result)

            context.update(
                {
                    "analyzed": True,
                    "layer1": layer1,
                    "layer2": layer2,
                    "model": model_result,
                    "trust_score": trust_score,
                    "binary_verdict": binary_verdict,
                    "detail_verdict": detail_verdict,
                    "hf_model_id": HF_MODEL_ID,
                }
            )
        except Exception as exc:
            context["error"] = f"Analysis failed: {exc}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
