from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
import json
import wave
import numpy as np

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def f(x):
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return x

def mode_val(arr):
    if len(arr) == 0:
        return 0.0
    vals, counts = np.unique(np.round(arr, 6), return_counts=True)
    return f(vals[np.argmax(counts)])

def parse_audio(b64):
    audio_bytes = base64.b64decode(b64)
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        data = data / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        data = data / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        data = data / 32768.0

    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)

    return data, sr, channels, frames

@app.get("/")
def root():
    return {"ok": True}

@app.post("/")
def analyze(req: AudioRequest):
    data, sr, channels, frames = parse_audio(req.audio_base64)

    flat = data.flatten()
    abs_flat = np.abs(flat)
    duration = f(frames / sr) if sr else 0.0
    rms = f(np.sqrt(np.mean(flat ** 2))) if len(flat) else 0.0
    zc = int(np.sum(np.diff(np.signbit(flat)) != 0)) if len(flat) > 1 else 0
    silence_ratio = f(np.mean(abs_flat < 0.01)) if len(flat) else 0.0

    columns = [
        "sample_rate",
        "channels",
        "frames",
        "duration_sec",
        "amplitude",
        "abs_amplitude",
        "rms",
        "zero_crossings",
        "silence_ratio"
    ]

    mean = {
        "sample_rate": f(sr),
        "channels": f(channels),
        "frames": f(frames),
        "duration_sec": duration,
        "amplitude": f(np.mean(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.mean(abs_flat)) if len(abs_flat) else 0.0,
        "rms": rms,
        "zero_crossings": f(zc),
        "silence_ratio": silence_ratio
    }

    std = {
        "sample_rate": 0.0,
        "channels": 0.0,
        "frames": 0.0,
        "duration_sec": 0.0,
        "amplitude": f(np.std(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.std(abs_flat)) if len(abs_flat) else 0.0,
        "rms": 0.0,
        "zero_crossings": 0.0,
        "silence_ratio": 0.0
    }

    variance = {
        "sample_rate": 0.0,
        "channels": 0.0,
        "frames": 0.0,
        "duration_sec": 0.0,
        "amplitude": f(np.var(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.var(abs_flat)) if len(abs_flat) else 0.0,
        "rms": 0.0,
        "zero_crossings": 0.0,
        "silence_ratio": 0.0
    }

    min_v = {
        "sample_rate": f(sr),
        "channels": f(channels),
        "frames": f(frames),
        "duration_sec": duration,
        "amplitude": f(np.min(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.min(abs_flat)) if len(abs_flat) else 0.0,
        "rms": rms,
        "zero_crossings": f(zc),
        "silence_ratio": silence_ratio
    }

    max_v = {
        "sample_rate": f(sr),
        "channels": f(channels),
        "frames": f(frames),
        "duration_sec": duration,
        "amplitude": f(np.max(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.max(abs_flat)) if len(abs_flat) else 0.0,
        "rms": rms,
        "zero_crossings": f(zc),
        "silence_ratio": silence_ratio
    }

    median = {
        "sample_rate": f(sr),
        "channels": f(channels),
        "frames": f(frames),
        "duration_sec": duration,
        "amplitude": f(np.median(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.median(abs_flat)) if len(abs_flat) else 0.0,
        "rms": rms,
        "zero_crossings": f(zc),
        "silence_ratio": silence_ratio
    }

    mode = {
        "sample_rate": f(sr),
        "channels": f(channels),
        "frames": f(frames),
        "duration_sec": duration,
        "amplitude": mode_val(flat),
        "abs_amplitude": mode_val(abs_flat),
        "rms": rms,
        "zero_crossings": f(zc),
        "silence_ratio": silence_ratio
    }

    range_v = {
        "sample_rate": 0.0,
        "channels": 0.0,
        "frames": 0.0,
        "duration_sec": 0.0,
        "amplitude": f(np.max(flat) - np.min(flat)) if len(flat) else 0.0,
        "abs_amplitude": f(np.max(abs_flat) - np.min(abs_flat)) if len(abs_flat) else 0.0,
        "rms": 0.0,
        "zero_crossings": 0.0,
        "silence_ratio": 0.0
    }

    allowed_values = {
        "sample_rate": [f(sr)],
        "channels": [f(channels)],
        "frames": [f(frames)]
    }

    value_range = {
        "sample_rate": {"min": f(sr), "max": f(sr)},
        "channels": {"min": f(channels), "max": f(channels)},
        "frames": {"min": f(frames), "max": f(frames)},
        "duration_sec": {"min": duration, "max": duration},
        "amplitude": {
            "min": f(np.min(flat)) if len(flat) else 0.0,
            "max": f(np.max(flat)) if len(flat) else 0.0
        },
        "abs_amplitude": {
            "min": f(np.min(abs_flat)) if len(abs_flat) else 0.0,
            "max": f(np.max(abs_flat)) if len(abs_flat) else 0.0
        },
        "rms": {"min": rms, "max": rms},
        "zero_crossings": {"min": f(zc), "max": f(zc)},
        "silence_ratio": {"min": silence_ratio, "max": silence_ratio}
    }

    correlation = []
    if data.shape[1] >= 2:
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        correlation.append({
            "column1": "channel_0",
            "column2": "channel_1",
            "value": f(corr)
        })

    return {
        "rows": int(len(flat)),
        "columns": columns,
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": min_v,
        "max": max_v,
        "median": median,
        "mode": mode,
        "range": range_v,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": correlation
    }
