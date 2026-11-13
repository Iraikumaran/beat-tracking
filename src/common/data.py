import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.environ.get("GTZAN_AUDIO", os.path.join(ROOT, "data", "genres_original"))
ANN_DIR = os.path.join(ROOT, "data", "gtzan_tempo_beat")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
SR = 22050


def split(name):
    with open(os.path.join(ROOT, "data", "splits", f"{name}_filtered.txt")) as f:
        tracks = [line.strip().split("/")[-1][:-4] for line in f if line.strip()]
    return [t for t in tracks if os.path.exists(audio_path(t)) and os.path.exists(beat_path(t))]


def genre(track):
    return track.split(".")[0]


def audio_path(track):
    return os.path.join(AUDIO_DIR, genre(track), track + ".wav")


def beat_path(track):
    return os.path.join(ANN_DIR, "beats", "gtzan_" + track.replace(".", "_") + ".beats")


def load_audio(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.5:
        x /= 32768.0
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    return x


def load_beats(track):
    a = np.loadtxt(beat_path(track), ndmin=2)
    beats = a[:, 0]
    pos = a[:, 1].astype(int) if a.shape[1] > 1 else None
    return beats, pos


def ref_tempo(beats):
    return 60.0 / np.median(np.diff(beats))
