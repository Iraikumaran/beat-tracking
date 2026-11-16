import numpy as np
from sklearn.neural_network import MLPClassifier
from ..common.features import log_mel, FPS

CONTEXT = 4


def frame_features(x):
    m = log_mel(x)
    d = np.maximum(np.diff(m, axis=1, prepend=m[:, :1]), 0)
    f = np.vstack([m, d]).T
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    pad = np.pad(f, ((CONTEXT, CONTEXT), (0, 0)), mode="edge")
    return np.hstack([pad[i:i + len(f)] for i in range(2 * CONTEXT + 1)]).astype(np.float32)


def frame_labels(beats, n, widen=1):
    y = np.zeros(n, dtype=np.int8)
    for b in np.round(np.asarray(beats) * FPS).astype(int):
        y[max(0, b - widen):b + widen + 1] = 1
    return y


def build_model(seed=0):
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                         max_iter=40, early_stopping=True, validation_fraction=0.1, random_state=seed)


def pick_beats(act, min_gap=0.2, delta=0.05):
    act = np.convolve(act, np.hanning(5) / np.hanning(5).sum(), mode="same")
    local = np.convolve(act, np.ones(int(FPS)) / int(FPS), mode="same")
    gap = int(min_gap * FPS)
    peaks = [i for i in range(1, len(act) - 1) if act[i] >= act[i - 1] and act[i] > act[i + 1] and act[i] > local[i] + delta]
    keep = []
    for p in sorted(peaks, key=lambda i: -act[i]):
        if all(abs(p - q) >= gap for q in keep):
            keep.append(p)
    return np.array(sorted(keep)) / FPS
