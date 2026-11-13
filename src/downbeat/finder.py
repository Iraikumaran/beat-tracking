import numpy as np
from ..common.features import FPS



def beat_strength(env, beats, win=0.05):
    w = int(win * FPS)
    idx = np.round(np.asarray(beats) * FPS).astype(int)
    return np.array([env[max(0, i - w):i + w + 1].max() if i < len(env) else 0 for i in idx])


def find_downbeats(beats, env, length):
    s = beat_strength(env, beats)
    phase = int(np.argmax([s[p::length].mean() for p in range(length)]))
    return np.asarray(beats)[phase::length]
