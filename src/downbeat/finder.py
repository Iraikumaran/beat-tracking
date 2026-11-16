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


def find_downbeats_auto(beats, env, lengths=(3, 4)):
    s = beat_strength(env, beats)
    best = max(((s[p::L].mean() - s.mean(), L, p) for L in lengths for p in range(L) if len(s[p::L])), key=lambda c: c[0])
    return np.asarray(beats)[best[2]::best[1]], best[1]
