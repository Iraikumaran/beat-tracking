import numpy as np
from ..common.features import FPS


def ellis_tempo(env, center=120.0, octaves=1.4):
    n = len(env)
    f = np.fft.rfft(env - env.mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lags = np.arange(1, n)
    w = np.exp(-0.5 * (np.log2(lags / (FPS * 60 / center)) / octaves) ** 2)
    lo, hi = int(FPS * 60 / 250), int(FPS * 60 / 40)
    score = (ac[1:] * w)[lo - 1:hi]
    return 60 * FPS / (lo + np.argmax(score))


def track_beats(env, bpm, tightness=100.0):
    period = 60 * FPS / bpm
    n = len(env)
    score = env.astype(float).copy()
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    offsets = np.arange(-hi, -lo + 1)
    penalty = -tightness * np.log(-offsets / period) ** 2
    for t in range(hi, n):
        prev = score[t + offsets] + penalty
        k = np.argmax(prev)
        score[t] = env[t] + prev[k]
        back[t] = t + offsets[k]
    tail = np.arange(max(0, n - int(period)), n)
    t = tail[np.argmax(score[tail])]
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1]) / FPS
