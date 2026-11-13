import numpy as np
from ..common.features import FPS

BPM_MIN, BPM_MAX = 40, 250


def autocorr(env):
    n = len(env)
    f = np.fft.rfft(env - env.mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / (ac[0] + 1e-9)


def lag_range():
    return int(FPS * 60 / BPM_MAX), int(FPS * 60 / BPM_MIN)


def tempo_autocorr(env):
    ac = autocorr(env)
    lo, hi = lag_range()
    lag = lo + np.argmax(ac[lo:hi + 1])
    return 60 * FPS / lag
