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


def candidates(ac, k=10):
    lo, hi = lag_range()
    seg = ac[lo:hi + 1]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    peaks = sorted(peaks, key=lambda i: -seg[i])[:k]
    return [lo + p for p in peaks]


def phase_means(env, period):
    n = (len(env) // period) * period
    return env[:n].reshape(-1, period).mean(axis=0)


def pulse_score(env, lag):
    total = np.zeros(lag)
    for mult, w in ((1, 1.0), (1.5, 0.5), (2, 0.5)):
        means = phase_means(env, int(round(mult * lag)))
        total += w * means[np.arange(lag) % len(means)]
    return total.max()


def tempo_pulse(env):
    ac = autocorr(env)
    lags = candidates(ac)
    if not lags:
        return tempo_autocorr(env)
    scores = [pulse_score(env, lag) * ac[lag] for lag in lags]
    return 60 * FPS / lags[int(np.argmax(scores))]


def tempo_ioi(env, thresh=1.0):
    peaks = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thresh)) + 1
    lo, hi = lag_range()
    d = (peaks[None, :] - peaks[:, None]).ravel()
    d = d[(d >= lo) & (d <= hi)]
    if len(d) == 0:
        return 120.0
    hist = np.bincount(d, minlength=hi + 1).astype(float)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    return 60 * FPS / np.argmax(hist)


def octave_prior(bpm, center=120.0, octaves=1.0):
    return np.exp(-0.5 * (np.log2(bpm / center) / octaves) ** 2)


def tempo_prior(env):
    a = autocorr(env)
    lags = candidates(a)
    if not lags:
        return tempo_autocorr(env)
    scores = [a[lag] * octave_prior(60 * FPS / lag) for lag in lags]
    return 60 * FPS / lags[int(np.argmax(scores))]
