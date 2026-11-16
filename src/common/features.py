import numpy as np
from .data import SR

N_FFT, HOP = 2048, 256
FPS = SR / HOP


def stft_mag(x):
    win = np.hanning(N_FFT).astype(np.float32)
    x = np.pad(x, N_FFT // 2)
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1)).T


def mel_fb(n_mels=40, fmin=30.0, fmax=11000.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700.0)
    hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)), 0, None)
    return fb


FB = mel_fb()


def log_mel(x):
    return np.log1p(100 * FB @ stft_mag(x))


def onset_envelope(x, bands=None):
    m = log_mel(x)
    if bands is not None:
        m = m[bands]
    flux = np.maximum(np.diff(m, axis=1, prepend=m[:, :1]), 0).sum(axis=0)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    return flux / (flux.std() + 1e-9)


def low_onset(x):
    return onset_envelope(x, bands=slice(0, 6))
