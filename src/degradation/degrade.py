import os
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from ..common.data import SR


def pink_noise(n, rng):
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    f[1:] /= np.sqrt(np.arange(1, len(f)))
    return np.fft.irfft(f, n)


def add_noise(x, snr_db, seed=0, white=False):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x)) if white else pink_noise(len(x), rng)
    noise *= np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return (x + noise).astype(np.float32)


def lowpass(x, cutoff):
    return sosfiltfilt(butter(6, cutoff, fs=SR, output="sos"), x).astype(np.float32)


def mp3(x, kbps):
    with tempfile.TemporaryDirectory() as d:
        src, enc, dst = (os.path.join(d, n) for n in ("in.wav", "x.mp3", "out.wav"))
        wavfile.write(src, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-b:a", f"{kbps}k", enc], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", enc, "-ar", str(SR), "-ac", "1", dst], check=True)
        y = wavfile.read(dst)[1].astype(np.float32) / 32768.0
    return y[:len(x)]


CONDITIONS = {
    "clean": lambda x: x,
    "noise_20dB": lambda x: add_noise(x, 20), "noise_10dB": lambda x: add_noise(x, 10),
    "noise_5dB": lambda x: add_noise(x, 5), "noise_0dB": lambda x: add_noise(x, 0),
    "noise_-10dB": lambda x: add_noise(x, -10), "white_0dB": lambda x: add_noise(x, 0, white=True),
    "lowpass_4k": lambda x: lowpass(x, 4000), "lowpass_2k": lambda x: lowpass(x, 2000), "lowpass_1k": lambda x: lowpass(x, 1000),
    "mp3_128k": lambda x: mp3(x, 128), "mp3_64k": lambda x: mp3(x, 64), "mp3_32k": lambda x: mp3(x, 32),
}
