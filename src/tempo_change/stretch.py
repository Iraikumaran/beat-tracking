import os
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile
from ..common.data import SR


def ffmpeg_filter(x, filt, codec_args=()):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        wavfile.write(src, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-af", filt, "-ar", str(SR), "-ac", "1", dst], check=True)
        return wavfile.read(dst)[1].astype(np.float32) / 32768.0


def stretch(x, factor):
    return ffmpeg_filter(x, f"atempo={factor}")


def splice(x, a, b, cut_time):
    i = int(cut_time * SR)
    first, second = stretch(x[:i], a), stretch(x[i:], b)
    fade = int(0.01 * SR)
    ramp = np.linspace(0, 1, fade)
    joined = np.concatenate([first[:-fade], first[-fade:] * (1 - ramp) + second[:fade] * ramp, second[fade:]])
    return joined, len(first) / SR
