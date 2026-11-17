import numpy as np
from scipy.ndimage import median_filter
from ..common.features import stft_mag, FB, onset_envelope

LOW_BANDS = slice(0, 11)


def percussive_envelope(x, kernel=17):
    s = stft_mag(x) ** 2
    h = median_filter(s, size=(1, kernel))
    p = median_filter(s, size=(kernel, 1))
    mask = p ** 2 / (p ** 2 + h ** 2 + 1e-12)
    m = np.log1p(100 * FB @ np.sqrt(s * mask))
    flux = np.maximum(np.diff(m, axis=1, prepend=m[:, :1]), 0).sum(axis=0)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    return flux / (flux.std() + 1e-9)


def low_band_envelope(x):
    return onset_envelope(x, bands=LOW_BANDS)


ENVELOPES = {"standard": onset_envelope, "percussive": percussive_envelope, "lowband": low_band_envelope}
