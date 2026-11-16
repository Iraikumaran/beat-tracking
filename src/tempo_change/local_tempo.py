import numpy as np
from ..common.features import FPS
from ..tempo.autocorr import tempo_prior


def local_tempo(env, win_s=8.0, hop_s=1.0):
    w, h = int(win_s * FPS), int(hop_s * FPS)
    times, bpms = [], []
    for start in range(0, max(1, len(env) - w + 1), h):
        bpms.append(tempo_prior(env[start:start + w]))
        times.append((start + w / 2) / FPS)
    return np.array(times), np.array(bpms)


def settle_time(times, bpms, change_t, target, tol=0.04):
    after = times >= change_t
    ok = np.abs(bpms - target) <= tol * target
    for i in np.flatnonzero(after):
        if ok[i:][after[i:]].all():
            return times[i] - change_t
    return np.nan
