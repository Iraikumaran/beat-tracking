import csv
import os
import numpy as np
import mir_eval

from .data import RESULTS

FIELDS = ["track", "dataset", "genre", "method", "condition", "metric", "value"]


def beat_scores(ref, est, trim=True):
    if trim:
        ref, est = mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est)
    f = mir_eval.beat.f_measure(ref, est)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref, est)
    return {"F": f, "CMLt": cmlt, "AMLt": amlt}


def write_rows(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def click_track(x, beats, sr, length=0.05):
    out = x.copy()
    t = np.arange(int(length * sr)) / sr
    click = 0.5 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 60)
    for b in beats:
        i = int(b * sr)
        out[i:i + len(click)] += click[:max(0, len(out) - i)]
    return out
