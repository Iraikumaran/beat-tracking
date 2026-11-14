import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import librosa
import mir_eval
from ..common import data
from ..common.features import onset_envelope, HOP
from ..common.evaluate import beat_scores, write_rows
from .ellis import ellis_tempo, track_beats


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def warm_up():
    librosa.beat.beat_track(y=np.random.default_rng(0).standard_normal(data.SR * 10).astype(np.float32), sr=data.SR)


def smooth(env, bpm):
    p = 60 * data.SR / HOP / bpm
    w = np.exp(-0.5 * (np.arange(-p, p + 1) * 32.0 / p) ** 2)
    return np.convolve(env, w, mode="same")


def ours(x):
    env = onset_envelope(x)
    bpm = ellis_tempo(env)
    return track_beats(env, bpm, 100), bpm


def run_track(t):
    rows = []
    ref, _ = data.load_beats(t)
    x = data.load_audio(data.audio_path(t))
    row = lambda m, c, k, v: rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=m, condition=c, metric=k, value=v))
    (dp_beats, dp_bpm), dp_time = timed(lambda: ours(x))
    (lib_bpm, lib_beats), lib_time = timed(lambda: librosa.beat.beat_track(y=x, sr=data.SR, units="time"))
    lib_bpm = float(np.atleast_1d(lib_bpm)[0])
    env = onset_envelope(x)
    for k, v in beat_scores(ref, track_beats(env, lib_bpm, 100)).items():
        row("dp_tight100", "librosa_tempo", k, v)
    for k, v in beat_scores(ref, lib_beats).items():
        row("librosa_beat_track", "default", k, v)
    lenv = librosa.onset.onset_strength(y=x, sr=data.SR, hop_length=HOP)
    bpm, lb = librosa.beat.beat_track(onset_envelope=lenv, sr=data.SR, hop_length=HOP, tightness=100, units="time")
    bpm = float(np.atleast_1d(bpm)[0])
    mine = track_beats(lenv / (lenv.std() + 1e-9), bpm, 100)
    row("dp_vs_librosa", "same_env_same_tempo", "agreement_F", mir_eval.beat.f_measure(lb, mine))
    row("dp_vs_librosa", "same_env_same_tempo", "agreement_F_trimmed", mir_eval.beat.f_measure(mir_eval.beat.trim_beats(lb), mir_eval.beat.trim_beats(mine)))
    mine_s = track_beats(smooth(lenv / (lenv.std() + 1e-9), bpm), bpm, 100)
    row("dp_vs_librosa", "same_env_same_tempo_smoothed", "agreement_F", mir_eval.beat.f_measure(lb, mine_s))
    row("dp_vs_librosa", "same_env_same_tempo_smoothed", "agreement_F_trimmed", mir_eval.beat.f_measure(mir_eval.beat.trim_beats(lb), mir_eval.beat.trim_beats(mine_s)))
    for m, v in (("ref", data.ref_tempo(ref)), ("dp_ellis_tempo", dp_bpm), ("librosa_tempo", lib_bpm)):
        row(m, "default", "bpm", v)
    row("dp_tight100", "est_tempo", "seconds", dp_time)
    row("librosa_beat_track", "default", "seconds", lib_time)
    print(t, flush=True)
    return rows


def main():
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 4)), initializer=warm_up) as ex:
        rows = [r for rs in ex.map(run_track, data.split("test")) for r in rs]
    print(write_rows("dp_validation.csv", rows))


if __name__ == "__main__":
    main()
