import os
import pickle
import time
import numpy as np
from scipy.signal import resample_poly
from madmom.audio.signal import Signal
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from ..common import data
from ..common.features import FPS
from ..common.evaluate import beat_scores, write_rows
from .model import frame_features


def to_signal(x):
    return Signal(resample_poly(x, 2, 1).astype(np.float32), sample_rate=2 * data.SR, num_channels=1)


N = int(os.environ.get("N_TRACKS", 0))


def tracks():
    t = data.split("test")
    return t[::max(1, len(t) // N)][:N] if N else t


def main():
    rnn, dbn = RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)
    dbn_ours = DBNBeatTrackingProcessor(fps=FPS)
    with open(os.path.join(data.RESULTS, "mlp_beat.pkl"), "rb") as f:
        model = pickle.load(f)
    rows = []
    for t in tracks():
        ref, _ = data.load_beats(t)
        x = data.load_audio(data.audio_path(t))
        minutes = len(x) / data.SR / 60
        t0 = time.perf_counter()
        est = dbn(rnn(to_signal(x)))
        mm_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        est2 = dbn_ours(model.predict_proba(frame_features(x))[:, 1])
        ours_time = time.perf_counter() - t0
        for method, b, sec in (("madmom_rnn_dbn", est, mm_time), ("mlp_dbn", est2, ours_time)):
            base = dict(track=t, dataset="gtzan", genre=data.genre(t), method=method, condition="clean")
            rows += [dict(base, metric=k, value=v) for k, v in beat_scores(ref, b).items()]
            rows.append(dict(base, metric="sec_per_min", value=sec / minutes))
        print(t, flush=True)
    print(write_rows("nn_madmom.csv", rows))


if __name__ == "__main__":
    main()
