import os
import numpy as np
import mir_eval
from scipy.signal import resample_poly
from madmom.audio.signal import Signal
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor, RNNBarProcessor, DBNBarTrackingProcessor
from ..common import data
from ..common.evaluate import write_rows


N = int(os.environ.get("N_TRACKS", 0))


def tracks():
    t = data.split("test")
    return t[::max(1, len(t) // N)][:N] if N else t


def main():
    rnn, dbn = RNNDownBeatProcessor(), DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    bar_rnn, bar_dbn = RNNBarProcessor(), DBNBarTrackingProcessor(beats_per_bar=[3, 4])
    rows = []
    for t in tracks():
        beats, pos = data.load_beats(t)
        if pos is None or pos.max() < 2:
            continue
        ref_db = beats[pos == 1]
        x = resample_poly(data.load_audio(data.audio_path(t)), 2, 1).astype(np.float32)
        sig = Signal(x, sample_rate=2 * data.SR, num_channels=1)
        out = dbn(rnn(sig))
        est = {"madmom_joint": out[out[:, 1] == 1, 0]}
        try:
            bars = bar_dbn(bar_rnn((sig, beats)))
            est["madmom_bar_on_annotated"] = bars[bars[:, 1] == 1, 0]
        except Exception as e:
            print(t, "bar tracker failed:", e)
        base = dict(track=t, dataset="gtzan", genre=data.genre(t), metric="F")
        for method, d in est.items():
            cond = "annotated_beats" if "annotated" in method else "audio"
            rows.append(dict(base, method=method.replace("_on_annotated", ""), condition=cond, value=mir_eval.beat.f_measure(ref_db, d)))
        print(t, flush=True)
    print(write_rows("downbeat_madmom.csv", rows))


if __name__ == "__main__":
    main()
