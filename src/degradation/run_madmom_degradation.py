import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.signal import resample_poly
from ..common import data
from ..common.evaluate import beat_scores, write_rows
from .degrade import CONDITIONS

N = int(os.environ.get("N_TRACKS", 58))
CONDS = os.environ.get("CONDS", ",".join(CONDITIONS)).split(",")


def run_track(t):
    from madmom.audio.signal import Signal
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    rnn, dbn = RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)
    ref, _ = data.load_beats(t)
    x = data.load_audio(data.audio_path(t))
    rows = []
    for cond in CONDS:
        fn = CONDITIONS[cond]
        y = resample_poly(fn(x), 2, 1).astype(np.float32)
        est = dbn(rnn(Signal(y, sample_rate=2 * data.SR, num_channels=1)))
        for k, v in beat_scores(ref, est).items():
            rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method="madmom_rnn_dbn", condition=cond, metric=k, value=v))
    print(t, flush=True)
    return rows


def main():
    tracks = data.split("test")[::max(1, len(data.split("test")) // N)][:N]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
        rows = [r for rs in ex.map(run_track, tracks) for r in rs]
    print(write_rows("degradation_madmom.csv", rows))


if __name__ == "__main__":
    main()
