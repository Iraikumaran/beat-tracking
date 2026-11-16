import os
import numpy as np
from ..common import data
from ..common.evaluate import write_rows
from ..tempo.metrics import acc1, acc2, octave_label
from .stretch import stretch
from .run_tempo_change import FACTORS, N


def librosa_tempo(x):
    import librosa
    fn = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    return float(np.atleast_1d(fn(y=x, sr=data.SR))[0])


def madmom_tempo(x):
    from scipy.signal import resample_poly
    from madmom.audio.signal import Signal
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
    y = resample_poly(x, 2, 1).astype(np.float32)
    act = RNNBeatProcessor()(Signal(y, sample_rate=2 * data.SR, num_channels=1))
    return float(TempoEstimationProcessor(fps=100)(act)[0][0])


def main():
    methods = {m: f for m, f in (("librosa", librosa_tempo), ("madmom", madmom_tempo)) if m in os.environ.get("METHODS", "librosa,madmom")}
    rows = []
    tracks = data.split("test")[::max(1, len(data.split("test")) // N)][:N]
    for t in tracks:
        x = data.load_audio(data.audio_path(t))
        ref = data.ref_tempo(data.load_beats(t)[0])
        for f in FACTORS:
            y, target = stretch(x, f), ref * f
            for name, fn in methods.items():
                est = fn(y)
                b = dict(dataset="gtzan", track=t, genre=data.genre(t), method=name, condition=f"stretch_{f}")
                rows += [dict(b, metric="acc1", value=int(acc1(est, target))), dict(b, metric="acc2", value=int(acc2(est, target))),
                         dict(b, metric="octave", value=octave_label(est, target))]
        print(t, flush=True)
    print(write_rows(f"tempo_change_{'_'.join(methods)}.csv", rows))


if __name__ == "__main__":
    main()
