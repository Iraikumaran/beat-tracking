import time
import numpy as np
from ..common import data
from ..common.evaluate import write_rows
from .metrics import acc1, acc2, octave_label


def librosa_tempo(path):
    import librosa
    y = data.load_audio(path)
    fn = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    return float(np.atleast_1d(fn(y=y, sr=data.SR))[0])


def madmom_tempo(path):
    from scipy.signal import resample_poly
    from madmom.audio.signal import Signal
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
    x = resample_poly(data.load_audio(path), 2, 1).astype(np.float32)
    act = RNNBeatProcessor()(Signal(x, sample_rate=2 * data.SR, num_channels=1))
    return float(TempoEstimationProcessor(fps=100)(act)[0][0])


def available():
    methods = {}
    for name, fn, mod in (("librosa", librosa_tempo, "librosa"), ("madmom", madmom_tempo, "madmom")):
        try:
            __import__(mod)
            methods[name] = fn
        except ImportError:
            print(name, "not installed, skipped")
    return methods


def main():
    methods = available()
    rows = []
    for t in data.split("test"):
        ref = data.ref_tempo(data.load_beats(t)[0])
        path = data.audio_path(t)
        for name, fn in methods.items():
            t0 = time.time()
            est = fn(path)
            dt = time.time() - t0
            base = dict(track=t, dataset="gtzan", genre=data.genre(t), method=name, condition="clean")
            rows += [dict(base, metric="bpm", value=est), dict(base, metric="ref_bpm", value=ref),
                     dict(base, metric="acc1", value=int(acc1(est, ref))), dict(base, metric="acc2", value=int(acc2(est, ref))),
                     dict(base, metric="octave", value=octave_label(est, ref)), dict(base, metric="seconds", value=dt)]
        print(t, flush=True)
    print(write_rows("tempo_compare.csv", rows))


if __name__ == "__main__":
    main()
