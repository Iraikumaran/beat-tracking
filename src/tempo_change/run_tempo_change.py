import os
import numpy as np
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import write_rows
from ..tempo import autocorr as ac
from ..tempo.metrics import acc1, acc2, octave_label
from .stretch import stretch, splice
from .local_tempo import local_tempo, settle_time

FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
WINDOWS = (4.0, 6.0, 8.0, 12.0)
N = int(os.environ.get("N_TRACKS", 60))


def main():
    rows, base = [], dict(dataset="gtzan")
    tracks = data.split("test")[::max(1, len(data.split("test")) // N)][:N]
    for t in tracks:
        x = data.load_audio(data.audio_path(t))
        ref = data.ref_tempo(data.load_beats(t)[0])
        g = data.genre(t)
        for f in FACTORS:
            env = onset_envelope(stretch(x, f))
            target = ref * f
            for name, fn in (("autocorr", ac.tempo_autocorr), ("autocorr+prior", ac.tempo_prior), ("ioi_hist", ac.tempo_ioi)):
                est = fn(env)
                b = dict(base, track=t, genre=g, method=name, condition=f"stretch_{f}")
                rows += [dict(b, metric="acc1", value=int(acc1(est, target))), dict(b, metric="acc2", value=int(acc2(est, target))),
                         dict(b, metric="octave", value=octave_label(est, target))]
            for w in WINDOWS:
                times, bpms = local_tempo(env, w)
                err = np.median(np.abs(bpms - target) / target)
                rows.append(dict(base, track=t, genre=g, method=f"local_{int(w)}s", condition=f"stretch_{f}", metric="rel_err", value=err))
        y, cut = splice(x, 0.85, 1.15, 15.0)
        env = onset_envelope(y)
        times, bpms = local_tempo(env, 6.0)
        truth = np.where(times < cut, ref * 0.85, ref * 1.15)
        b = dict(base, track=t, genre=g, method="local_6s", condition="splice_0.85_1.15")
        rows += [dict(b, metric="mae_bpm", value=np.mean(np.abs(bpms - truth))),
                 dict(b, metric="settle_s", value=settle_time(times, bpms, cut, ref * 1.15))]
        print(t, flush=True)
    print(write_rows("tempo_change.csv", rows))


if __name__ == "__main__":
    main()
