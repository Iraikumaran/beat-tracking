import time
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import write_rows
from . import autocorr as ac
from .metrics import acc1, acc2, octave_label

METHODS = {"autocorr": ac.tempo_autocorr, "autocorr+pulse": ac.tempo_pulse, "autocorr+prior": ac.tempo_prior, "ioi_hist": ac.tempo_ioi}


def main():
    rows = []
    for t in data.split("test"):
        beats, _ = data.load_beats(t)
        ref = data.ref_tempo(beats)
        env = onset_envelope(data.load_audio(data.audio_path(t)))
        for name, fn in METHODS.items():
            t0 = time.time()
            est = fn(env)
            dt = time.time() - t0
            base = dict(track=t, dataset="gtzan", genre=data.genre(t), method=name, condition="clean")
            rows += [dict(base, metric="bpm", value=est), dict(base, metric="ref_bpm", value=ref),
                     dict(base, metric="acc1", value=int(acc1(est, ref))), dict(base, metric="acc2", value=int(acc2(est, ref))),
                     dict(base, metric="octave", value=octave_label(est, ref)), dict(base, metric="seconds", value=dt)]
    print(write_rows("tempo_gtzan.csv", rows))


if __name__ == "__main__":
    main()
