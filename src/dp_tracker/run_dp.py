import numpy as np
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import beat_scores, write_rows
from .ellis import ellis_tempo, track_beats

TIGHTNESS = (10, 30, 100, 300, 1000, 3000)


def main():
    rows = []
    for t in data.split("test"):
        ref, _ = data.load_beats(t)
        env = onset_envelope(data.load_audio(data.audio_path(t)))
        tempos = {"est_tempo": ellis_tempo(env), "oracle_tempo": data.ref_tempo(ref)}
        for cond, bpm in tempos.items():
            for tight in TIGHTNESS:
                if cond == "oracle_tempo" and tight != 100:
                    continue
                s = beat_scores(ref, track_beats(env, bpm, tight))
                for k, v in s.items():
                    rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=f"dp_tight{tight}", condition=cond, metric=k, value=v))
        print(t, flush=True)
    print(write_rows("dp_gtzan.csv", rows))


if __name__ == "__main__":
    main()
