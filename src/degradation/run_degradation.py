import os
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import beat_scores, write_rows
from ..dp_tracker.ellis import ellis_tempo, track_beats
from .degrade import CONDITIONS

N = int(os.environ.get("N_TRACKS", 60))


def main():
    rows = []
    for t in data.split("test")[::max(1, len(data.split("test")) // N)][:N]:
        ref, _ = data.load_beats(t)
        x = data.load_audio(data.audio_path(t))
        for cond, fn in CONDITIONS.items():
            env = onset_envelope(fn(x))
            for tempo_in, bpm in (("est_tempo", ellis_tempo(env)), ("oracle_tempo", data.ref_tempo(ref))):
                s = beat_scores(ref, track_beats(env, bpm, 100))
                for k, v in s.items():
                    rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=f"dp_{tempo_in}", condition=cond, metric=k, value=v))
        print(t, flush=True)
    print(write_rows("degradation.csv", rows))


if __name__ == "__main__":
    main()
