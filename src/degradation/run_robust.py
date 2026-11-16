import os
from ..common import data
from ..common.evaluate import beat_scores, write_rows
from ..dp_tracker.ellis import ellis_tempo, track_beats
from .degrade import CONDITIONS
from .robust_onset import percussive_envelope

N = int(os.environ.get("N_TRACKS", 58))


def main():
    rows = []
    tracks = data.split("test")[::max(1, len(data.split("test")) // N)][:N]
    for t in tracks:
        ref, _ = data.load_beats(t)
        x = data.load_audio(data.audio_path(t))
        for cond, fn in CONDITIONS.items():
            env = percussive_envelope(fn(x))
            for k, v in beat_scores(ref, track_beats(env, ellis_tempo(env), 100)).items():
                rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method="dp_percussive", condition=cond, metric=k, value=v))
        print(t, flush=True)
    print(write_rows("degradation_percussive.csv", rows))


if __name__ == "__main__":
    main()
