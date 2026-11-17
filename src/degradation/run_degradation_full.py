import os
from concurrent.futures import ProcessPoolExecutor
from ..common import data
from ..common.evaluate import beat_scores, write_rows
from ..dp_tracker.ellis import ellis_tempo, track_beats
from .degrade import CONDITIONS
from .robust_onset import ENVELOPES

N = int(os.environ.get("N_TRACKS", 0))


def run_track(t):
    rows = []
    ref, _ = data.load_beats(t)
    x = data.load_audio(data.audio_path(t))
    for cond, fn in CONDITIONS.items():
        y = fn(x)
        for name, env_fn in ENVELOPES.items():
            env = env_fn(y)
            for tempo_in, bpm in (("est_tempo", ellis_tempo(env)), ("oracle_tempo", data.ref_tempo(ref))):
                for k, v in beat_scores(ref, track_beats(env, bpm, 100)).items():
                    rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=f"dp_{name}_{tempo_in}", condition=cond, metric=k, value=v))
            rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=f"dp_{name}_est_tempo", condition=cond, metric="bpm", value=ellis_tempo(env)))
    print(t, flush=True)
    return rows


def main():
    tracks = data.split("test")
    if N:
        tracks = tracks[::max(1, len(tracks) // N)][:N]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
        rows = [r for rs in ex.map(run_track, tracks) for r in rs]
    print(write_rows("degradation_full.csv", rows))


if __name__ == "__main__":
    main()
