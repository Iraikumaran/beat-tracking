import os
import pickle
import numpy as np
from ..common import data
from ..common.evaluate import beat_scores, write_rows
from .model import frame_features, frame_labels, build_model, pick_beats

NEG_RATIO = 3
PER_GENRE = int(os.environ.get("TRAIN_PER_GENRE", 16))


def training_set(tracks, rng):
    X, Y = [], []
    for t in tracks:
        f = frame_features(data.load_audio(data.audio_path(t)))
        y = frame_labels(data.load_beats(t)[0], len(f))
        pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
        neg = rng.choice(neg, min(len(neg), NEG_RATIO * len(pos)), replace=False)
        idx = np.concatenate([pos, neg])
        X.append(f[idx]); Y.append(y[idx])
    return np.vstack(X), np.concatenate(Y)


def main():
    rng = np.random.default_rng(0)
    pool = data.split("train") + data.split("valid")
    tracks = [t for g in sorted({data.genre(t) for t in pool}) for t in [u for u in pool if data.genre(u) == g][:PER_GENRE]]
    X, Y = training_set(tracks, rng)
    print("training on", X.shape, flush=True)
    model = build_model().fit(X, Y)
    with open(os.path.join(data.RESULTS, "mlp_beat.pkl"), "wb") as f:
        pickle.dump(model, f)
    rows = []
    for t in data.split("test"):
        ref, _ = data.load_beats(t)
        act = model.predict_proba(frame_features(data.load_audio(data.audio_path(t))))[:, 1]
        est = pick_beats(act)
        np.save(os.path.join(data.RESULTS, "nn_beats", t + ".npy"), est)
        for k, v in beat_scores(ref, est).items():
            rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method="mlp_peaks", condition="clean", metric=k, value=v))
    print(write_rows("nn_gtzan.csv", rows))


if __name__ == "__main__":
    os.makedirs(os.path.join(data.RESULTS, "nn_beats"), exist_ok=True)
    main()
