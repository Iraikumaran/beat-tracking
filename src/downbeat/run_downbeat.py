import os
import numpy as np
import mir_eval
from ..common import data
from ..common.features import low_onset
from ..common.evaluate import write_rows
from .finder import find_downbeats


def main():
    rows = []
    nn_dir = os.path.join(data.RESULTS, "nn_beats")
    for t in data.split("test"):
        beats, pos = data.load_beats(t)
        if pos is None or pos.max() < 2:
            continue
        ref_db = beats[pos == 1]
        length = int(pos.max())
        env = low_onset(data.load_audio(data.audio_path(t)))
        inputs = {"annotated_beats": beats}
        nn_path = os.path.join(nn_dir, t + ".npy")
        if os.path.exists(nn_path):
            inputs["nn_beats"] = np.load(nn_path)
        for cond, b in inputs.items():
            if len(b) < length:
                continue
            for method, est in (("low_onset_finder", find_downbeats(b, env, length)), ("first_beat", b[::length])):
                f = mir_eval.beat.f_measure(ref_db, est)
                rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=method, condition=cond, metric="F", value=f))
    print(write_rows("downbeat_gtzan.csv", rows))


if __name__ == "__main__":
    main()
