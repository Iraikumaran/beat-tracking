import os
import numpy as np
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import write_rows
from .stretch import splice
from .local_tempo import local_tempo
from .detect import detect_change

N = int(os.environ.get("N_TRACKS", 60))
PAIRS = ((0.85, 1.15), (1.15, 0.85))


def main():
    rows = []
    rng = np.random.default_rng(0)
    tracks = data.split("test")[::max(1, len(data.split("test")) // N)][:N]
    for t in tracks:
        x = data.load_audio(data.audio_path(t))
        cut_src = rng.uniform(10, 20)
        for a, b in PAIRS:
            y, cut = splice(x, a, b, cut_src)
            times, bpms = local_tempo(onset_envelope(y), 6.0)
            for method, folded in (("change_folded", True), ("change_raw", False)):
                err = abs(detect_change(times, bpms, folded) - cut)
                rows.append(dict(track=t, dataset="gtzan", genre=data.genre(t), method=method, condition=f"splice_{a}_{b}", metric="abs_err_s", value=err))
        print(t, flush=True)
    print(write_rows("tempo_change_detect.csv", rows))


if __name__ == "__main__":
    main()
