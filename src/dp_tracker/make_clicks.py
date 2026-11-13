import os
import numpy as np
from scipy.io import wavfile
from ..common import data
from ..common.features import onset_envelope
from ..common.evaluate import click_track
from .ellis import ellis_tempo, track_beats

TRACKS = ("blues.00012", "classical.00020", "disco.00021", "jazz.00073", "rock.00090")


def main():
    out = os.path.join(data.ROOT, "data", "click_tracks")
    os.makedirs(out, exist_ok=True)
    for t in TRACKS:
        if t not in data.split("test"):
            continue
        ref, _ = data.load_beats(t)
        x = data.load_audio(data.audio_path(t))
        env = onset_envelope(x)
        for name, bpm in (("est", ellis_tempo(env)), ("oracle", data.ref_tempo(ref))):
            beats = track_beats(env, bpm, 100)
            y = click_track(x, beats, data.SR)
            wavfile.write(os.path.join(out, f"{t}_{name}_{bpm:.0f}bpm.wav"), data.SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
            np.savetxt(os.path.join(out, f"{t}_{name}.beats"), beats, fmt="%.3f")
        print(t, flush=True)


if __name__ == "__main__":
    main()
