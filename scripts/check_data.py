import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import data

for name in ("train", "valid", "test"):
    print(name, len(data.split(name)), "songs found")
t = data.split("test")[0]
x = data.load_audio(data.audio_path(t))
beats, pos = data.load_beats(t)
print(t, f"{len(x) / data.SR:.1f} s,", len(beats), "beats,", f"{data.ref_tempo(beats):.1f} BPM")
