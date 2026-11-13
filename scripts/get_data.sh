#!/bin/bash
set -e
cd "$(dirname "$0")/.."
[ -d data/gtzan_tempo_beat ] || git clone -q https://github.com/TempoBeatDownbeat/gtzan_tempo_beat data/gtzan_tempo_beat
echo "Annotations: data/gtzan_tempo_beat"
echo "Audio: put GTZAN genres_original in data/genres_original, or set GTZAN_AUDIO to its path"
