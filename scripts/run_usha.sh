#!/bin/bash
set -e
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg is missing. Install it with: brew install ffmpeg"; exit 1; }
[ -d venv ] || { python3 -m venv venv; ./venv/bin/pip install -q -r requirements.txt; }
source venv/bin/activate
[ -d data/gtzan_tempo_beat ] || git clone -q https://github.com/TempoBeatDownbeat/gtzan_tempo_beat data/gtzan_tempo_beat
if [ -z "$GTZAN_AUDIO" ]; then
  if [ -d data/genres_original ]; then export GTZAN_AUDIO=$(pwd)/data/genres_original; else export GTZAN_AUDIO=$(cd .. && pwd)/genres_original; fi
fi
[ -d "$GTZAN_AUDIO/blues" ] || { echo "GTZAN audio not found. Put the genres_original folder inside data/"; exit 1; }
if [ "$1" = "full" ]; then export N_TRACKS=58; else export N_TRACKS=5 RESULTS_DIR=$(pwd)/results_test; fi
echo "Audio: $GTZAN_AUDIO   tracks for slow studies: $N_TRACKS"
python -m src.tcn.run_nn
python -m src.downbeat.run_downbeat
echo "Done. Results in ${RESULTS_DIR:-results}"
