# Push rules (DP2_B)

## What goes in the repo
- Our code: `src/`, `scripts/`
- Split lists: `data/splits/`
- Our results: small CSV files in `results/`, figures in `figures/`
- Report source and PDF in `report/`
- `README.md`, `requirements.txt`, `.gitignore`, `PUSH_RULES.md`

## What never goes in the repo
- GTZAN audio (`data/genres_original/`): 1.2 GB and copyrighted. Everyone downloads it from Kaggle.
- Beat annotations (`data/gtzan_tempo_beat/`): someone else's repo. `scripts/get_data.sh` downloads it.
- `venv/`, `results_test/`, `__pycache__/`, `.DS_Store`
- Any file over 50 MB
`.gitignore` already blocks these.

## How we push
1. Work on the `Dev` branch. Never push straight to `main`.
2. `git pull` before you start and before you push.
3. Only change your own folders:
   - Irai: `src/common/`, `src/tempo/`, `src/tempo_change/`, `report/`, `scripts/make_figures.py`
   - Nalluraj: `src/dp_tracker/`, `src/degradation/`
   - Usha: `src/nn/`, `src/downbeat/`
   Ask in the group chat before touching someone else's files.
4. Run your code once before pushing.
5. `git status` before committing. Only your files should be listed.
6. One clear commit message: what you added, e.g. "Week 2: autocorrelation tempo estimator".
7. Never use `git push --force`.
8. Irai merges `Dev` into `main` after each week.
