import os
import numpy as np
import pandas as pd
from ..common.data import RESULTS


def tempo_accuracy(bpm, ref):
    r = bpm / ref
    acc1 = (abs(r - 1) <= 0.04).mean()
    acc2 = np.any([abs(r / k - 1) <= 0.04 for k in (1, 2, 3, 1 / 2, 1 / 3)], axis=0).mean()
    return acc1, acc2, (abs(r / 2 - 1) <= 0.04).mean(), (abs(r * 2 - 1) <= 0.04).mean()


def main():
    dp = pd.read_csv(os.path.join(RESULTS, "dp_gtzan.csv"))
    v = pd.read_csv(os.path.join(RESULTS, "dp_validation.csv"))
    beats = pd.concat([dp[dp.method == "dp_tight100"], v[v.metric.isin(["F", "CMLt", "AMLt"])]])
    print(beats.pivot_table(index=["method", "condition"], columns="metric", values="value").round(3), "\n")
    a = v[v.metric.str.startswith("agreement")]
    print(a.groupby(["condition", "metric"]).value.agg(mean="mean", median="median", share_ge_095=lambda s: (s >= 0.95).mean()).round(3), "\n")
    print((v[v.metric == "seconds"].groupby("method").value.mean() * 2).round(3).rename("seconds per minute of audio"), "\n")
    b = v[v.metric == "bpm"].pivot_table(index="track", columns="method", values="value")
    for m in ("dp_ellis_tempo", "librosa_tempo"):
        print(m, "Acc1 %.3f Acc2 %.3f double %.3f half %.3f" % tempo_accuracy(b[m], b["ref"]))
    g = pd.concat([dp[(dp.method == "dp_tight100") & (dp.metric == "F")], v[(v.method == "librosa_beat_track") & (v.metric == "F")]])
    print("\n", g.pivot_table(index="genre", columns=["method", "condition"], values="value").round(3))


if __name__ == "__main__":
    main()
