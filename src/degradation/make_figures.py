import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..common import data
from ..common.features import onset_envelope, FPS
from ..dp_tracker.ellis import ellis_tempo, track_beats

OUT = os.path.join(data.ROOT, "figures")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a64"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": GREY, "figure.dpi": 150})
LINES = {"dp_standard_est_tempo": ("standard", BLUE, "o", "-"), "dp_percussive_est_tempo": ("percussive", ORANGE, "s", "-"),
         "dp_lowband_est_tempo": ("low band", AQUA, "^", "-"), "dp_standard_oracle_tempo": ("standard, oracle tempo", GREY, "o", "--")}


def read(name):
    return pd.read_csv(os.path.join(data.RESULTS, name))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def degradation():
    d = read("degradation_full.csv")
    f = d[d.metric == "F"].pivot_table(index="condition", columns="method", values="value")
    panels = (("Pink noise SNR (dB)", ["clean", "noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB", "noise_-10dB"], ["clean", "20", "10", "5", "0", "-10"]),
              ("Low pass cutoff (kHz)", ["clean", "lowpass_4k", "lowpass_2k", "lowpass_1k"], ["clean", "4", "2", "1"]),
              ("MP3 bit rate (kbps)", ["clean", "mp3_128k", "mp3_64k", "mp3_32k"], ["clean", "128", "64", "32"]))
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True, gridspec_kw={"width_ratios": [6, 4, 4]})
    for ax, (title, conds, labels) in zip(axes, panels):
        x = np.arange(len(conds))
        for m, (lab, c, mk, ls) in LINES.items():
            ax.plot(x, f.loc[conds, m], color=c, marker=mk, ls=ls, lw=1.5, ms=4, label=lab)
        ax.set_xticks(x, labels)
        ax.set_xlabel(title)
    axes[0].set_ylabel("Beat F-measure")
    axes[0].set_ylim(0.4, 0.9)
    axes[0].legend(loc="lower left", frameon=False, fontsize=7)
    save(fig, "fig_nalluraj_degradation")


def genres():
    dp = read("dp_gtzan.csv")
    v = read("dp_validation.csv")
    g = pd.concat([dp[(dp.method == "dp_tight100") & (dp.metric == "F")], v[(v.method == "librosa_beat_track") & (v.metric == "F")]])
    g["key"] = g.method + "/" + g.condition
    t = g.pivot_table(index="genre", columns="key", values="value")
    series = (("dp_tight100/est_tempo", "DP, estimated tempo", BLUE), ("dp_tight100/oracle_tempo", "DP, oracle tempo", ORANGE),
              ("librosa_beat_track/default", "librosa beat_track", AQUA))
    fig, ax = plt.subplots(figsize=(7, 2.2))
    x = np.arange(len(t))
    for i, (k, lab, c) in enumerate(series):
        ax.bar(x + (i - 1) * 0.27, t[k], 0.25, color=c, label=lab, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, t.index)
    ax.set_ylabel("Beat F-measure")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=7)
    save(fig, "fig_nalluraj_genre")


def tightness():
    dp = read("dp_gtzan.csv")
    f = dp[(dp.metric == "F") & (dp.condition == "est_tempo")].copy()
    f["alpha"] = f.method.str.replace("dp_tight", "").astype(int)
    t = f.pivot_table(index="alpha", columns="genre", values="value")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for gname in t.columns:
        ax.plot(t.index, t[gname], color="#c9c8c1", lw=1)
    for gname in ("disco", "classical"):
        ax.plot(t.index, t[gname], color=GREY, lw=1.2)
        ax.annotate(gname, (t.index[-1], t[gname].iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=GREY)
    ax.plot(t.index, t.mean(axis=1), color=BLUE, lw=2, marker="o", ms=4)
    ax.annotate("mean", (t.index[-1], t.mean(axis=1).iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel(r"Tightness $\alpha$")
    ax.set_ylabel("Beat F-measure")
    save(fig, "fig_nalluraj_tightness")


def example(track="blues.00012"):
    ref, _ = data.load_beats(track)
    env = onset_envelope(data.load_audio(data.audio_path(track)))
    est, ora = track_beats(env, ellis_tempo(env), 100), track_beats(env, data.ref_tempo(ref), 100)
    n = np.arange(len(env)) / FPS
    fig, ax = plt.subplots(figsize=(7, 1.9))
    m = (n >= 10) & (n <= 16)
    ax.plot(n[m], env[m], color=GREY, lw=0.8)
    top = env[m].max()
    for b, y, c, lab in ((ref, 1.08, "black", "annotated"), (ora, 1.18, ORANGE, f"DP, oracle {data.ref_tempo(ref):.0f} BPM"), (est, 1.28, BLUE, f"DP, estimated {ellis_tempo(env):.0f} BPM")):
        b = b[(b >= 10) & (b <= 16)]
        ax.vlines(b, top * (y - 0.05), top * (y + 0.04), color=c, lw=1.6)
        ax.text(16.1, top * y, lab, va="center", fontsize=7, color="black")
    ax.set_xlim(10, 16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset strength")
    ax.set_yticks([])
    save(fig, "fig_nalluraj_example")


def genre_tightness():
    dp = read("dp_gtzan.csv")
    v = read("dp_validation.csv")
    g = pd.concat([dp[(dp.method == "dp_tight100") & (dp.metric == "F")], v[(v.method == "librosa_beat_track") & (v.metric == "F")]])
    g["key"] = g.method + "/" + g.condition
    t = g.pivot_table(index="genre", columns="key", values="value")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 2.1), gridspec_kw={"width_ratios": [2.6, 1]})
    x = np.arange(len(t))
    for i, (k, lab, c) in enumerate((("dp_tight100/est_tempo", "DP, estimated tempo", BLUE), ("dp_tight100/oracle_tempo", "DP, oracle tempo", ORANGE),
                                     ("librosa_beat_track/default", "librosa beat_track", AQUA))):
        ax.bar(x + (i - 1) * 0.27, t[k], 0.25, color=c, label=lab, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, t.index, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Beat F-measure")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=7)
    f = dp[(dp.metric == "F") & (dp.condition == "est_tempo")].copy()
    f["alpha"] = f.method.str.replace("dp_tight", "").astype(int)
    s = f.pivot_table(index="alpha", columns="genre", values="value")
    for gname in s.columns:
        bx.plot(s.index, s[gname], color="#c9c8c1", lw=1)
    for gname in ("disco", "classical"):
        bx.plot(s.index, s[gname], color=GREY, lw=1.2)
        bx.annotate(gname, (s.index[-1], s[gname].iloc[-1]), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5, color=GREY, annotation_clip=False)
    bx.plot(s.index, s.mean(axis=1), color=BLUE, lw=2, marker="o", ms=3.5)
    bx.annotate("mean", (s.index[-1], s.mean(axis=1).iloc[-1]), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5, color=BLUE)
    bx.set_xscale("log")
    bx.set_xlabel(r"Tightness $\alpha$")
    bx.set_ylabel("Beat F-measure")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_nalluraj_genre_tightness")


def main():
    genre_tightness()
    degradation()
    genres()
    tightness()
    example()


if __name__ == "__main__":
    main()
