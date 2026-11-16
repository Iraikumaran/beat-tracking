import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
F = os.environ.get("FIG_DIR", os.path.join(ROOT, "figures"))
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
macros = {}


def load(name):
    return pd.read_csv(os.path.join(R, name))


def has(name):
    return os.path.exists(os.path.join(R, name))


def num(d, metric):
    d = d[d.metric == metric].copy()
    d["value"] = d.value.astype(float)
    return d


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(F, name))
    plt.close(fig)


def table(name, rows):
    open(os.path.join(F, name), "w").write("\n".join(rows))


def tempo():
    d = load("tempo_gtzan.csv")
    if has("tempo_compare.csv"):
        d = pd.concat([d, load("tempo_compare.csv")])
    t = num(d, "acc1").groupby("method").value.mean() * 100
    t2 = num(d, "acc2").groupby("method").value.mean() * 100
    oc = d[d.metric == "octave"].groupby(["method", "value"]).size().unstack(fill_value=0)
    sec = num(d, "seconds").groupby("method").value.mean() * 1000
    n = d.track.nunique()
    macros["NTempo"] = n
    rows = []
    names = (("ioi_hist", "IOI histogram"), ("autocorr", "Autocorr. (PI1)"), ("autocorr+pulse", "+ pulse trains (PI3)"),
             ("autocorr+prior", "+ octave decision (PI5)"), ("librosa", "librosa"), ("madmom", "madmom"))
    for m, name in names:
        if m in t:
            o = oc.loc[m]
            rows.append(f"{name} & {t[m]:.1f} & {t2[m]:.1f} & {100*o.get('half',0)/n:.1f} & {100*o.get('double',0)/n:.1f} & {sec[m]:.1f} \\\\")
    table("tab_tempo.tex", rows)
    for key, m in (("AC", "autocorr"), ("Pulse", "autocorr+pulse"), ("Prior", "autocorr+prior"), ("IOI", "ioi_hist"), ("Librosa", "librosa"), ("Madmom", "madmom")):
        macros[f"AccOne{key}"] = f"{t.get(m, np.nan):.1f}"
        macros[f"AccTwo{key}"] = f"{t2.get(m, np.nan):.1f}"
    macros["HalfAC"] = f"{100*oc.loc['autocorr'].get('half',0)/n:.1f}"
    macros["MadmomMs"] = f"{sec.get('madmom', np.nan)/1000:.1f}"
    macros["LibrosaMs"] = f"{sec.get('librosa', np.nan):.0f}"
    a = num(d, "acc1")
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    i = np.arange(len(GENRES))
    ms = [(m, lab, c) for m, lab, c in (("autocorr", "autocorr.", "0.85"), ("autocorr+prior", "+ octave dec.", "0.5"), ("madmom", "madmom", "0.15")) if m in t]
    w = 0.8 / len(ms)
    for k, (m, lab, c) in enumerate(ms):
        g = a[a.method == m].groupby("genre").value.mean() * 100
        ax.bar(i + (k - (len(ms) - 1) / 2) * w, [g.get(x, np.nan) for x in GENRES], w, color=c, label=lab, edgecolor="k", linewidth=0.3)
    ax.set_xticks(i); ax.set_xticklabels([g[:4] for g in GENRES], rotation=45)
    ax.set_ylabel("Acc1 (%)"); ax.set_ylim(0, 125); ax.set_yticks([0, 50, 100])
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper center")
    save(fig, "fig_tempo_genre.pdf")


def tempo_change():
    d = load("tempo_change.csv")
    a = num(d[d.method == "autocorr+prior"], "acc1")
    a2 = num(d[d.method == "autocorr+prior"], "acc2")
    fac = sorted(a.condition.unique(), key=lambda c: float(c.split("_")[1]))
    x = [float(c.split("_")[1]) for c in fac]
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.5))
    ax[0].plot(x, [a[a.condition == c].value.mean() * 100 for c in fac], "o-", label="Acc1", color="k")
    ax[0].plot(x, [a2[a2.condition == c].value.mean() * 100 for c in fac], "s--", label="Acc2", color="0.5")
    ax[0].set_xlabel("stretch factor"); ax[0].set_ylabel("acc. (%)"); ax[0].set_ylim(0, 100); ax[0].legend(frameon=False, fontsize=6)
    loc = num(d[d.method.str.startswith("local_")], "rel_err")
    wins = sorted(loc.method.unique(), key=lambda m: int(m.split("_")[1][:-1]))
    med = {w: loc[loc.method == w].value.median() * 100 for w in wins}
    ax[1].bar([w.split("_")[1] for w in wins], [med[w] for w in wins], color="0.6")
    ax[1].set_xlabel("window length"); ax[1].set_ylabel("rel. error (%)")
    save(fig, "fig_tempo_change.pdf")
    macros["AccOneStretchMin"] = f"{min(a[a.condition == c].value.mean() * 100 for c in fac):.1f}"
    macros["AccOneStretchMax"] = f"{max(a[a.condition == c].value.mean() * 100 for c in fac):.1f}"
    macros["RelErrFour"] = f"{med.get('local_4s', np.nan):.1f}"
    macros["RelErrTwelve"] = f"{med.get('local_12s', np.nan):.1f}"
    sp = d[d.condition.str.startswith("splice")]
    macros["SpliceMAE"] = f"{num(sp, 'mae_bpm').value.median():.1f}"
    st = num(sp, "settle_s").value
    macros["SettleMed"] = f"{st.median():.1f}"
    macros["SettleFrac"] = f"{100*st.notna().mean():.0f}"
    macros["NChange"] = d.track.nunique()
    if has("tempo_change_detect.csv"):
        e = num(load("tempo_change_detect.csv"), "abs_err_s")
        for key, m in (("Fold", "change_folded"), ("Raw", "change_raw")):
            v = e[e.method == m].value
            macros[f"Det{key}Med"] = f"{v.median():.1f}"
            macros[f"Det{key}Two"] = f"{100*(v <= 2).mean():.0f}"


def splice_example():
    sys.path.insert(0, ROOT)
    from src.common import data
    from src.common.features import onset_envelope
    from src.tempo_change.stretch import splice
    from src.tempo_change.local_tempo import local_tempo
    from src.tempo_change.detect import detect_change
    t = "rock.00004" if "rock.00004" in data.split("test") else data.split("test")[-1]
    x = data.load_audio(data.audio_path(t))
    ref = data.ref_tempo(data.load_beats(t)[0])
    y, cut = splice(x, 0.85, 1.15, 15.0)
    env = onset_envelope(y)
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    for w, c in ((4.0, "0.65"), (12.0, "0.35")):
        times, bpms = local_tempo(env, w)
        ax.plot(times, bpms, "o-", ms=2, lw=0.8, color=c, label=f"{int(w)} s window")
    times, bpms = local_tempo(env, 6.0)
    ax.plot(times, bpms, "o-", ms=2, lw=1, color="k", label="6 s window")
    tt = np.linspace(0, len(y) / data.SR, 200)
    ax.plot(tt, np.where(tt < cut, ref * 0.85, ref * 1.15), ls="--", color="r", lw=0.8, label="true tempo")
    ax.axvline(detect_change(times, bpms), color="b", ls=":", lw=0.9, label="detected change")
    ax.set_xlabel(f"time (s), {t}"); ax.set_ylabel("BPM")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(os.path.join(F, "fig_splice.pdf"))
    plt.close(fig)


def beats():
    dp = load("dp_gtzan.csv")
    est = num(dp[dp.condition == "est_tempo"], "F")
    tights = sorted({int(m[8:]) for m in est.method})
    best = max(tights, key=lambda k: est[est.method == f"dp_tight{k}"].value.mean())
    macros["BestTight"] = best
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    ax.plot(tights, [est[est.method == f"dp_tight{k}"].value.mean() for k in tights], "o-", color="k")
    ax.set_xscale("log"); ax.set_xlabel("tightness $\\alpha$"); ax.set_ylabel("mean F")
    save(fig, "fig_tightness.pdf")
    sources = [("DP, estimated tempo", dp[(dp.method == "dp_tight100") & (dp.condition == "est_tempo")], "0.2"),
               ("DP, oracle tempo", dp[(dp.method == "dp_tight100") & (dp.condition == "oracle_tempo")], None)]
    if has("dp_validation.csv"):
        v = load("dp_validation.csv")
        sources.append(("librosa beat\\_track", v[v.method == "librosa_beat_track"], None))
    nnm = load("nn_madmom.csv") if has("nn_madmom.csv") else None
    if nnm is not None:
        sources.append(("madmom RNN + DBN", nnm[nnm.method == "madmom_rnn_dbn"], "0.85"))
    nn = load("nn_gtzan.csv")
    sources.append(("MLP + peak picking", nn, "0.55"))
    if nnm is not None:
        sources.append(("MLP + DBN", nnm[nnm.method == "mlp_dbn"], "1.0"))
    rows, gen = [], {}
    for lab, fr, col in sources:
        vals = [num(fr, m).value.mean() for m in ("F", "CMLt", "AMLt")]
        rows.append(f"{lab} & " + " & ".join(f"{x:.3f}" for x in vals) + " \\\\")
        gen[lab] = (num(fr, "F").groupby("genre").value.mean(), col, vals)
    table("tab_beats.tex", rows)
    key = {"DP, estimated tempo": "DP", "DP, oracle tempo": "DPOracle", "librosa beat\\_track": "Lib", "madmom RNN + DBN": "Mad",
           "MLP + peak picking": "NN", "MLP + DBN": "NNDBN"}
    for lab, (_, _, vals) in gen.items():
        macros[f"{key[lab]}F"], macros[f"{key[lab]}CML"], macros[f"{key[lab]}AML"] = (f"{x:.2f}" for x in vals)
    macros["NBeat"] = nn.track.nunique()
    plot = [(lab, g, c) for lab, (g, c, _) in gen.items() if c is not None]
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    i = np.arange(len(GENRES)); w = 0.8 / len(plot)
    for k, (lab, g, c) in enumerate(plot):
        ax.bar(i + (k - (len(plot) - 1) / 2) * w, [g.get(x, np.nan) for x in GENRES], w, label=lab.replace("\\", ""), color=c, edgecolor="k", linewidth=0.3)
    ax.set_xticks(i); ax.set_xticklabels([g[:4] for g in GENRES], rotation=45)
    ax.set_ylabel("F-measure"); ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper center")
    save(fig, "fig_genre_f.pdf")
    if has("dp_validation.csv"):
        v = load("dp_validation.csv")
        ag = num(v[(v.method == "dp_vs_librosa") & (v.condition == "same_env_same_tempo_smoothed")], "agreement_F")
        macros["AgreeMed"] = f"{ag.value.median():.2f}"
        macros["AgreeShare"] = f"{100*(ag.value >= 0.95).mean():.0f}"
        ag0 = num(v[(v.method == "dp_vs_librosa") & (v.condition == "same_env_same_tempo")], "agreement_F")
        macros["AgreeRawMed"] = f"{ag0.value.median():.2f}"
        sec = num(v, "seconds")
        macros["LibBeatTime"] = f"{(sec[sec.method == 'librosa_beat_track'].value.mean() * 2):.2f}"
    if nnm is not None:
        s = num(nnm, "sec_per_min").groupby("method").value.mean()
        macros["MadTime"] = f"{s['madmom_rnn_dbn']:.1f}"
        macros["NNDBNTime"] = f"{s['mlp_dbn']:.2f}"
    tm = dict(l.strip().split("=") for l in open(os.path.join(R, "timing.txt")) if "=" in l)
    macros["MLPTime"] = f"{float(tm['mlp_s_per_min']):.2f}"
    macros["DPTime"] = f"{float(tm['dp_s_per_min']):.2f}"


DEG = ["clean", "noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB", "noise_-10dB", "white_0dB",
       "lowpass_4k", "lowpass_2k", "lowpass_1k", "mp3_128k", "mp3_64k", "mp3_32k"]
DEG_LAB = ["clean", "N20", "N10", "N5", "N0", "N-10", "W0", "LP4k", "LP2k", "LP1k", "M128", "M64", "M32"]


def degradation():
    d = num(load("degradation.csv"), "F")
    sets = [("DP (spectral flux)", d[d.method == "dp_est_tempo"], "0.2")]
    if has("degradation_percussive.csv"):
        p = num(load("degradation_percussive.csv"), "F")
        sets.append(("DP (percussive)", p[p.method == p.method.iloc[0]] if p.method.nunique() == 1 else p[p.method.str.contains("est")], "0.55"))
    if has("degradation_madmom.csv"):
        sets.append(("madmom", num(load("degradation_madmom.csv"), "F"), "0.88"))
    order = [c for c in DEG if c in set(d.condition)]
    labels = [DEG_LAB[DEG.index(c)] for c in order]
    fig, ax = plt.subplots(figsize=(3.4, 1.65))
    i = np.arange(len(order)); w = 0.8 / len(sets)
    means = {}
    for k, (lab, fr, c) in enumerate(sets):
        g = fr.groupby("condition").value.mean()
        means[lab] = g
        ax.bar(i + (k - (len(sets) - 1) / 2) * w, [g.get(x, np.nan) for x in order], w, color=c, label=lab, edgecolor="k", linewidth=0.3)
    ax.set_xticks(i); ax.set_xticklabels(labels, rotation=60, fontsize=6)
    ax.set_ylabel("F-measure"); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center")
    save(fig, "fig_degradation.pdf")
    rows = []
    for c, lab in zip(order, labels):
        rows.append(f"{lab} & " + " & ".join(f"{means[s].get(c, np.nan):.2f}" for s, _, _ in sets) + " \\\\")
    table("tab_degradation.tex", rows)
    e = means["DP (spectral flux)"]
    o = d[d.method == "dp_oracle_tempo"].groupby("condition").value.mean()
    for c, k in (("clean", "Clean"), ("noise_0dB", "NoiseZero"), ("noise_-10dB", "NoiseMinusTen"), ("white_0dB", "White"),
                 ("lowpass_1k", "LPOne"), ("lowpass_4k", "LPFour"), ("mp3_32k", "MPThree")):
        if c in e:
            macros[f"Deg{k}"] = f"{e[c]:.2f}"
            macros[f"DegOr{k}"] = f"{o[c]:.2f}"
            for s, pre in (("DP (percussive)", "Perc"), ("madmom", "MadDeg")):
                if s in means:
                    macros[f"{pre}{k}"] = f"{means[s][c]:.2f}"
    macros["NDeg"] = d.track.nunique()


def downbeat():
    d = num(load("downbeat_gtzan.csv"), "F")
    m = num(load("downbeat_madmom.csv"), "F") if has("downbeat_madmom.csv") else None
    cell = lambda fr: f"{fr.value.mean():.3f}" if len(fr) else "--"
    rows = []
    a = d[d.condition == "annotated_beats"]
    rows.append("annotated beats & " + cell(a[a.method == "first_beat"]) + " & " + cell(a[a.method == "low_onset_finder"]) + " & " +
                (cell(m[m.method == "madmom_bar"]) if m is not None else "--") + " \\\\")
    b = d[d.condition == "nn_beats"]
    rows.append("MLP beats & " + cell(b[b.method == "first_beat"]) + " & " + cell(b[b.method == "low_onset_finder"]) + " & -- \\\\")
    if m is not None:
        rows.append("audio (joint model) & -- & -- & " + cell(m[m.method == "madmom_joint"]) + " \\\\")
    table("tab_downbeat.tex", rows)
    s = a[a.method == "low_onset_finder"]
    macros["DBOracle"] = f"{s.value.mean():.2f}"; macros["NDown"] = s.track.nunique()
    macros["DBFirst"] = f"{a[a.method == 'first_beat'].value.mean():.2f}"
    macros["DBNN"] = f"{b[b.method == 'low_onset_finder'].value.mean():.2f}"
    macros["DBNNFirst"] = f"{b[b.method == 'first_beat'].value.mean():.2f}"
    if m is not None:
        macros["DBMadJoint"] = f"{m[m.method == 'madmom_joint'].value.mean():.2f}"
        macros["DBMadBar"] = f"{m[m.method == 'madmom_bar'].value.mean():.2f}"
    sets = [("first beat", a[a.method == "first_beat"], "0.85"), ("low-freq. finder", s, "0.5")]
    if m is not None:
        sets.append(("madmom bar tracker", m[m.method == "madmom_bar"], "0.15"))
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    i = np.arange(len(GENRES)); w = 0.8 / len(sets)
    for k, (lab, fr, c) in enumerate(sets):
        g = fr.groupby("genre").value.mean()
        ax.bar(i + (k - (len(sets) - 1) / 2) * w, [g.get(x, np.nan) for x in GENRES], w, color=c, label=lab, edgecolor="k", linewidth=0.3)
    ax.set_xticks(i); ax.set_xticklabels([g[:4] for g in GENRES], rotation=45)
    ax.set_ylabel("downbeat F"); ax.set_ylim(0, 1.35); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper center")
    save(fig, "fig_downbeat_genre.pdf")


def example():
    sys.path.insert(0, ROOT)
    import pickle
    from src.common import data
    from src.tcn.model import frame_features
    from src.common.features import onset_envelope, FPS
    t = data.split("test")[0]
    x = data.load_audio(data.audio_path(t))
    ref, _ = data.load_beats(t)
    act = pickle.load(open(os.path.join(R, "mlp_beat.pkl"), "rb")).predict_proba(frame_features(x))[:, 1]
    env = onset_envelope(x)
    ts = np.arange(len(env)) / FPS
    m = (ts > 10) & (ts < 16)
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    ax.plot(ts[m], env[m] / env[m].max(), color="0.6", lw=0.7, label="onset envelope")
    ax.plot(ts[m], act[m], color="k", lw=0.9, label="MLP beat activation")
    for b in ref[(ref > 10) & (ref < 16)]:
        ax.axvline(b, color="k", ls=":", lw=0.6)
    ax.set_xlabel(f"time (s), {t}"); ax.set_ylim(0, 1.6); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center")
    save(fig, "fig_example.pdf")


for fn in (tempo, tempo_change, splice_example, beats, degradation, downbeat, example):
    try:
        fn()
    except Exception as e:
        print("skipped", fn.__name__, repr(e))
with open(os.path.join(F, "macros.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(macros)
