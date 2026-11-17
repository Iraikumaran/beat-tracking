import os
import pandas as pd
from ..common.data import RESULTS

ORDER = ["clean", "noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB", "noise_-10dB", "white_0dB",
         "lowpass_4k", "lowpass_2k", "lowpass_1k", "mp3_128k", "mp3_64k", "mp3_32k"]


def main():
    d = pd.read_csv(os.path.join(RESULTS, "degradation_full.csv"))
    for metric in ("F", "AMLt"):
        t = d[d.metric == metric].pivot_table(index="condition", columns="method", values="value").loc[ORDER]
        print(metric, "\n", t.round(3).to_string(), "\n")
    f = d[d.metric == "F"].pivot_table(index="condition", columns="method", values="value")
    base = f.loc["clean", "dp_standard_est_tempo"]
    print("Change in F vs. standard envelope under each condition (estimated tempo)")
    for c in ORDER[1:]:
        loss = base - f.loc[c, "dp_standard_est_tempo"]
        print(f"{c:12s} loss {loss:+.3f}  percussive {f.loc[c, 'dp_percussive_est_tempo'] - f.loc[c, 'dp_standard_est_tempo']:+.3f}  low band {f.loc[c, 'dp_lowband_est_tempo'] - f.loc[c, 'dp_standard_est_tempo']:+.3f}")
    g = d[(d.metric == "F") & (d.method == "dp_standard_est_tempo")].pivot_table(index="genre", columns="condition", values="value")
    drop = pd.DataFrame({c: g[c] - g["clean"] for c in ("noise_0dB", "lowpass_1k", "mp3_32k")})
    drop["noise_hurts_more_than_lowpass (H3)"] = drop.noise_0dB < drop.lowpass_1k
    print("\nDrop in F per genre (standard envelope, estimated tempo)\n", drop.round(3).to_string())


if __name__ == "__main__":
    main()
