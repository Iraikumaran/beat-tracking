import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frames = [pd.read_csv(os.path.join(ROOT, "results", f)) for f in ("tempo_gtzan.csv", "tempo_compare.csv")
          if os.path.exists(os.path.join(ROOT, "results", f))]
d = pd.concat(frames)
for m in ("acc1", "acc2"):
    x = d[d.metric == m].copy()
    x["value"] = x.value.astype(float)
    print(m, (x.groupby("method").value.mean() * 100).round(1).to_dict())
s = d[d.metric == "seconds"].copy()
s["value"] = s.value.astype(float)
print("ms per song", (s.groupby("method").value.mean() * 1000).round(1).to_dict())
print(d[d.metric == "octave"].groupby(["method", "value"]).size().unstack(fill_value=0))
