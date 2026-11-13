import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "results", "tempo_gtzan.csv"))
for m in ("acc1", "acc2"):
    x = d[d.metric == m].copy()
    x["value"] = x.value.astype(float)
    print(m, (x.groupby("method").value.mean() * 100).round(1).to_dict())
print(d[d.metric == "octave"].groupby(["method", "value"]).size().unstack(fill_value=0))
