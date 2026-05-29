from pathlib import Path
import pandas as pd
from HGM import HGMST
from utils import preprocess

base = Path("Data/1.DLPFC")
slices = sorted([str(p).replace("\\", "/") + "/" for p in base.iterdir() if p.is_dir()])
test_slices = [slices[i] for i in [0, 7, -4, -3, -2, -1]]

use_test_slices = False
if use_test_slices:
    slices = test_slices

k1 = 8
k2 = 8
seeds = [2, 100, 2020, 2026, 10086]

dgi_configs = [
    {"tag": "no_dgi", "use_dgi": False, "dgi_weight": 0.0},
    {"tag": "dgi_w0.05", "use_dgi": True, "dgi_weight": 0.05},
    {"tag": "dgi_w0.1", "use_dgi": True, "dgi_weight": 0.1},
]

all_records = []
for seed in seeds:
    for path in slices:
        adata1 = preprocess(path)
        for cfg in dgi_configs:
            hgm = HGMST(
                adata=adata1,
                k1=k1,
                k2=k2,
                seed=seed,
                use_dgi=cfg["use_dgi"],
                dgi_weight=cfg["dgi_weight"],
            )
            hgm.train(epochs=100)
            _, res_df = hgm.eval()

            m = res_df.loc["mclust", ["ARI", "NMI", "FMI"]]
            all_records.append(
                {
                    "mode": cfg["tag"],
                    "seed": seed,
                    "slice": path.split("/")[-2],
                    "k1": k1,
                    "k2": k2,
                    "ARI": float(m["ARI"]),
                    "NMI": float(m["NMI"]),
                    "FMI": float(m["FMI"]),
                }
            )

results = (
    pd.DataFrame(all_records)
    .sort_values(["mode", "seed", "slice"])
    .reset_index(drop=True)
)

summary = (
    results.groupby(["mode"])[["ARI", "NMI", "FMI"]]
    .agg(["mean", "std", "max", "min", "median"])
    .reset_index()
)
summary.columns = [
    col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
    for col in summary.columns.to_flat_index()
]
summary = summary.sort_values(
    ["ARI_mean", "NMI_mean", "FMI_mean"], ascending=False
).reset_index(drop=True)

print("\nDGI ablation summary across all seed x slice runs (sorted by ARI_mean):")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
