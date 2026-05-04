from pathlib import Path
import pandas as pd
from HGM import HGMST
from utils import preprocess

base = Path("Data/1.DLPFC")
slices = sorted([str(p).replace("\\", "/") + "/" for p in base.iterdir() if p.is_dir()])
test_slices=[slices[i] for i in [0,7,-4,-3,-2,-1]]

default_k1 = 8
default_k2 = 8
k1_candidates = [4, 8, 12]
k2_candidates = [6, 8, 10, 12, 16, 20]

all_records = []
for seed in [2, 100, 2020, 2026, 10086]:
    # print(f"\n===== Running seed={seed} =====")
    for path in slices:
    # for path in test_slices:
        # print(f"  -> {path} | baseline k1={default_k1}, k2={default_k2}")
        adata1 = preprocess(path)
        # hgm = HGMST(
        #     adata=adata1,
        #     k1=default_k1,
        #     k2=default_k2,
        #     seed=seed,
        # )
        # hgm.train(epochs=100)
        # _, res_df = hgm.eval()

        # m = res_df.loc["mclust", ["ARI", "NMI", "FMI"]]
        # all_records.append({
        #     "mode": "default",
        #     "seed": seed,
        #     "slice": path.split("/")[-2],
        #     "k1": default_k1,
        #     "k2": default_k2,
        #     "ARI": float(m["ARI"]),
        #     "NMI": float(m["NMI"]),
        #     "FMI": float(m["FMI"]),
        # })

        for k1 in k1_candidates:
            for k2 in k2_candidates:
                # print(f"  -> {path} | k1={k1}, k2={k2}")
                hgm = HGMST(
                    adata=adata1,
                    k1=k1,
                    k2=k2,
                    seed=seed,
                )
                hgm.train(epochs=100)
                _, res_df = hgm.eval()

                m = res_df.loc["mclust", ["ARI", "NMI", "FMI"]]
                all_records.append({
                    "mode": "grid",
                    "seed": seed,
                    "slice": path.split("/")[-2],
                    "k1": k1,
                    "k2": k2,
                    "ARI": float(m["ARI"]),
                    "NMI": float(m["NMI"]),
                    "FMI": float(m["FMI"]),
                })

results = pd.DataFrame(all_records).sort_values(["mode", "k1", "k2", "seed", "slice"]).reset_index(drop=True)

# baseline_summary = (
#     results[results["mode"] == "default"]
#     .groupby("seed", as_index=False)[["ARI", "NMI", "FMI"]]
#     .mean()
#     .sort_values("seed")
#     .reset_index(drop=True)
# )

grid_summary = (
    results[results["mode"] == "grid"]
    .groupby(["k1", "k2"])[["ARI", "NMI", "FMI"]]
    .agg(["mean", "std", "max", "min", "median"])
)
grid_summary = grid_summary.reset_index()
grid_summary.columns = [
    col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
    for col in grid_summary.columns.to_flat_index()
 ]
grid_summary = grid_summary.sort_values(["ARI_mean", "NMI_mean", "FMI_mean"], ascending=False).reset_index(drop=True)

# print("Baseline (default k1=k2=8) per-seed means:")
# print(baseline_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nGrid search summary across all seed x slice runs (sorted by ARI_mean):")
print(grid_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))