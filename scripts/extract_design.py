"""Extract design/ID columns of the PISA 2025 COG file to parquet (one pass)."""
import sys, time, pyreadstat, pandas as pd
from pathlib import Path

RAW = Path("data/raw/CY09_MS_COG_20260806.sav")
OUT = Path("data/processed/cog_design.parquet")
_, meta = pyreadstat.read_sav(str(RAW), metadataonly=True, encoding="latin1")
design = [c for c in meta.column_names[:60] if not c[:2] in ("CA", "DA")]  # leading design block
scored = [c for c in meta.column_names if c.endswith("S") and c[0] == "C" and c not in design]
t = time.time()
parts = []
for df, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_sav, str(RAW), chunksize=50000,
                                            usecols=design + scored, encoding="latin1"):
    d = df[design].copy()
    # store which scored items are present (non-missing) as a compact sorted tuple string
    nn = df[scored].notna().to_numpy()
    d["n_items"] = nn.sum(1)
    parts.append((d, pd.DataFrame(nn, columns=scored).astype("bool")))
    print(len(parts) * 50000, round(time.time() - t), flush=True)
pd.concat([p[0] for p in parts], ignore_index=True).to_parquet(OUT)
pd.concat([p[1] for p in parts], ignore_index=True).to_parquet(OUT.with_name("cog_presence.parquet"))
pd.Series(scored).to_csv(OUT.with_name("cog_scored_items.csv"), index=False)
print("done", round(time.time() - t))
