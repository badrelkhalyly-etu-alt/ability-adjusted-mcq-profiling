"""Extract external (non-cognitive) validation variables for students in the analytic forms."""
import pandas as pd, pyreadstat
from pathlib import Path
P = Path("data/processed")
COLS = ["CNT", "CNTSTUID", "ST004D01T", "ST001D01T", "ESCS", "IMMIG", "EFFORT1", "EFFORT2", "PERSEV", "W_FSTUWT"]
ids = pd.concat([pd.read_parquet(P / f"math_form{x}.parquet", columns=["CNT", "CNTSTUID"]) for x in "ABC"])
keys = set(zip(ids.CNT, ids.CNTSTUID))
parts = []
for df, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_sav, "data/raw/CY09_MS_STU_PUF.sav", chunksize=60000,
                                            usecols=COLS, encoding="latin1", user_missing=False):
    parts.append(df[[k in keys for k in zip(df.CNT, df.CNTSTUID)]])
out = pd.concat(parts, ignore_index=True)
for c in ["ST004D01T", "IMMIG", "EFFORT1", "EFFORT2"]:
    out.loc[out[c] >= 95, c] = float("nan")
out.to_parquet(P / "student_external.parquet")
print(out.shape, out.notna().mean().round(3).to_dict())
