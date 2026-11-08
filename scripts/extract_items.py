"""Extract scored science/math items (computer-based, linear paths) and their item timing.

Codes in the PUF scored variables are strings: '0','1','2' credit; '7' not applicable
(not administered); 'r' not reached; 't' technical issue. They are kept distinct here:
    0/1/2 -> credit, 'r' -> -8, 't' -> -9, '7'/'' -> NaN (not administered).
Timing variables (``...TT``) in the process file are kept as float; units are
checked in the data audit.
"""
import time
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
COG = RAW / "CY09_MS_COG_20260806.sav"
PRO = RAW / "CY09_MS_COG_PROCESS_20260806.sav"
ID = ["CNT", "CNTSCHID", "CNTSTUID", "OECD", "ADMINMODE", "LANGTEST_COG", "BOOKID",
      "SPATH", "MPATH", "SCORE_ItemSet", "SS1_LEV", "SS2_LEV", "MCORE_ItemSet", "MS1_LEV", "MS2_LEV", "Option_UH"]
CODE = {"0": 0, "1": 1, "2": 2, "r": -8, "t": -9}


def domain_items(names):
    sci = [c for c in names if c.endswith("S") and (c.startswith("CS") or c.startswith("CACERS"))]
    mat = [c for c in names if c.endswith("S") and (c.startswith("CM"))]
    return sci, mat


def main():
    _, m = pyreadstat.read_sav(str(COG), metadataonly=True, encoding="latin1")
    sci, mat = domain_items(m.column_names)
    items = sci + mat
    pd.Series(dict(zip(m.column_names, m.column_labels)))[items].to_csv(OUT / "item_labels.csv", header=["label"])
    t = time.time()
    parts = []
    for df, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_sav, str(COG), chunksize=40000,
                                                usecols=ID + items, encoding="latin1"):
        keep = (df.SPATH == 1) | (df.MPATH == 1)
        df = df.loc[keep]
        out = df[ID].copy()
        it = df[items].apply(lambda s: s.map(CODE)).astype("float32")
        parts.append(pd.concat([out.reset_index(drop=True), it.reset_index(drop=True)], axis=1))
        print("cog", sum(len(p) for p in parts), round(time.time() - t), flush=True)
    cog = pd.concat(parts, ignore_index=True)
    cog.to_parquet(OUT / "cog_linear_items.parquet")

    _, mp = pyreadstat.read_sav(str(PRO), metadataonly=True, encoding="latin1")
    stems = {c[:-1] for c in items}  # e.g. CACERS001Q01
    tcols = [c for c in mp.column_names if c.endswith("TT") and c[:-2] in stems]
    acols = [c for c in mp.column_names if c.endswith("A") and c[:-1] in stems]
    fcols = [c for c in mp.column_names if c.endswith("F") and c[:-1] in stems]
    ids = set(zip(cog.CNT, cog.CNTSTUID))
    parts = []
    t = time.time()
    for df, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_sav, str(PRO), chunksize=40000,
                                                usecols=["CNT", "CNTSTUID"] + tcols + acols + fcols,
                                                encoding="latin1"):
        k = [(a, b) in ids for a, b in zip(df.CNT, df.CNTSTUID)]
        df = df.loc[k]
        num = df[tcols + acols + fcols].apply(pd.to_numeric, errors="coerce").astype("float32")
        parts.append(pd.concat([df[["CNT", "CNTSTUID"]].reset_index(drop=True), num.reset_index(drop=True)], axis=1))
        print("proc", sum(len(p) for p in parts), round(time.time() - t), flush=True)
    pro = pd.concat(parts, ignore_index=True)
    pro.to_parquet(OUT / "process_linear_items.parquet")
    pd.Series(dict(zip(mp.column_names, mp.column_labels)))[tcols[:5]].to_csv(OUT / "timing_label_examples.csv")
    print("done")


if __name__ == "__main__":
    main()
