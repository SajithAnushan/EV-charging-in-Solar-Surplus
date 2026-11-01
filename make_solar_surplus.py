import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
EXCESS_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential\Solar_Excess_vs_Mean.xlsx"
BESS_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\bess_2026-09-09_to_2026-09-24.xlsx"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\EV_Framework\data"
OUTPUT_NAME = "solar_surplus.csv"
WRITE_CSV = True                      # also write solar_surplus.csv (same content)

EXCESS_SHEET = "Average_daily_curve"
EXCESS_COLUMN = "Average, all days (MW)"
DAY_START = "06:30"                   # BESS window, inclusive
DAY_END = "18:30"
# =============================================================================

SLOTS_15 = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, 15)]


def hhmm(c):
    """Normalise a column/row label to 'HH:MM' ('24:00' -> '00:00')."""
    if hasattr(c, "strftime"):
        t = c.strftime("%H:%M")
    else:
        t = str(c).strip()
        if " " in t:                                  # e.g. '1900-01-01 00:15:00'
            t = t.split(" ")[-1]
        h, m = t.split(":")[:2]
        t = f"{int(h):02d}:{int(m):02d}"
    return "00:00" if t == "24:00" else t


def read_surplus(path):
    df = pd.read_excel(path, sheet_name=EXCESS_SHEET)
    if EXCESS_COLUMN not in df.columns:
        raise ValueError(f"Column '{EXCESS_COLUMN}' not found in {EXCESS_SHEET}")
    rows = {}
    for t, v in zip(df.iloc[:, 0], df[EXCESS_COLUMN]):
        try:
            rows[hhmm(t)] = float(v)
        except (ValueError, TypeError):
            continue                                  # blank row, 'Energy (MWh/day)' row
    missing = [s for s in SLOTS_15 if s not in rows]
    if missing:
        raise ValueError(f"Surplus curve is missing timestamps, e.g. {missing[:3]}")
    return pd.Series(rows)[SLOTS_15]


def read_bess(path):
    df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")      # drops 'Average' and blank rows
    df.columns = [hhmm(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    missing = [s for s in SLOTS_15 if s not in df.columns]
    if missing:
        raise ValueError(f"BESS file is missing timestamps, e.g. {missing[:3]}")
    return df[SLOTS_15]


def main():
    exc_path = Path(sys.argv[1] if len(sys.argv) > 1 else EXCESS_FILE)
    bess_path = Path(sys.argv[2] if len(sys.argv) > 2 else BESS_FILE)
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR)
    for p in (exc_path, bess_path):
        if not p.is_file():
            print(f"File not found: {p}")
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {exc_path.name} ({EXCESS_COLUMN}) ...")
    surplus = read_surplus(exc_path)

    print(f"Reading {bess_path.name} ...")
    bess = read_bess(bess_path)
    avg = bess.mean(axis=0, skipna=True).fillna(0.0)
    inside = np.array([DAY_START <= t <= DAY_END for t in SLOTS_15])
    dispatch = np.where(inside & (avg.values < 0), -avg.values, 0.0)

    out = pd.DataFrame({
        "time": [f"{int(t[:2])}:{t[3:]}" for t in SLOTS_15],     # 0:00, 0:15, ... as in the sample
        "surplus_MW": surplus.values.round(2),
        "bess_dispatch_MW": dispatch.round(4),
    })

    path = out_dir / OUTPUT_NAME
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="solar_surplus", index=False)
        ws = xw.sheets["solar_surplus"]
        ws.freeze_panes = "A2"
        for c, w in zip("ABC", (8, 14, 18)):
            ws.column_dimensions[c].width = w
    if WRITE_CSV:
        out.to_csv(path.with_suffix(".csv"), index=False)

    print(f"  {len(bess)} BESS days, {bess.index.min():%d %b} to {bess.index.max():%d %b}")
    print(f"  surplus energy : {out['surplus_MW'].sum() / 4:,.1f} MWh per average day")
    print(f"  BESS charging  : {out['bess_dispatch_MW'].sum() / 4:,.2f} MWh per average day "
          f"({DAY_START}-{DAY_END}), peak {out['bess_dispatch_MW'].max():.3f} MW")
    print(f"Done. Written {path}" + (f" and {path.with_suffix('.csv').name}" if WRITE_CSV else ""))


if __name__ == "__main__":
    main()
