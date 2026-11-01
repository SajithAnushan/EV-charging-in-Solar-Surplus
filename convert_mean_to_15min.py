import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
INPUT_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential\Solar_Generation_10min_MW.xlsx"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential"
OUTPUT_NAME = "National_Mean_15min_MW.xlsx"

COLUMN = "National total (MW)"   # mean potential; any MW_10min column works, e.g. "National P50 (MW)"

# What a 15-minute label means in your generation data:
#   "end"   : 12:15 = average over 12:00-12:15  (matches the 2026 generation data)
#   "start" : 12:15 = average over 12:15-12:30
TARGET_LABEL = "end"

OUTPUT_YEAR = 2026             # year printed on the dates (the curve is a typical year)
# =============================================================================

MINUTES_10 = 144
SLOTS_15 = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, 15)]


def read_10min(path):
    """Return the 10-minute series (365*144 values) for COLUMN."""
    df = pd.read_excel(path, sheet_name="MW_10min")
    if COLUMN not in df.columns:
        raise ValueError(f"Column '{COLUMN}' not found. Available: {', '.join(map(str, df.columns[3:]))}")
    s = pd.to_numeric(df[COLUMN], errors="coerce")
    if s.notna().all():
        return s.values.astype(float), "calculated values stored in the workbook"

    # The workbook was never opened and saved in Excel, so formulas have no
    # stored results. Rebuild the value from its inputs instead.
    ghi = pd.read_excel(path, sheet_name="GHI_Wm2")
    inp = pd.read_excel(path, sheet_name="Inputs", header=None)
    pr = float(inp.iat[2, 1])
    g_stc = float(inp.iat[3, 1])
    rows = inp.iloc[6:, [0, 2]].copy()
    rows.columns = ["name", "cap"]
    rows["cap"] = pd.to_numeric(rows["cap"], errors="coerce")
    rows = rows[rows["cap"].notna() & (rows["name"].astype(str) != "Total")]
    total = rows["cap"].sum()

    if COLUMN == "National total (MW)":
        # Mean potential = sum of each station's mean GHI x its own capacity
        v = np.zeros(len(ghi))
        for name, cap in zip(rows["name"], rows["cap"]):
            v += ghi[f"{name} (W/m2)"].values.astype(float) * cap / g_stc * pr
        return v, f"rebuilt from station mean GHI (capacity {total:,.3f} MW, PR {pr})"
    if COLUMN.startswith("National P"):
        g = ghi[COLUMN.replace("(MW)", "(W/m2)")].values.astype(float)
        return g * total / g_stc * pr, f"rebuilt from GHI (capacity {total:,.3f} MW, PR {pr})"
    if COLUMN.replace("(MW)", "(W/m2)") in ghi.columns:          # a single station
        g = ghi[COLUMN.replace("(MW)", "(W/m2)")].values.astype(float)
        cap = rows.loc[rows["name"].astype(str) == COLUMN.replace(" (MW)", ""), "cap"].sum()
        return g * cap / g_stc * pr, f"rebuilt from GHI (capacity {cap:,.3f} MW, PR {pr})"
    raise ValueError("The workbook has no calculated values. Open it in Excel, save it, and run again.")


def to_15min(v10):
    """Exact 15-minute averages from 10-minute averages (labels = interval end)."""
    # 10-min value i covers (10i-10, 10i]; split each into two 5-minute pieces.
    # 5-min piece k then covers (5k-10, 5k-5].
    f = np.repeat(v10, 2)
    if TARGET_LABEL == "end":        # 15-min m covers (15m-15, 15m]  -> pieces 3m-1, 3m, 3m+1
        f = np.roll(f, 1)
    elif TARGET_LABEL == "start":    # 15-min m covers (15m, 15m+15] -> pieces 3m+2, 3m+3, 3m+4
        f = np.roll(f, -2)
    else:
        raise ValueError("TARGET_LABEL must be 'end' or 'start'")
    return f.reshape(-1, 3).mean(axis=1)


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
    if not inp.is_file():
        print(f"Input file not found: {inp}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {inp.name} (MW_10min sheet) ...")
    v10, source = read_10min(inp)
    if len(v10) != 365 * MINUTES_10:
        print(f"Unexpected length {len(v10)} - expected {365 * MINUTES_10} ten-minute values.")
        return
    print(f"  using {source}")

    v15 = to_15min(np.nan_to_num(v10))
    days = pd.date_range(f"{OUTPUT_YEAR}-01-01", periods=365)   # 29 Feb is not in the typical year
    table = pd.DataFrame(v15.reshape(365, 96).round(2), index=days, columns=SLOTS_15)
    table.index.name = "Date"

    e10 = v10.sum() / 6 / 1000
    e15 = v15.sum() / 4 / 1000
    daily = pd.DataFrame({"Energy (MWh)": (table.sum(axis=1) / 4).round(1),
                          "Peak (MW)": table.max(axis=1)}, index=days)
    daily.index.name = "Date"
    notes = pd.DataFrame({"Item": ["Source file", "Column", "15-minute label", "Method",
                                   "Annual energy, 10-min (GWh)", "Annual energy, 15-min (GWh)"],
                          "Value": [inp.name, COLUMN,
                                    "End of interval (12:15 = 12:00-12:15)" if TARGET_LABEL == "end"
                                    else "Start of interval (12:15 = 12:15-12:30)",
                                    "Exact time-weighted average of the 10-minute values",
                                    round(e10, 3), round(e15, 3)]})

    path = out_dir / OUTPUT_NAME
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd") as xw:
        table.to_excel(xw, sheet_name=COLUMN.replace("(MW)", "15min")[:31].strip())
        daily.to_excel(xw, sheet_name="Daily")
        notes.to_excel(xw, sheet_name="Notes", index=False)

    print(f"  annual energy: {e10:,.1f} GWh (10-min) -> {e15:,.1f} GWh (15-min)")
    print(f"Done. Written {path}")


if __name__ == "__main__":
    main()
