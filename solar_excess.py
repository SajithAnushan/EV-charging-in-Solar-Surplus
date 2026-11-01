import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font
from scipy.ndimage import gaussian_filter

# =============================================================================
# SETTINGS
# =============================================================================
POTENTIAL_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential\National_Mean_15min_MW.xlsx"
GENERATION_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\solar_pv_2026-03-26_to_2026-09-24.xlsx"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential"
OUTPUT_NAME = "Solar_Excess_vs_Mean.xlsx"

START_MMDD = "03-26"     # first day to include (month-day)
END_MMDD = "10-16"       # last day to include (month-day); data now runs to 16 Oct
DAY_START = "06:30"      # first timestamp counted each day
DAY_END = "18:30"        # last timestamp counted each day (inclusive); all others are 0

# Filling of negative values
SIGMA_TIME_SLOTS = 2     # neighbourhood width along the day (2 x 15 min = 30 min)
SIGMA_DAYS = 3           # neighbourhood width across days
MAX_WIDENING = 6         # widen the neighbourhood up to 2^6 = 64x if nothing is nearby
MIN_SURPLUS_PCT = 2.0    # every filled value >= this % of actual generation
MIN_SURPLUS_MW = 1.0     # and >= this many MW
# =============================================================================

SLOTS_15 = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, 15)]


def read_potential_15min(path):
    """15-minute mean potential for the typical year: 365 x 96 array, indexed by day of year."""
    df = pd.read_excel(path, sheet_name=0, index_col=0)          # first sheet = the 15-min table
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.columns = [c.strftime("%H:%M") if hasattr(c, "strftime") else str(c).strip()[:5] for c in df.columns]
    missing = [c for c in SLOTS_15 if c not in df.columns]
    if missing:
        raise ValueError(f"Potential file is missing time columns, e.g. {missing[:3]}")
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    doy = pd.to_datetime({"year": 2023, "month": df.index.month, "day": df.index.day}).dt.dayofyear.values - 1
    if len(set(doy)) != 365:
        raise ValueError(f"Potential file should have 365 days, found {len(set(doy))}")
    out = np.full((365, 96), np.nan)
    out[doy] = df[SLOTS_15].apply(pd.to_numeric, errors="coerce").values
    try:                                                          # label convention must match
        notes = pd.read_excel(path, sheet_name="Notes")
        label = str(notes.loc[notes["Item"] == "15-minute label", "Value"].iloc[0])
        if not label.lower().startswith("end"):
            print(f"  WARNING: potential file uses '{label}' labels; generation data uses end of interval")
    except Exception:
        pass
    return np.nan_to_num(out)


def read_generation(path):
    df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.columns = [c.strftime("%H:%M") if hasattr(c, "strftime") else str(c).strip()[:5] for c in df.columns]
    missing = [c for c in SLOTS_15 if c not in df.columns]
    if missing:
        raise ValueError(f"Generation file is missing time columns, e.g. {missing[:3]}")
    return df[SLOTS_15].apply(pd.to_numeric, errors="coerce")


def normalised_fill(values, known, sigma):
    """Weighted average of known neighbours; NaN where no neighbour carries weight."""
    num = gaussian_filter(np.where(known, values, 0.0), sigma=sigma, mode="nearest")
    den = gaussian_filter(known.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out


def fill_negatives(win, actual_win):
    """Replace values <= 0 with estimates from neighbouring positive values."""
    missing = win <= 0
    known = ~missing
    filled = win.copy()
    todo = missing.copy()
    sigma = np.array([SIGMA_DAYS, SIGMA_TIME_SLOTS], dtype=float)
    for step in range(MAX_WIDENING + 1):
        est = normalised_fill(win, known, sigma * 2 ** step)
        ok = todo & np.isfinite(est)
        filled[ok] = est[ok]
        todo &= ~ok
        if not todo.any():
            break
    floor = np.maximum(MIN_SURPLUS_MW, actual_win * MIN_SURPLUS_PCT / 100)
    filled[missing] = np.maximum(np.nan_to_num(filled[missing]), floor[missing])
    return filled, missing


def main():
    pot_path = Path(sys.argv[1] if len(sys.argv) > 1 else POTENTIAL_FILE)
    gen_path = Path(sys.argv[2] if len(sys.argv) > 2 else GENERATION_FILE)
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR)
    for p in (pot_path, gen_path):
        if not p.is_file():
            print(f"File not found: {p}")
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {pot_path.name} (15-minute mean potential) ...")
    potential = read_potential_15min(pot_path)

    print(f"Reading {gen_path.name} ...")
    gen = read_generation(gen_path)
    mmdd = gen.index.strftime("%m-%d")
    gen = gen[(mmdd >= START_MMDD) & (mmdd <= END_MMDD)]
    gen = gen[~((gen.index.month == 2) & (gen.index.day == 29))]
    if gen.empty:
        print("No generation data between the selected dates.")
        return

    # Match each date to the same day and month in the typical year.
    doy = pd.to_datetime({"year": 2023, "month": gen.index.month, "day": gen.index.day}).dt.dayofyear.values - 1
    actual = gen.fillna(0).values          # blanks at night = no generation
    pot_days = potential[doy]

    inside = np.array([DAY_START <= t <= DAY_END for t in SLOTS_15])
    cols = SLOTS_15

    diff = np.round(pot_days - actual, 2)                  # actual above potential -> negative
    diff[:, ~inside] = 0.0
    diff_zero = np.where(diff < 0, 0.0, diff)

    filled_win, missing = fill_negatives(diff[:, inside], actual[:, inside])
    diff_filled = np.zeros_like(diff)
    diff_filled[:, inside] = np.round(filled_win, 2)
    flag = np.zeros(diff.shape, dtype=int)
    flag[:, np.flatnonzero(inside)] = missing.astype(int)

    sheets = {
        "Difference": diff,
        "Excess_set_to_0": diff_zero,
        "Excess_filled": diff_filled,
        "Filled_flag": flag,
    }

    # Average daily curve from the Excess_filled sheet
    ef = pd.DataFrame(diff_filled, index=gen.index, columns=cols)
    avg = pd.DataFrame({"Average, all days (MW)": ef.mean(axis=0)})
    for m, grp in ef.groupby(ef.index.month):
        avg[f"{pd.Timestamp(2000, m, 1):%b} ({len(grp)} days) (MW)"] = grp.mean(axis=0)
    avg = avg.round(2)
    avg.index.name = "Time"
    avg_energy = avg.sum(axis=0) / 4          # MWh on the average day

    path = out_dir / OUTPUT_NAME
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd") as xw:
        for name, arr in sheets.items():
            df = pd.DataFrame(arr, index=gen.index, columns=cols)
            df.index.name = "Date"
            df.to_excel(xw, sheet_name=name)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2"
            ws.column_dimensions["A"].width = 12

        avg.to_excel(xw, sheet_name="Average_daily_curve")
        ws = xw.sheets["Average_daily_curve"]
        ws.freeze_panes = "B2"
        ws.column_dimensions["A"].width = 10
        for i in range(2, avg.shape[1] + 2):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = 16
        last = len(avg) + 1
        ws.cell(row=last + 2, column=1, value="Energy (MWh/day)").font = Font(bold=True)
        for i, v in enumerate(avg_energy.values):
            ws.cell(row=last + 2, column=i + 2, value=round(float(v), 1)).font = Font(bold=True)

        chart_col = ws.cell(row=1, column=avg.shape[1] + 3).column_letter
        ch = LineChart()
        ch.title = f"Average daily excess curve ({gen.index.min():%d %b} - {gen.index.max():%d %b})"
        ch.y_axis.title = "MW (potential - generation)"
        ch.x_axis.title = "Time of day"
        ch.add_data(Reference(ws, min_col=2, min_row=1, max_row=last), titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
        ch.series[0].smooth = False
        ch.legend = None
        ch.height, ch.width = 9, 22
        ws.add_chart(ch, f"{chart_col}2")

        ch2 = LineChart()
        ch2.title = "Average daily excess curve by month"
        ch2.y_axis.title = "MW"
        ch2.x_axis.title = "Time of day"
        ch2.add_data(Reference(ws, min_col=3, max_col=avg.shape[1] + 1, min_row=1, max_row=last),
                     titles_from_data=True)
        ch2.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
        for sr in ch2.series:
            sr.smooth = False
        ch2.height, ch2.width = 10, 22
        ws.add_chart(ch2, f"{chart_col}22")

    n_day = int(inside.sum()) * len(gen)
    n_fill = int(missing.sum())
    before = diff_zero.sum() / 4
    after = diff_filled.sum() / 4
    print(f"  {len(gen)} days, {gen.index.min():%d %b} to {gen.index.max():%d %b}, "
          f"timestamps {DAY_START}-{DAY_END} counted ({inside.sum()} per day), others set to 0")
    print(f"  actual above potential in {n_fill:,} of {n_day:,} daytime intervals ({n_fill / n_day:.0%}) - refilled")
    print(f"  excess energy: {before:,.0f} MWh with negatives set to 0 -> {after:,.0f} MWh after filling")
    peak_t = avg.iloc[:, 0].idxmax()
    print(f"  average daily excess curve: peak {avg.iloc[:, 0].max():,.1f} MW at {peak_t}, "
          f"{avg_energy.iloc[0]:,.0f} MWh per average day")
    print(f"Done. Written {path}")


if __name__ == "__main__":
    main()
