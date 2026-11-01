import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter as col
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.workbook.properties import CalcProperties

# =============================================================================
# SETTINGS
# =============================================================================
CURVES_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Daily_Curves"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential"
OUTPUT_NAME = "Solar_Generation_10min_MW.xlsx"

# file name -> (display name, installed MW) - installed capacity at end of 2025
# Keep the MW values identical to WEIGHTS in national_curve.py.
STATIONS = {
    "Rathnapura":   ("Ratnapura", 786.500),
    "Kurunagala":   ("Kurunegala", 662.523),
    "Anuradhapura": ("Anuradhapura", 379.123),
    "Hambanthota":  ("Hambantota", 270.337),
    "Jaffna":       ("Jaffna", 253.600),
    "Potuvil":      ("Pottuvil", 214.500),
    "Badulla":      ("Badulla", 114.000),
    "Puttalam":     ("Puttalam", 88.500),
    "Monaragala":   ("Monaragala", 77.500),
    "Mannar":       ("Mannar", 17.000),
}
PERFORMANCE_RATIO = 0.82   # use 0.80 / 0.85 as sensitivity
REQUIRE_NATIONAL = True    # stop if National_daily_curves.xlsx is missing or uses other weights
G_STC = 1000
YEAR_SHOWN = 2023          # nominal calendar year used for the timestamps
# =============================================================================

ARIAL = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
BLUE = Font(name="Arial", size=10, color="0000FF")
TITLE = Font(name="Arial", size=14, bold=True)
YELLOW = PatternFill("solid", start_color="FFFF00")
HEAD_FILL = PatternFill("solid", start_color="D9E1F2")
N = 365 * 144


def cell(ws, value, font=ARIAL, fmt=None, fill=None):
    c = WriteOnlyCell(ws, value=value)
    c.font = font
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    return c


def header(ws, names):
    return [cell(ws, n, BOLD, fill=HEAD_FILL) for n in names]


def read_curve(path, name):
    return pd.read_excel(path, sheet_name=f"{name}_Wm2", index_col=0).astype(float).values.ravel()


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else CURVES_DIR)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
    if not src.is_dir():
        print(f"Curves folder not found: {src}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    ghi, stations = {}, []
    for key in STATIONS:
        f = src / f"{key}_daily_curves.xlsx"
        if f.is_file():
            print(f"  reading {f.name}")
            ghi[key] = read_curve(f, "Mean")
            stations.append(key)
        else:
            print(f"  WARNING: {f.name} not found - station left out")
    if not stations:
        print("Nothing to process.")
        return
    total_mw = sum(STATIONS[s][1] for s in stations)

    nat_pct = {}
    nat = src / "National_daily_curves.xlsx"
    problem = None
    if nat.is_file():
        w = pd.read_excel(nat, sheet_name="Weights", index_col=0)["Weight"]
        target = pd.Series({s: STATIONS[s][1] / total_mw for s in stations})
        if set(w.index) == set(target.index) and \
           np.allclose(w.reindex(target.index).values, target.values, atol=1e-3):
            for p in ("P10", "P50", "P90"):
                nat_pct[p] = read_curve(nat, p)
            print("  national P10/P50/P90 included")
        else:
            problem = ("National_daily_curves.xlsx was built with different weights - "
                       "set WEIGHTS in national_curve.py to the capacities and run it again")
    else:
        problem = f"National_daily_curves.xlsx not found in {src} - run national_curve.py first"
    if problem:
        if REQUIRE_NATIONAL:
            print("  STOPPED: " + problem)
            return
        print("  WARNING: " + problem + " (national P10/P50/P90 left out)")


    start = datetime(YEAR_SHOWN, 1, 1)
    stamps = [start + timedelta(minutes=10 * i) for i in range(N)]

    # ---------------------------------------------------------------- workbook
    wb = Workbook(write_only=True)
    wb._fonts = IndexedList([Font(name="Arial", size=10)])     # default font Arial
    wb.calculation = CalcProperties(fullCalcOnLoad=True)

    ns = len(stations)
    cap_row = {s: 7 + i for i, s in enumerate(stations)}
    tot_row = 7 + ns

    # Inputs -------------------------------------------------------------
    ws = wb.create_sheet("Inputs")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 12
    ws.append([cell(ws, "Solar generation potential - inputs", TITLE)])
    ws.append([])
    ws.append([cell(ws, "Performance ratio (eta)"), cell(ws, PERFORMANCE_RATIO, BLUE, "0.00", YELLOW)])
    ws.append([cell(ws, "Rated irradiance (W/m2)"), cell(ws, G_STC, BLUE, "#,##0")])
    ws.append([])
    ws.append(header(ws, ["Station", "Curve file name", "Capacity (MW)", "Share"]))
    for s in stations:
        r = cap_row[s]
        ws.append([cell(ws, STATIONS[s][0]), cell(ws, s),
                   cell(ws, STATIONS[s][1], BLUE, "#,##0", YELLOW),
                   cell(ws, f"=C{r}/$C${tot_row}", fmt="0.0%")])
    ws.append([cell(ws, "Total", BOLD), None,
               cell(ws, f"=SUM(C7:C{tot_row - 1})", BOLD, "#,##0"),
               cell(ws, f"=SUM(D7:D{tot_row - 1})", BOLD, "0.0%")])
    ws.append([])
    notes = [
        "Formula: MW = GHI x Capacity / Rated irradiance x Performance ratio.",
        "Performance ratio covers inverter, wiring, soiling, heat and mismatch losses; "
        "panel efficiency is already inside the MW rating.",
        f"Timestamps are a typical year shown on the {YEAR_SHOWN} calendar. "
        "12:10 = average over 12:00-12:10.",
        "GHI = average (Mean) curve of each station from solar_daily_curves.py.",
        "National total = sum of the station MW values at each timestamp (expected output).",
        "National P10/P50/P90 = percentiles of the capacity-weighted national irradiance, "
        "combined day by day. Use them for a single time of day, not for annual energy: "
        "each 10-minute slot is ranked separately, so no real day stays at P90 all day.",
        "National P10/P50/P90 follow changes to the performance ratio and total capacity, "
        "but not to the split between stations (re-run national_curve.py for that).",
    ]
    for t in notes:
        ws.append([cell(ws, t)])

    # GHI sheet (data) -----------------------------------------------------
    gs = wb.create_sheet("GHI_Wm2")
    gs.freeze_panes = "B2"
    gs.column_dimensions["A"].width = 16
    ghi_cols = [f"{STATIONS[s][0]} (W/m2)" for s in stations] + \
               [f"National {p} (W/m2)" for p in nat_pct]
    gs.append(header(gs, ["Timestamp"] + ghi_cols))
    arrays = [ghi[s] for s in stations] + [nat_pct[p] for p in nat_pct]
    for i in range(N):
        gs.append([cell(gs, stamps[i], fmt="dd-mmm hh:mm")] + [float(a[i]) for a in arrays])

    # MW sheet (formulas) --------------------------------------------------
    ms = wb.create_sheet("MW_10min")
    ms.freeze_panes = "D2"
    ms.column_dimensions["A"].width = 16
    heads = ["Timestamp", "Time", "Month"] + [f"{STATIONS[s][0]} (MW)" for s in stations] + \
            ["National total (MW)"] + [f"National {p} (MW)" for p in nat_pct]
    ms.append(header(ms, heads))
    first_st, last_st = 4, 3 + ns                     # D .. station columns
    nat_col = last_st + 1
    for i in range(N):
        r = i + 2
        row = [cell(ms, stamps[i], fmt="dd-mmm hh:mm"), stamps[i].strftime("%H:%M"), stamps[i].month]
        for j, s in enumerate(stations):
            g = col(2 + j)
            row.append(f"=ROUND(GHI_Wm2!{g}{r}*Inputs!$C${cap_row[s]}/Inputs!$B$4*Inputs!$B$3,2)")
        row.append(f"=SUM({col(first_st)}{r}:{col(last_st)}{r})")
        for k, p in enumerate(nat_pct):
            g = col(2 + ns + k)
            row.append(f"=ROUND(GHI_Wm2!{g}{r}*Inputs!$C${tot_row}/Inputs!$B$4*Inputs!$B$3,2)")
        ms.append(row)
    mw_last_col = nat_col + len(nat_pct)

    # Daily energy (formulas) ----------------------------------------------
    ds = wb.create_sheet("Daily_MWh")
    ds.freeze_panes = "B2"
    ds.column_dimensions["A"].width = 12
    dheads = ["Date"] + [h.replace("(MW)", "(MWh)") for h in heads[3:]]
    ds.append(header(ds, dheads))
    for d in range(365):
        r0, r1 = d * 144 + 2, d * 144 + 145
        row = [cell(ds, start + timedelta(days=d), fmt="dd-mmm")]
        for c in range(first_st, mw_last_col + 1):
            L = col(c)
            row.append(cell(ds, f"=SUM(MW_10min!{L}{r0}:{L}{r1})/6", fmt="#,##0"))
        ds.append(row)
    ds.append([])
    tot = [cell(ds, "Year (GWh)", BOLD)]
    for c in range(2, len(dheads) + 1):
        L = col(c)
        tot.append(cell(ds, f"=SUM({L}2:{L}366)/1000", BOLD, "#,##0.0"))
    ds.append(tot)

    # Monthly average day (formulas) -------------------------------------
    months = pd.date_range(f"{YEAR_SHOWN}-01-01", periods=12, freq="MS").strftime("%b")
    N_ = N + 1

    def profile_sheet(title, src_col):
        sh = wb.create_sheet(title)
        sh.freeze_panes = "B2"
        sh.append(header(sh, ["Time"] + [f"{m} (MW)" for m in months]))
        L = col(src_col)
        for t in range(144):
            tt = f"{t // 6:02d}:{t % 6 * 10:02d}"
            row = [tt]
            for m in range(1, 13):
                row.append(cell(sh, f"=AVERAGEIFS(MW_10min!${L}$2:${L}${N_},"
                                    f"MW_10min!$C$2:$C${N_},{m},MW_10min!$B$2:$B${N_},$A{t + 2})",
                                fmt="#,##0"))
            sh.append(row)
        return sh

    ps = profile_sheet("Monthly_profile", nat_col)
    pp = profile_sheet("Monthly_profile_P90", nat_col + 3) if "P90" in nat_pct else None

    # Charts ---------------------------------------------------------------
    cs = wb.create_sheet("National_charts")

    c1 = LineChart()
    c1.title = "National solar generation potential - every 10 minutes (MW)"
    c1.y_axis.title = "MW"
    c1.x_axis.title = "Time of year"
    c1.x_axis.number_format = "mmm"
    c1.add_data(Reference(ms, min_col=nat_col, min_row=1, max_row=N + 1), titles_from_data=True)
    c1.set_categories(Reference(ms, min_col=1, min_row=2, max_row=N + 1))
    c1.series[0].graphicalProperties.line.width = 6000
    c1.series[0].smooth = False
    c1.legend = None
    c1.height, c1.width = 9, 30
    cs.add_chart(c1, "A1")

    c2 = LineChart()
    c2.title = "National daily energy (MWh)"
    c2.y_axis.title = "MWh per day"
    c2.x_axis.number_format = "dd-mmm"
    first_nat_d = 2 + ns
    c2.add_data(Reference(ds, min_col=first_nat_d, max_col=first_nat_d + len(nat_pct),
                          min_row=1, max_row=366), titles_from_data=True)
    c2.set_categories(Reference(ds, min_col=1, min_row=2, max_row=366))
    for s in c2.series:
        s.smooth = False
    c2.height, c2.width = 9, 30
    cs.add_chart(c2, "A20")

    c3 = LineChart()
    c3.title = "National average day by month (MW)"
    c3.y_axis.title = "MW"
    c3.x_axis.title = "Time of day"
    c3.add_data(Reference(ps, min_col=2, max_col=13, min_row=1, max_row=145), titles_from_data=True)
    c3.set_categories(Reference(ps, min_col=1, min_row=2, max_row=145))
    for s in c3.series:
        s.smooth = False
    c3.height, c3.width = 11, 30
    cs.add_chart(c3, "A39")

    if pp is not None:
        c4 = LineChart()
        c4.title = "National P90 (high-output) day by month (MW)"
        c4.y_axis.title = "MW"
        c4.x_axis.title = "Time of day"
        c4.add_data(Reference(pp, min_col=2, max_col=13, min_row=1, max_row=145), titles_from_data=True)
        c4.set_categories(Reference(pp, min_col=1, min_row=2, max_row=145))
        for s in c4.series:
            s.smooth = False
        c4.height, c4.width = 11, 30
        cs.add_chart(c4, "A62")

    path = out_dir / OUTPUT_NAME
    print(f"  writing {path.name} (this takes a few minutes) ...")
    wb.save(path)
    print(f"Done. Written {path}")


if __name__ == "__main__":
    main()
