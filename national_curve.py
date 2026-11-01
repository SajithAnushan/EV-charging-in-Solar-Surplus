import sys
from pathlib import Path

import numpy as np
import pandas as pd

import solar_daily_curves as sdc

# =============================================================================
# SETTINGS
# =============================================================================
# Input workbook (one sheet per station) and folder for the national output.
# These are used instead of the paths in solar_daily_curves.py.
INPUT_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Solar_Radiation_All_Stations.xlsx"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Daily_Curves"

# Station weights = installed capacity allocated to each station (MW), end of 2025.
# This makes the national curve a capacity-weighted irradiance, so that
# national MW = national curve x total capacity / 1000 x performance ratio.
# Keep these identical to STATIONS in generation_10min_excel.py.
WEIGHTS = {
    "Rathnapura":   786.500,
    "Kurunagala":   662.523,
    "Anuradhapura": 379.123,
    "Hambanthota":  270.337,
    "Jaffna":       253.600,
    "Potuvil":      214.500,
    "Badulla":       114.000,
    "Puttalam":      88.500,
    "Monaragala":    77.500,
    "Mannar":        17.000,
}

# A national day is usable only if stations holding at least this share of
# the total weight have real (usable) data that day. Missing stations are
# filled with their own average curve for that day of year.
MIN_REAL_WEIGHT = 0.70

OUTPUT_NAME = "National_daily_curves.xlsx"
# =============================================================================


def station_arrays(name, raw_df, dates, doy, yr):
    """Run the station pipeline; return QC'd irradiance, usable days, mean curve."""
    lat, lon, alt = sdc.STATIONS[name]
    site = sdc.Location(lat, lon, tz=sdc.TIMEZONE, altitude=alt, name=name)
    flags = {}
    R = sdc.load_station(raw_df, dates)
    G = sdc.stage1_qc(R, flags)
    shift = float(sdc.MANUAL_SHIFT.get(name, sdc.DEFAULT_SHIFT_MIN))
    elev, _, cs = sdc.solar_grid(site, dates, shift)
    K, Gf, usable = sdc.stage2_qc(G, elev, cs, flags)
    ref_elev, _, ref_cs = sdc.solar_grid(site, sdc.REF_DAYS, shift)
    curves, _ = sdc.build_curves(K, Gf, usable, doy, yr, ref_elev, ref_cs)
    return Gf, usable, curves["Mean"]


def national_curves(Gnat, usable, doy, yr):
    names = ["Mean"] + [f"P{p}" for p in sdc.PERCENTILES]
    curves = {n: np.zeros((365, 144)) for n in names}
    info = []
    for d in range(365):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        half = sdc.WINDOW_HALF
        while True:
            sel = usable & (dist <= half)
            if sel.sum() >= sdc.MIN_DAYS_IN_WINDOW or half >= sdc.WINDOW_MAX_HALF:
                break
            half += 1
        if sel.sum() == 0:
            for n in names:
                curves[n][d] = np.nan
            info.append([sdc.REF_LABELS[d], half, 0, 0, 0, 0])
            continue
        mean, nyears, counts = sdc.year_mean(Gnat, sel, yr)
        curves["Mean"][d] = sdc.fill_curve(mean)
        for p in sdc.PERCENTILES:
            curves[f"P{p}"][d] = sdc.fill_curve(np.nanpercentile(Gnat[sel], p, axis=0))
        info.append([sdc.REF_LABELS[d], half, int(sel.sum()), counts[0], counts[1], nyears])

    info = pd.DataFrame(info, columns=["Day", "Window_half_days", "Days_used",
                                       "Days_year1", "Days_year2", "Years_balanced"])
    for n in names:
        info[f"{n}_kWh_m2_day"] = (np.nansum(curves[n], axis=1) * 600 / 3.6e6).round(3)
    return curves, info


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
    if not inp.is_file():
        print(f"Input file not found: {inp}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    w = {k: float(v) for k, v in WEIGHTS.items() if v and v > 0}
    total = sum(w.values())
    if total == 0:
        print("All weights are zero - nothing to combine.")
        return
    w = {k: v / total for k, v in w.items()}

    print(f"Reading {inp.name} ...")
    sheets = pd.read_excel(inp, sheet_name=None)
    dates = pd.date_range(sdc.START_DATE, sdc.END_DATE)
    doy = sdc.ref_doy(dates)
    yr = (dates >= pd.Timestamp(sdc.YEAR_SPLIT)).astype(int)

    Gnat = np.zeros((len(dates), 144))
    real_weight = np.zeros(len(dates))
    station_mean = np.zeros((365, 144))
    used = {}

    for sheet, df in sheets.items():
        name = sheet.split("-")[0].strip()
        if name not in w:
            continue
        if name not in sdc.STATIONS:
            print(f"  {name}: no coordinates in STATIONS - skipped")
            continue
        print(f"  processing {name} ...")
        Gf, usable, mean_curve = station_arrays(name, df, dates, doy, yr)

        clim = mean_curve[doy]                        # station's own average for that day of year
        filled = np.where(usable[:, None] & np.isfinite(Gf), Gf, clim)
        Gnat += w[name] * filled
        real_weight += w[name] * usable
        station_mean += w[name] * mean_curve
        used[name] = {"Weight": round(w[name], 4), "Usable_days": int(usable.sum())}

    missing = set(w) - set(used)
    if missing:
        print(f"  WARNING: weighted stations not found in workbook: {', '.join(sorted(missing))}")
        scale = sum(v["Weight"] for v in used.values())
        Gnat, real_weight, station_mean = Gnat / scale, real_weight / scale, station_mean / scale

    usable_nat = real_weight >= MIN_REAL_WEIGHT
    print(f"  national usable days: {usable_nat.sum()}/{len(dates)}")

    curves, info = national_curves(Gnat, usable_nat, doy, yr)

    # Check: the national mean should closely match the weighted average of station means
    check = pd.DataFrame({
        "National_mean_kWh_m2_day": info["Mean_kWh_m2_day"].values,
        "Weighted_station_means_kWh_m2_day": (station_mean.sum(axis=1) * 600 / 3.6e6).round(3),
    }, index=sdc.REF_LABELS)
    check["Difference_%"] = ((check.iloc[:, 0] / check.iloc[:, 1] - 1) * 100).round(2)
    monthly = check.iloc[:, :2].groupby(sdc.REF_DAYS.month).mean().round(3)
    monthly.index = pd.to_datetime(monthly.index, format="%m").strftime("%b")
    monthly["Difference_%"] = ((monthly.iloc[:, 0] / monthly.iloc[:, 1] - 1) * 100).round(2)

    path = out_dir / OUTPUT_NAME
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for n, arr in curves.items():
            pd.DataFrame(arr.round(1), index=sdc.REF_LABELS, columns=sdc.SLOTS) \
              .rename_axis("Day").to_excel(xw, sheet_name=f"{n}_Wm2")
        info.to_excel(xw, sheet_name="Info", index=False)
        monthly.rename_axis("Month").to_excel(xw, sheet_name="Monthly_check")
        check.rename_axis("Day").to_excel(xw, sheet_name="Daily_check")
        pd.DataFrame(used).T.rename_axis("Station").to_excel(xw, sheet_name="Weights")

    print(f"\nDone. Written {path}")


if __name__ == "__main__":
    main()
