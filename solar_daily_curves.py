import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location

# =============================================================================
# SETTINGS
# =============================================================================
INPUT_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Solar_Radiation_All_Stations.xlsx"
OUTPUT_DIR = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Daily_Curves"

START_DATE = "2022-09-15"
END_DATE = "2024-08-14"
YEAR_SPLIT = "2023-09-15"          # first day of the second data year
TIMEZONE = "Asia/Colombo"

EXCLUDE = ["Matale", "Vavniya", "Vavuniya"]

# (latitude, longitude, altitude in metres) - station coordinates
# Keys must match the start of each sheet name (text before the first "-").
STATIONS = {
    "Anuradhapura": (8.35, 80.38, 93),
    "Badulla":      (6.98, 81.05, 670),
    "Hambanthota":  (6.12, 81.13, 20),
    "Jaffna":       (9.65, 80.05, 2),
    "Kurunagala":   (7.47, 80.37, 116),
    "Mannar":       (8.98, 79.92, 4),
    "Monaragala":   (6.87, 81.35, 155),
    "Potuvil":      (6.88, 81.83, 5),
    "Puttalam":     (8.03, 79.83, 2),
    "Rathnapura":   (6.68, 80.40, 34),
}

# Unit conversion: raw x 0.01 = kJ/m2 per 10 min;  x 1000 = J/m2;  / 600 s = W/m2
RAW_TO_WM2 = 0.01 * 1000 / 600

# Timestamp handling. The label is taken to mark the END of its interval,
# so the interval midpoint is label - 5 min. Puttalam's data runs about
# 10 min later than the other stations, so it gets its own value.
# AUTO_SHIFT=True estimates a shift per station from morning/afternoon
# symmetry instead. It is off by default because afternoon clouds bias it
# by a few minutes at inland stations.
DEFAULT_SHIFT_MIN = -5.0
MANUAL_SHIFT = {"Puttalam": -12.5}
AUTO_SHIFT = False
SHIFT_CANDIDATES = np.arange(-20.0, 10.01, 2.5)

# Quality control
FROZEN_MIN_RUN = 3        # identical consecutive readings -> logger freeze
FROZEN_MIN_RAW = 3000     # only check freezes above this raw value
OFFSET_COLS = ("00:00", "04:50")   # pre-dawn window used to measure sensor offset
OFFSET_MAX_RAW = 3000     # larger pre-dawn values are not treated as an offset
OFFSET_FILL_DAYS = 3      # borrow offset from neighbouring days if pre-dawn is missing
ZERO_CHECK_ELEV = 15      # a zero reading with sun above this (deg) is an outage
K_MIN_ELEV = 5            # clear-sky index only used above this sun elevation (deg)
MAX_K = 1.35              # measured / clear-sky above this -> spike
MAX_GAP_SLOTS = 6         # fill gaps up to 6 slots (1 hour) by interpolation
MIN_DAY_COVERAGE = 0.75   # usable day: >= 75% of daylight slots valid

# Averaging
WINDOW_HALF = 15          # +/- days around each day of year
WINDOW_MAX_HALF = 30      # widen up to this if too few usable days
MIN_DAYS_IN_WINDOW = 20
MIN_DAYS_PER_YEAR = 5     # both years need this many days to be balanced
PERCENTILES = [10, 50, 90]
# =============================================================================

SLOTS = [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, 10)]
REF_DAYS = pd.date_range("2023-01-01", "2023-12-31")        # 365-day calendar
REF_LABELS = REF_DAYS.strftime("%d-%b")

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected


# ----------------------------------------------------------------------------- loading
def clean_col(c):
    if hasattr(c, "strftime"):
        return c.strftime("%H:%M")
    return str(c).strip()[:5]


def load_station(df, dates):
    df = df.copy()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [clean_col(c) for c in df.columns]
    df = df.reindex(columns=SLOTS).apply(pd.to_numeric, errors="coerce")
    return df.reindex(dates).values.astype(float)   # out-of-range dates dropped


# ----------------------------------------------------------------------------- sun
def solar_grid(site, days, shift_min):
    """Sun elevation, azimuth and Ineichen clear-sky GHI at each interval midpoint."""
    offsets = np.tile(np.arange(144) * 600, len(days)).astype("timedelta64[s]")
    t = np.repeat(days.values, 144) + offsets
    times = (pd.DatetimeIndex(t) + pd.Timedelta(minutes=shift_min)).tz_localize(TIMEZONE)
    sp = site.get_solarposition(times)
    cs = site.get_clearsky(times, model="ineichen", solar_position=sp)
    shape = (len(days), 144)
    return (sp["apparent_elevation"].values.reshape(shape),
            sp["azimuth"].values.reshape(shape),
            cs["ghi"].values.reshape(shape))


# ----------------------------------------------------------------------------- QC
def frozen_mask(R):
    mask = np.zeros(R.shape, dtype=bool)
    for i, row in enumerate(R):
        j, n = 0, len(row)
        while j < n:
            v = row[j]
            if not np.isfinite(v) or v <= FROZEN_MIN_RAW:
                j += 1
                continue
            k = j + 1
            while k < n and row[k] == v:
                k += 1
            if k - j >= FROZEN_MIN_RUN:
                mask[i, j:k] = True
            j = k
    return mask


def stage1_qc(R, flags):
    """Checks that don't need sun position: negatives, freezes, sensor offset."""
    R = R.copy()
    neg = R < 0
    flags["negative"] = int(neg.sum())
    R[neg] = np.nan

    fr = frozen_mask(R)
    flags["frozen"] = int(fr.sum())
    R[fr] = np.nan

    a, b = SLOTS.index(OFFSET_COLS[0]), SLOTS.index(OFFSET_COLS[1]) + 1
    pre = pd.Series(np.nanmedian(R[:, a:b], axis=1))
    pre[pre > OFFSET_MAX_RAW] = np.nan
    off = pre.ffill(limit=OFFSET_FILL_DAYS).bfill(limit=OFFSET_FILL_DAYS).fillna(0).values
    flags["offset_days"] = int((off > 0).sum())
    flags["max_offset_Wm2"] = round(float(off.max()) * RAW_TO_WM2, 1)

    return np.clip(R - off[:, None], 0, None) * RAW_TO_WM2


def symmetry_ratio(G, elev, azim, cs):
    """Morning / afternoon ratio of the clear-sky-index envelope (1.0 = aligned)."""
    band = (elev > 10) & (elev < 37) & np.isfinite(G) & (cs > 0)
    k = np.where(band, G / np.where(cs > 0, cs, np.nan), np.nan)
    am, pm = band & (azim < 180), band & (azim >= 180)
    if am.sum() < 200 or pm.sum() < 200:
        return np.nan
    return np.nanpercentile(k[am], 90) / np.nanpercentile(k[pm], 90)


def fill_short_gaps(K):
    K = K.copy()
    for row in K:
        valid = np.flatnonzero(np.isfinite(row))
        if len(valid) < 2:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            gap = b - a - 1
            if 0 < gap <= MAX_GAP_SLOTS:
                row[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [row[a], row[b]])
    return K


def stage2_qc(G, elev, cs, flags):
    """Sun-dependent checks, clear-sky index, gap filling, usable days."""
    G = G.copy()
    zero = (elev > ZERO_CHECK_ELEV) & (G <= 0)
    flags["zero_in_daylight"] = int(zero.sum())
    G[zero] = np.nan

    region = elev > K_MIN_ELEV
    low = (elev > 0) & ~region
    night = elev <= 0

    K = np.where(region, G / np.where(cs > 0, cs, np.nan), np.nan)
    spike = region & (K > MAX_K)
    spike_low = low & (G > MAX_K * cs + 20)
    flags["spikes"] = int(spike.sum() + spike_low.sum())
    K[spike] = np.nan
    G[spike_low] = np.nan

    missing_before = np.isnan(K) & region
    K = fill_short_gaps(K)
    flags["gap_filled_slots"] = int((missing_before & np.isfinite(K)).sum())

    coverage = (np.isfinite(K) & region).sum(axis=1) / np.maximum(region.sum(axis=1), 1)
    usable = coverage >= MIN_DAY_COVERAGE

    Gf = np.where(region, K * cs, G)      # QC'd, gap-filled irradiance
    Gf[night] = 0.0
    return K, Gf, usable


# ----------------------------------------------------------------------------- averaging
def ref_doy(dates):
    m, d = dates.month.values, dates.day.values
    d = np.where((m == 2) & (d == 29), 28, d)        # fold 29 Feb into 28 Feb
    return pd.to_datetime(pd.DataFrame({"year": 2023, "month": m, "day": d})).dt.dayofyear.values - 1


def year_mean(X, sel, yr):
    counts = [int((sel & (yr == y)).sum()) for y in (0, 1)]
    if min(counts) >= MIN_DAYS_PER_YEAR:
        parts = [np.nanmean(X[sel & (yr == y)], axis=0) for y in (0, 1)]
        return np.nanmean(np.vstack(parts), axis=0), 2, counts
    return np.nanmean(X[sel], axis=0), 1, counts


def fill_curve(c):
    c = c.copy()
    ok = np.isfinite(c)
    if ok.sum() >= 2:
        c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok])
    return np.nan_to_num(c, nan=0.0)


def build_curves(K, Gf, usable, doy, yr, ref_elev, ref_cs):
    names = ["Mean"] + [f"P{p}" for p in PERCENTILES]
    curves = {n: np.zeros((365, 144)) for n in names}
    info = []
    for d in range(365):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        half = WINDOW_HALF
        while True:
            sel = usable & (dist <= half)
            if sel.sum() >= MIN_DAYS_IN_WINDOW or half >= WINDOW_MAX_HALF:
                break
            half += 1

        region = ref_elev[d] > K_MIN_ELEV
        night = ref_elev[d] <= 0

        if sel.sum() == 0:
            for n in names:
                curves[n][d] = np.nan
            info.append([REF_LABELS[d], half, 0, 0, 0, 0])
            continue

        mk, nyears, counts = year_mean(K, sel, yr)
        mg, _, _ = year_mean(Gf, sel, yr)
        stats = {"Mean": (mk, mg)}
        for p in PERCENTILES:
            stats[f"P{p}"] = (np.nanpercentile(K[sel], p, axis=0),
                              np.nanpercentile(Gf[sel], p, axis=0))
        for n, (kk, gg) in stats.items():
            c = np.where(region & np.isfinite(kk), kk * ref_cs[d], gg)
            c[night] = 0.0
            curves[n][d] = fill_curve(c)

        info.append([REF_LABELS[d], half, int(sel.sum()), counts[0], counts[1], nyears])

    info = pd.DataFrame(info, columns=["Day", "Window_half_days", "Days_used",
                                       "Days_year1", "Days_year2", "Years_balanced"])
    for n in names:
        info[f"{n}_kWh_m2_day"] = (np.nansum(curves[n], axis=1) * 600 / 3.6e6).round(3)
    return curves, info


def monthly_check(K, Gf, usable, cs, elev, dates, info):
    region = elev > K_MIN_ELEV
    kday = np.nanmean(K, axis=1)
    G = np.where(region & np.isnan(Gf), kday[:, None] * cs, Gf)
    G = np.nan_to_num(G, nan=0.0)
    e = G.sum(axis=1) * 600 / 3.6e6
    obs = pd.DataFrame({"m": dates.month[usable], "e": e[usable]}).groupby("m")["e"]
    curve = pd.Series(info["Mean_kWh_m2_day"].values, index=REF_DAYS.month).groupby(level=0).mean()
    out = pd.DataFrame({"Usable_days": obs.size(),
                        "Observed_kWh_m2_day": obs.mean().round(3),
                        "Curve_kWh_m2_day": curve.round(3)})
    out["Curve_vs_observed_%"] = ((out["Curve_kWh_m2_day"] / out["Observed_kWh_m2_day"] - 1) * 100).round(1)
    out.index = pd.to_datetime(out.index, format="%m").strftime("%b")
    out.index.name = "Month"
    return out


# ----------------------------------------------------------------------------- main
def process_station(name, raw_df, dates, doy, yr, out_dir):
    lat, lon, alt = STATIONS[name]
    site = Location(lat, lon, tz=TIMEZONE, altitude=alt, name=name)
    flags = {"station": name}

    R = load_station(raw_df, dates)
    G = stage1_qc(R, flags)

    if name in MANUAL_SHIFT:
        shift = float(MANUAL_SHIFT[name])
    elif AUTO_SHIFT:
        ratios = {}
        for s in SHIFT_CANDIDATES:
            e, a, c = solar_grid(site, dates, s)
            ratios[s] = symmetry_ratio(G, e, a, c)
        good = {s: r for s, r in ratios.items() if np.isfinite(r)}
        shift = min(good, key=lambda s: abs(np.log(good[s]))) if good else DEFAULT_SHIFT_MIN
        flags["symmetry_at_default"] = round(ratios.get(DEFAULT_SHIFT_MIN, np.nan), 3)
        flags["symmetry_at_chosen"] = round(good.get(shift, np.nan), 3)
    else:
        shift = DEFAULT_SHIFT_MIN
    flags["time_shift_min"] = shift

    elev, _, cs = solar_grid(site, dates, shift)
    K, Gf, usable = stage2_qc(G, elev, cs, flags)
    flags["usable_days"] = int(usable.sum())
    flags["total_days"] = len(dates)

    ref_elev, _, ref_cs = solar_grid(site, REF_DAYS, shift)
    curves, info = build_curves(K, Gf, usable, doy, yr, ref_elev, ref_cs)
    check = monthly_check(K, Gf, usable, cs, elev, dates, info)

    path = out_dir / f"{name}_daily_curves.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for n, arr in curves.items():
            pd.DataFrame(arr.round(1), index=REF_LABELS, columns=SLOTS) \
              .rename_axis("Day").to_excel(xw, sheet_name=f"{n}_Wm2")
        info.to_excel(xw, sheet_name="Info", index=False)
        check.to_excel(xw, sheet_name="Monthly_check")
        pd.DataFrame({"Item": list(flags), "Value": list(flags.values())}) \
          .to_excel(xw, sheet_name="QC", index=False)

    print(f"  {name}: {flags['usable_days']}/{len(dates)} usable days, "
          f"shift {shift:+.1f} min, frozen {flags['frozen']}, "
          f"zeros {flags['zero_in_daylight']}, spikes {flags['spikes']}, "
          f"offset days {flags['offset_days']}")
    return flags, info["Mean_kWh_m2_day"].values


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
    if not inp.is_file():
        print(f"Input file not found: {inp}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {inp.name} ...")
    sheets = pd.read_excel(inp, sheet_name=None)
    dates = pd.date_range(START_DATE, END_DATE)
    doy = ref_doy(dates)
    yr = (dates >= pd.Timestamp(YEAR_SPLIT)).astype(int)

    summary, energy = [], {}
    for sheet, df in sheets.items():
        name = sheet.split("-")[0].strip()
        if any(name.lower() == x.lower() for x in EXCLUDE):
            print(f"  {name}: excluded")
            continue
        if name not in STATIONS:
            print(f"  {name}: no coordinates in STATIONS - skipped")
            continue
        try:
            flags, e = process_station(name, df, dates, doy, yr, out_dir)
            summary.append(flags)
            energy[name] = e
        except Exception as ex:
            print(f"  {name}: ERROR - {ex}")

    if summary:
        with pd.ExcelWriter(out_dir / "Summary_All_Stations.xlsx", engine="openpyxl") as xw:
            pd.DataFrame(summary).to_excel(xw, sheet_name="QC_summary", index=False)
            pd.DataFrame(energy, index=REF_LABELS).rename_axis("Day") \
              .to_excel(xw, sheet_name="Mean_kWh_m2_day")
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
