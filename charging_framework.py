
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:                                    # LP solver for P4 (falls back to the heuristic if missing)
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ===========================================================================

INTERVAL_MIN = 15
DT = INTERVAL_MIN / 60.0
N_INTERVALS = int(24 * 60 / INTERVAL_MIN)

N_EV   = 18500        
ETA_C  = 0.16        # fixed value, used only when ETA_C_MODE = "fixed"
P_MAX_KW  = 7.0
P_SITE_MW = 500.0

# Column names in the tariff table of data/tariff.csv
COL_START    = "Start"
COL_END      = "End"
COL_DOMESTIC = "Domestic Rate"
COL_EVCS     = "EV Charging Station Rate"
COL_SURPLUS  = "Suggested Surplus Rate"

T_SW = (9.0, 15.0)
T_NT = (18.0, 6.0)
T_PK = (18.5, 22.5)
T_OP = (22.5, 5.5)
T_DAY = (5.5, 18.5)

ALPHA_S20, ALPHA_S40 = 0.20, 0.40

SURVEY_FILE = "data/survey_responses.csv"
SOLAR_FILE  = "data/solar_surplus.csv"
TARIFF_FILE = "data/tariff.csv"
RESULTS_DIR = "results"

# data/tariff.csv holds TWO tables separated by one or more blank rows:
#   1) tariff periods: Period, Start, End, Domestic Rate, EV Charging Station Rate, Suggested Surplus Rate
#   2) generation costs: Fuel, Unit Cost   (Rs/kWh; rows such as Coal, Furnace Oil, Diesel, Average fuel, Solar)
COL_FUEL       = "Fuel"
COL_UNIT_COST  = "Unit Cost"
SOLAR_COST_KEY = "solar"            # cost row used as the cost of the surplus solar energy (lower case)
AVG_FUEL_KEY   = "average fuel"     # cost row used as the headline fuel cost (lower case)
DAYS_PER_YEAR  = 365                # for the annualised saving
KEEP_COMBINED_PROFILE_GRAPH = True            # also write the old 2x2 panel next to the 4 separate images

USE_PROXY_SOLAR = False
PROXY_S_PEAK = 540.0
PROXY_BESS   = 265.0
PROXY_SIGMA  = 1.6

# solar_surplus.csv timestamps mark the END of each 15-minute interval
# (12:15 = 12:00-12:15). The model's time axis uses interval STARTS, so the
# file is shifted by one interval when loaded. Set False if the file uses starts.
SOLAR_LABEL_IS_END = True

# P4 optimisation
P4_METHOD = "lp"               # "lp" (linear programme) or "heuristic" (original greedy fill)
SEMI_DAYTIME_CHARGING = True   # True = original assumption: semi-flexible users (willing, no daytime access) may charge
                               # in daytime? False = only in the night window T_NT

# ---------------------------------------------------------------------------
# ETA_C model: fleet energy drawn from the grid per km driven (kWh/km)
#
#   For every respondent i (vehicle model from Q_model):
#     c_i   = vehicle consumption at the battery (kWh/km)          VEHICLE_MODELS
#     eta_i = charging efficiency, grid -> battery                 ETA_CHG_AC / _DC (Q_method)
#     UF_i  = share of weekly km driven on grid electricity        1 for BEV; for PHEV
#             min(1, charges/week x usable battery / (weekly km x c_i))  (Q_charge_freq)
#     E_i   = weekly_km_i / 7 x UF_i x c_i x AUX_FACTOR / eta_i    (kWh/day from the grid)
#
#   E_avg = weighted mean of E_i (weights = 1, or FLEET_MIX re-weighting)
#   ETA_C = sum(w_i x weekly_km_i x UF_i x c_i x AUX / eta_i) / sum(w_i x weekly_km_i)
#   Both are computed for the low / mid / high weekly-distance scenarios.
# ---------------------------------------------------------------------------
ETA_C_MODE = "model"      # "model" (derive from survey) or "fixed" (use ETA_C above)

# regex on Q_model (lower case) -> (name, type, battery-side kWh/100 km, usable battery kWh for PHEV)
# kWh/100 km = real-world combined consumption at the battery (e.g. EV Database "real" figures).
# These are starting values - check each against a source you can cite and update.
# None = no reliable figure: the fleet-average of the known BEVs is used.
VEHICLE_MODELS = [
    (r"leaf",                 ("Nissan Leaf",          "BEV",  16.5, None)),
    (r"dolphin",              ("BYD Dolphin",          "BEV",  16.5, None)),
    (r"atto\s*3",             ("BYD Atto 3",           "BEV",  17.5, None)),
    (r"atto\s*2",             ("BYD Atto 2",           "BEV",  16.0, None)),
    (r"atto\s*1|seagull",     ("BYD Atto 1",           "BEV",  14.0, None)),
    (r"\bseal\b(?!ion)",      ("BYD Seal",             "BEV",  17.0, None)),
    (r"mg\s*zs|zs\s*ev",       ("MG ZS EV",             "BEV",  17.5, None)),
    (r"kona",                 ("Hyundai Kona Electric","BEV",  15.5, None)),
    (r"baw",                  ("BAW E7 Pro",           "BEV",  None, None)),
    (r"sea\s*lion\s*5|sealion\s*5", ("BYD Sealion 5 DM-i", "PHEV", 17.0, 17.0)),
    (r"shark",                ("BYD Shark 6 (DMO)",    "PHEV", 24.0, 26.0)),
]
UNKNOWN_MODEL = ("Unknown / not stated", "BEV", None, None)   # e.g. answer "1"

ETA_CHG_AC = 0.88         # AC home / slow charging, grid -> battery (on-board charger + cable)
ETA_CHG_DC = 0.92         # DC fast charging, grid -> battery
AUX_FACTOR = 1.00         # extra for continuous A/C and traffic in Sri Lanka (e.g. 1.05-1.10 as sensitivity)

# Charging frequency (Q_charge_freq) -> charges per week (used for PHEV electric share)
CHARGE_FREQ = {"everyday": 7.0, "every day": 7.0, "every 2 days": 3.5, "every 3-4 days": 2.0,
               "once a week": 1.0, "every 5-7 days": 7.0 / 6.0, "every 2 weeks": 0.5}
DEFAULT_CHARGES_PER_WEEK = 1.0

# Optional re-weighting of the survey to a known national fleet mix, by model name above,
# e.g. {"Nissan Leaf": 0.55, "BYD Dolphin": 0.20, ...}. Models not listed keep weight 1.
# None = every respondent counts equally (the survey's own mix).
FLEET_MIX = None

# ===========================================================================

def time_axis():
    return np.arange(0, 24, DT)

def in_window(t, start, end):
    return (t >= start and t < end) if start < end else (t >= start or t < end)

def window_mask(win):
    s, e = win
    return np.array([in_window(ti, s, e) for ti in time_axis()])

def hhmm_to_hr(s):
    s = str(s).strip(); h, m = s.split(":")
    return int(h) + int(m) / 60.0

def parse_km_range(label):
    if not isinstance(label, str):
        return None
    s = label.lower().replace("km", "").strip()
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", s)]
    if "less than" in s and len(nums) == 1:
        lo, hi = 0.0, nums[0]
    elif ("more than" in s or "above" in s) and len(nums) == 1:
        lo, hi = nums[0], nums[0] * 1.5
    elif len(nums) >= 2:
        lo, hi = nums[0], nums[1]
    else:
        return None
    return lo, (lo + hi) / 2.0, hi

def parse_band(label):
    if not isinstance(label, str):
        return None
    def to_hr(tok):
        m = re.search(r"(\d+)\.?(\d*)\s*(am|pm)", tok.lower())
        if not m:
            return None
        h = int(m.group(1)); mm = int(m.group(2)) if m.group(2) else 0
        ap = m.group(3)
        if ap == "pm" and h != 12: h += 12
        if ap == "am" and h == 12: h = 0
        return h + mm / 60.0
    parts = re.split(r"\bto\b", label.lower())
    if len(parts) != 2:
        return None
    return to_hr(parts[0]), to_hr(parts[1])

def clean(series):
    return (series.astype(str).str.strip()
            .replace({"nan": np.nan, "": np.nan, "None": np.nan}))

# ===========================================================================

def match_vehicle(label):
    s = str(label).strip().lower() if isinstance(label, str) else ""
    for pat, spec in VEHICLE_MODELS:
        if re.search(pat, s):
            return spec
    return UNKNOWN_MODEL

def charges_per_week(label):
    s = str(label).strip().lower() if isinstance(label, str) else ""
    return CHARGE_FREQ.get(s, DEFAULT_CHARGES_PER_WEEK)

def charging_eff(label):
    s = str(label).strip().lower() if isinstance(label, str) else ""
    return ETA_CHG_DC if "fast" in s else ETA_CHG_AC

def derive_energy(df, parsed):
    """Per-respondent grid energy -> E_avg (kWh/day) and ETA_C (kWh/km) per scenario."""
    rows = []
    for i, rng in parsed.items():
        if rng is None:
            continue
        name, vtype, c100, usable = match_vehicle(df.at[i, "Q_model"] if "Q_model" in df else None)
        rows.append({"id": df.at[i, "id"] if "id" in df else i, "model_answer": df.at[i, "Q_model"],
                     "vehicle": name, "type": vtype,
                     "c_kWh_km": np.nan if c100 is None else c100 / 100.0,
                     "usable_kWh": usable,
                     "charges_week": charges_per_week(df.at[i, "Q_charge_freq"] if "Q_charge_freq" in df else None),
                     "eta_charge": charging_eff(df.at[i, "Q_method"] if "Q_method" in df else None),
                     "km_low": rng[0], "km_mid": rng[1], "km_high": rng[2]})
    t = pd.DataFrame(rows)
    known_bev = t[(t["type"] == "BEV") & t["c_kWh_km"].notna()]
    c_default = known_bev["c_kWh_km"].mean() if len(known_bev) else ETA_C
    t["c_source"] = np.where(t["c_kWh_km"].notna(), "table", "fleet BEV average")
    t["c_kWh_km"] = t["c_kWh_km"].fillna(c_default)

    t["weight"] = 1.0
    if FLEET_MIX:
        sample = t["vehicle"].value_counts(normalize=True)
        for v, share in FLEET_MIX.items():
            if v in sample.index:
                t.loc[t["vehicle"] == v, "weight"] = share / sample[v]

    E_avg, eta = {}, {}
    for sc in ("low", "mid", "high"):
        km = t[f"km_{sc}"]
        if ETA_C_MODE == "fixed":
            uf = pd.Series(1.0, index=t.index); grid_km = pd.Series(ETA_C, index=t.index)
        else:
            e_km = km * t["c_kWh_km"]                               # battery kWh/week if all-electric
            cap = t["charges_week"] * t["usable_kWh"].astype(float)  # PHEV battery kWh/week
            uf = np.where((t["type"] == "PHEV") & (e_km > 0), np.minimum(1.0, cap / e_km), 1.0)
            uf = pd.Series(np.nan_to_num(uf, nan=1.0), index=t.index)
            grid_km = t["c_kWh_km"] * AUX_FACTOR / t["eta_charge"]
        E_i = km / 7.0 * uf * grid_km
        t[f"UF_{sc}"] = uf.round(3)
        t[f"E_{sc}_kWh_day"] = E_i.round(3)
        w = t["weight"]
        E_avg[sc] = float((w * E_i).sum() / w.sum())
        eta[sc] = float((w * km * uf * grid_km).sum() / (w * km).sum())
    return E_avg, eta, t, c_default

def load_survey():
    df = pd.read_csv(SURVEY_FILE)
    df = df.dropna(how="all").reset_index(drop=True)
    total = len(df)
    print(f"[AUTO-SCALE] Detected {total} survey responses.")
    for c in ["Q_weekly_km", "Q_start_band", "Q_willing_tou",
              "Q_has_facility", "Q_daytime_location", "Q_charge_now"]:
        if c in df.columns:
            df[c] = clean(df[c])

    parsed = df["Q_weekly_km"].apply(parse_km_range)
    valid = parsed.dropna()
    if len(valid) == 0:
        raise ValueError("No valid Q_weekly_km values parsed.")
    E_avg, eta_c, veh, c_default = derive_energy(df, parsed)
    import os; os.makedirs(RESULTS_DIR, exist_ok=True)
    veh.to_csv(f"{RESULTS_DIR}/eta_c_derivation.csv", index=False)

    f_bl = np.zeros(N_INTERVALS); t = time_axis()
    used, unknown = 0, set()
    for band in df["Q_start_band"].dropna():
        rng = parse_band(band)
        if rng is None or rng[0] is None or rng[1] is None:
            unknown.add(band); continue
        m = np.array([in_window(ti, rng[0], rng[1]) for ti in t])
        if m.sum() > 0:
            f_bl[m] += 1.0 / m.sum(); used += 1
    if used == 0:
        raise ValueError("No valid Q_start_band values parsed.")
    f_bl = f_bl / f_bl.sum()

    def yes_ratio(series):
        s = series.dropna().str.lower()
        return (s == "yes").sum() / len(s) if len(s) else 0.0
    W_sh = yes_ratio(df["Q_willing_tou"])

    loc = df["Q_daytime_location"].fillna("").str.lower()
    fac = df["Q_has_facility"].fillna("").str.lower()
    W_inf = (loc.str.contains("home") | fac.eq("yes")).mean()

    cn = df["Q_charge_now"].fillna("").str.lower()
    answered = cn.str.len() > 0
    home_frac = ((cn.str.contains("home")) & answered).sum() / answered.sum() if answered.sum() else 1.0

    print("=" * 74)
    print("SURVEY DERIVATION")
    print("=" * 74)
    print(f"Responses used                 : {total}")
    if ETA_C_MODE == "fixed":
        print(f"ETA_C (fixed)                  : {ETA_C:.3f} kWh/km")
    else:
        mix = veh["vehicle"].value_counts()
        print("Vehicle mix                    : " + ", ".join(f"{k} {v}" for k, v in mix.items()))
        print(f"BEV default consumption        : {c_default*100:.1f} kWh/100 km (battery side)")
        print(f"ETA_C LOW/MID/HIGH (kWh/km)    : "
              f"{eta_c['low']:.3f} / {eta_c['mid']:.3f} / {eta_c['high']:.3f}  (grid side)")
    print(f"E_avg LOW/MID/HIGH (kWh/day)   : "
          f"{E_avg['low']:.2f} / {E_avg['mid']:.2f} / {E_avg['high']:.2f}")
    print(f"W_sh  (willing to shift)       : {W_sh:.1%}")
    print(f"W_inf (daytime access)         : {W_inf:.1%}")
    print(f"Charge-now HOME / STATION      : {home_frac:.1%} / {1-home_frac:.1%}")
    print("=" * 74)
    return f_bl, E_avg, W_sh, W_inf, home_frac

def load_solar():
    if USE_PROXY_SOLAR:
        t = time_axis()
        s = PROXY_S_PEAK * np.exp(-((t - 12.0) ** 2) / (2 * PROXY_SIGMA ** 2))
        s[s < 5.0] = 0.0
        return np.maximum(0, s - np.minimum(s, PROXY_BESS))
    df = pd.read_csv(SOLAR_FILE)
    net = np.maximum(0, df["surplus_MW"].values - df["bess_dispatch_MW"].values)
    # place each row on the model time axis by its timestamp, not by row order
    t_file = np.array([hhmm_to_hr(x) for x in df["time"]])
    if SOLAR_LABEL_IS_END:
        t_file = (t_file - DT) % 24.0                # label = interval end -> interval start
    idx = np.rint(t_file / DT).astype(int) % N_INTERVALS
    if len(df) != N_INTERVALS or len(set(idx)) != N_INTERVALS:
        raise ValueError(f"{SOLAR_FILE}: expected {N_INTERVALS} rows at {INTERVAL_MIN}-minute "
                         f"steps, found {len(df)} rows / {len(set(idx))} distinct times.")
    S = np.zeros(N_INTERVALS)
    S[idx] = net
    print(f"[SOLAR] {len(df)} rows at {INTERVAL_MIN} min | peak {S.max():.1f} MW | "
          f"{S.sum() * DT:,.1f} MWh/day after BESS")
    return S

def _rate_curve(tdf, column):
    t = time_axis(); c = np.full(N_INTERVALS, np.nan)
    for _, row in tdf.iterrows():
        s = hhmm_to_hr(row[COL_START]); e = hhmm_to_hr(row[COL_END])
        rate = float(row[column])
        for i, ti in enumerate(t):
            if in_window(ti, s, e):
                c[i] = rate
    if np.isnan(c).any():
        c = np.where(np.isnan(c), np.nanmax(c), c)
    return c

def _split_tables(path):
    """Split a CSV that holds several tables separated by blank rows into DataFrames."""
    import io
    with open(path, encoding="utf-8-sig") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    blocks, cur = [], []
    for ln in lines:
        if ln.replace(",", "").strip() == "":
            if cur:
                blocks.append(cur); cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    tables = []
    for blk in blocks:
        df = pd.read_csv(io.StringIO("\n".join(blk)))
        df = df.dropna(axis=1, how="all")                      # drop the empty padding columns
        df.columns = [str(c).strip() for c in df.columns]
        tables.append(df)
    return tables

def _read_tariff_table():
    tables = _split_tables(TARIFF_FILE)
    for t in tables:
        if COL_START in t.columns and COL_END in t.columns:
            return t
    raise ValueError(f"{TARIFF_FILE}: no table with columns '{COL_START}' and '{COL_END}'")

def load_generation_costs():
    """Return {fuel: Rs/kWh} from the cost table in TARIFF_FILE (keys lower-case, stripped)."""
    tables = _split_tables(TARIFF_FILE)
    cdf = next((t for t in tables if COL_FUEL in t.columns and COL_UNIT_COST in t.columns), None)
    if cdf is None:
        raise ValueError(f"{TARIFF_FILE}: no table with columns '{COL_FUEL}' and '{COL_UNIT_COST}'")
    costs = {str(k).strip().lower(): float(v) for k, v in zip(cdf[COL_FUEL], cdf[COL_UNIT_COST]) if pd.notna(v)}
    for key in (SOLAR_COST_KEY, AVG_FUEL_KEY):
        if key not in costs:
            raise ValueError(f"cost table has no row '{key}'; found {list(costs)}")
    print("[COST] " + " | ".join(f"{k} {v:.1f}" for k, v in costs.items()) + " Rs/kWh")
    return costs

def load_tariffs():
    tdf = _read_tariff_table()
    curves = {"domestic": _rate_curve(tdf, COL_DOMESTIC),
              "evcs":     _rate_curve(tdf, COL_EVCS),
              "surplus":  _rate_curve(tdf, COL_SURPLUS)}
    print(f"[TARIFF] {len(tdf)} periods loaded | "
          f"domestic {curves['domestic'].min():.0f}-{curves['domestic'].max():.0f} | "
          f"evcs {curves['evcs'].min():.0f}-{curves['evcs'].max():.0f} | "
          f"surplus {curves['surplus'].min():.0f}-{curves['surplus'].max():.0f} Rs/kWh")
    return curves

# ===========================================================================

def a1_baseline(f_bl, E_avg):
    D = (N_EV * E_avg * f_bl) / DT / 1000.0
    R_nt = f_bl[window_mask(T_NT)].sum()
    return D, N_EV * R_nt, R_nt

def a_shift(D_BL, N_nt, E_avg, S_res, alpha):
    N_sh = alpha * N_nt
    E_sh = N_sh * E_avg
    sw = window_mask(T_SW)
    f_sw = np.zeros(N_INTERVALS)
    if S_res[sw].sum() > 0:
        f_sw[sw] = S_res[sw] / S_res[sw].sum()
    E_max = S_res[sw].sum() * DT * 1000.0
    E_act = min(E_sh, E_max); E_ovf = max(0, E_sh - E_max)
    # Spread E_act in proportion to the surplus, capped at the shifted fleet's
    # power; energy clipped by the cap is re-spread over the remaining
    # uncapped intervals (water-filling) so no energy is lost.
    cap_sh = (N_sh * P_MAX_KW) / 1000.0
    D_add = np.zeros(N_INTERVALS); left = E_act; free = sw & (S_res > 0)
    for _ in range(N_INTERVALS):
        if left <= 1e-9 or not free.any(): break
        w = np.where(free, S_res, 0.0)
        room = np.where(free, cap_sh - D_add, 0.0)
        want = (left * w / w.sum()) / DT / 1000.0
        step = np.minimum(want, room)
        D_add += step; left -= step.sum() * DT * 1000.0
        free &= D_add < cap_sh - 1e-12
    E_ovf += max(0.0, left)
    D_rem = np.zeros(N_INTERVALS); E_to_remove = E_sh
    t = time_axis()
    pk_idx = [i for i, ti in enumerate(t) if in_window(ti, *T_PK)]
    op_idx = [i for i, ti in enumerate(t) if in_window(ti, *T_OP)]
    nt_rest = [i for i, ti in enumerate(t)
               if in_window(ti, *T_NT) and i not in pk_idx and i not in op_idx]
    for group in (pk_idx, op_idx, nt_rest):
        for i in group:
            if E_to_remove <= 0: break
            avail = D_BL[i] * DT * 1000.0
            take = min(avail, E_to_remove)
            D_rem[i] += take / DT / 1000.0
            E_to_remove -= take
    D_day = np.zeros(N_INTERVALS)
    if E_ovf > 0:
        do = window_mask((T_DAY[0], T_SW[0])) | window_mask((T_SW[1], T_PK[0]))
        h = do.sum() * DT
        if h > 0: D_day[do] = (E_ovf / h) / 1000.0
    return D_BL - D_rem + D_add + D_day

def a4_heuristic(D_BL, E_avg, S_res, W_sh, W_inf, tariffs=None):
    """Original P4: greedy fill (surplus hours first, then day, off-peak, peak)."""
    N_flex = N_EV * W_sh * W_inf
    N_semi = N_EV * W_sh * (1 - W_inf)
    N_fix  = N_EV * (1 - W_sh)
    D_fix = D_BL * (N_fix / N_EV)
    E_rem = (N_flex + N_semi) * E_avg
    D_flex = np.zeros(N_INTERVALS)
    cap0 = ((N_flex + N_semi) * P_MAX_KW) / 1000.0
    t = time_axis()
    isw = sorted([i for i, ti in enumerate(t) if in_window(ti, *T_SW)],
                 key=lambda i: S_res[i], reverse=True)
    iday = [i for i, ti in enumerate(t) if in_window(ti, *T_DAY) and i not in isw]
    iop = [i for i, ti in enumerate(t) if in_window(ti, *T_OP)]
    ipk = [i for i, ti in enumerate(t) if in_window(ti, *T_PK)]
    def fill(ix, sur=False):
        nonlocal E_rem
        for i in ix:
            if E_rem <= 0: break
            cap = min(cap0, P_SITE_MW - D_fix[i] - D_flex[i])
            if sur: cap = min(cap, S_res[i])
            if cap <= 0: continue
            add = min(cap, (E_rem / DT) / 1000.0)
            D_flex[i] += add; E_rem -= add * DT * 1000.0
    fill(isw, True); fill(iday); fill(iop); fill(ipk)
    return D_fix + D_flex

def a4_lp(D_BL, E_avg, S_res, W_sh, W_inf, tariffs):
    """P4 as a linear programme (two stages, solved with HiGHS).

    Groups (as in the heuristic):
      fixed : N_EV*(1-W_sh), keeps the baseline shape
      flex  : N_EV*W_sh*W_inf, willing and with daytime access - any interval
      semi  : N_EV*W_sh*(1-W_inf), willing, no daytime access - night window
              only (unless SEMI_DAYTIME_CHARGING)
    Constraints: each group delivers exactly its daily energy; group power
    <= N*P_MAX; total load <= P_SITE_MW.
    Stage 1 maximises the solar surplus absorbed, sum(min(D, S_res)).
    Stage 2 keeps that absorption and minimises the cost at the surplus tariff;
    ties are broken towards higher surplus and the solar window T_SW.
    Stage 3 keeps both and spreads the flexible load in proportion to the
    surplus, instead of stacking it in a few intervals.
    """
    n = N_INTERVALS; t = time_axis()
    N_flex = N_EV * W_sh * W_inf
    N_semi = N_EV * W_sh * (1 - W_inf)
    N_fix  = N_EV * (1 - W_sh)
    D_fix = D_BL * (N_fix / N_EV)
    E_flex = N_flex * E_avg / 1000.0           # MWh
    E_semi = N_semi * E_avg / 1000.0
    cap_flex = N_flex * P_MAX_KW / 1000.0      # MW
    cap_semi = N_semi * P_MAX_KW / 1000.0
    semi_ok = np.ones(n, bool) if SEMI_DAYTIME_CHARGING else window_mask(T_NT)

    # variables: [x_flex (n) | x_semi (n) | u (n)]  u = absorbed surplus
    bounds = ([(0, cap_flex)] * n
              + [(0, cap_semi if semi_ok[i] else 0) for i in range(n)]
              + [(0, max(0.0, S_res[i])) for i in range(n)])
    I = np.eye(n); Z = np.zeros((n, n))
    A_ub = np.vstack([np.hstack([I, I, Z]),            # site limit
                      np.hstack([-I, -I, I])])         # u <= D_fix + x_flex + x_semi
    b_ub = np.concatenate([P_SITE_MW - D_fix, D_fix])
    A_eq = np.vstack([np.concatenate([np.full(n, DT), np.zeros(2 * n)]),
                      np.concatenate([np.zeros(n), np.full(n, DT), np.zeros(n)])])
    b_eq = np.array([E_flex, E_semi])

    c1 = np.concatenate([np.zeros(2 * n), -np.ones(n) * DT])
    r1 = linprog(c1, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r1.status != 0:
        raise RuntimeError(f"stage 1: {r1.message}")
    absorbed = -r1.fun

    # Tie-break (tiny, below 1 Rs per MW per interval): among intervals with the
    # same tariff, prefer those with more surplus and inside the solar window T_SW.
    price = tariffs["surplus"] * DT * 1000.0            # Rs per MW in each interval
    pref = S_res / S_res.max() if S_res.max() > 0 else np.zeros(n)
    tie = -0.5 * pref - 0.5 * window_mask(T_SW)
    c2 = np.concatenate([price + tie, price + tie, np.zeros(n)])
    A_ub2 = np.vstack([A_ub, np.concatenate([np.zeros(2 * n), -np.ones(n) * DT])])
    b_ub2 = np.concatenate([b_ub, [-(absorbed - 1e-6 * max(1.0, absorbed))]])
    r2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r2.status != 0:
        x = r1.x
    else:
        x = r2.x
        # Stage 3: an LP optimum is often "bang-bang" (all flexible charging
        # piled into one or two intervals at the full fleet power). Keeping the
        # absorption and cost of stage 2, spread the flexible load: minimise
        # the largest share of the surplus it uses (r), plus its largest load
        # in hours without surplus (m, relative to the fleet power).
        # extra variables: r, m
        cost2 = float(np.concatenate([price, price]) @ x[:2 * n])
        has_s = S_res > 1e-9
        cap_all = max(cap_flex + cap_semi, 1e-9)
        rows, rhs = [], []
        for i in range(n):
            row = np.zeros(3 * n + 2); row[i] = 1; row[n + i] = 1
            if has_s[i]:
                row[3 * n] = -S_res[i]          # x_i <= r * S_res_i
            else:
                row[3 * n + 1] = -1             # x_i <= m
            rows.append(row); rhs.append(0.0)
        pad = lambda M: np.hstack([M, np.zeros((M.shape[0], 2))])
        cost_row = np.concatenate([price, price, np.zeros(n), [0, 0]])
        # keep stage 2's placement inside the solar window: energy outside T_SW
        # may not grow (it only spreads within the hours stage 2 chose)
        out_sw = (~window_mask(T_SW)).astype(float)
        e_out2 = float(DT * (out_sw @ (x[:n] + x[n:2 * n])))
        out_row = np.concatenate([out_sw * DT, out_sw * DT, np.zeros(n), [0, 0]])
        A_ub3 = np.vstack([pad(A_ub2), np.array(rows), cost_row, out_row])
        b_ub3 = np.concatenate([b_ub2, rhs, [cost2 + 1e-6 * max(1.0, cost2) + 1.0],
                                [e_out2 + 1e-6 * max(1.0, e_out2)]])
        c3 = np.concatenate([np.zeros(3 * n), [1.0, 1.0 / cap_all]])
        r3 = linprog(c3, A_ub=A_ub3, b_ub=b_ub3, A_eq=pad(A_eq), b_eq=b_eq,
                     bounds=bounds + [(0, None), (0, None)], method="highs")
        if r3.status == 0:
            x = r3.x[:3 * n]
    D = D_fix + x[:n] + x[n:2 * n]
    return np.where(np.abs(D) < 1e-9, 0.0, D)


def a4_optimised(D_BL, E_avg, S_res, W_sh, W_inf, tariffs):
    if P4_METHOD == "lp" and HAVE_SCIPY:
        try:
            return a4_lp(D_BL, E_avg, S_res, W_sh, W_inf, tariffs)
        except Exception as ex:
            print(f"[P4] LP failed ({ex}) - using the heuristic instead")
    elif P4_METHOD == "lp":
        print("[P4] scipy not installed - using the heuristic instead")
    return a4_heuristic(D_BL, E_avg, S_res, W_sh, W_inf, tariffs)

def profile_cost(name, D, tariffs, home_frac):
    if name == "P1_BL":
        blended = home_frac * tariffs["domestic"] + (1 - home_frac) * tariffs["evcs"]
        return np.sum(blended * D * DT * 1000.0)
    return np.sum(tariffs["surplus"] * D * DT * 1000.0)

def surplus_window_energy(D):
    """Energy (MWh) charged INSIDE the solar-surplus window T_SW (09:00-15:00)."""
    return np.sum(D[window_mask(T_SW)]) * DT   # MW * h = MWh

def a5_kpis(profiles, S_res, tariffs, home_frac):
    nt, pk, op = window_mask(T_NT), window_mask(T_PK), window_mask(T_OP)
    D_BL = profiles["P1_BL"]
    cbl = profile_cost("P1_BL", D_BL, tariffs, home_frac)
    ntbl = np.sum(D_BL[nt]) * DT; pkbl = np.sum(D_BL[pk]) * DT
    opbl = np.sum(D_BL[op]) * DT
    sw_bl = surplus_window_energy(D_BL)          # baseline surplus-window energy
    ea_bl = np.sum(np.minimum(D_BL, S_res) * DT) # baseline energy already served by surplus
    rows = []
    for nm, D in profiles.items():
        Ea = np.sum(np.minimum(D, S_res) * DT); Et = np.sum(D * DT)
        cost = profile_cost(nm, D, tariffs, home_frac)
        sw_energy = surplus_window_energy(D)     # total charged in window
        shifted_in = sw_energy - sw_bl           # extra vs baseline (the shift)
        rows.append({
            "Profile": nm,
            "Tariff_Basis": ("home/EVCS blend" if nm == "P1_BL" else "surplus"),
            "Coverage_%": round(Ea / Et * 100, 1) if Et else 0,
            "Curtail_Absorbed_MWh": round(Ea, 2),
            "Extra_Absorbed_vs_BL_MWh": round(Ea - ea_bl, 2),   # surplus energy that replaces fuel generation
            "SurplusWindow_Energy_MWh": round(sw_energy, 2),   # NEW
            "Shifted_into_Surplus_MWh": round(shifted_in, 2),  # NEW
            "Cost_Rs": round(cost, 0),
            "Cost_Saving_vs_BL_%": round((cbl - cost) / cbl * 100, 1) if cbl else 0,
            "Night_Reduction_%": round((ntbl - np.sum(D[nt]) * DT) / ntbl * 100, 1) if ntbl else 0,
            "Peak_Reduction_%": round((pkbl - np.sum(D[pk]) * DT) / pkbl * 100, 1) if pkbl else 0,
            "OffPeak_Reduction_%": round((opbl - np.sum(D[op]) * DT) / opbl * 100, 1) if opbl else 0,
        })
    return pd.DataFrame(rows)

def ceb_saving(kpi, costs):
    """
    CEB generation-cost saving per profile.
    The energy that a profile moves out of fuel-supplied hours and into hours where it is
    served by otherwise unused solar surplus is Extra_Absorbed_vs_BL_MWh (increase in surplus
    energy absorbed, relative to the baseline). That energy was previously generated by fuel.
      gross saving  = E_shift x fuel unit cost           (fuel no longer burned)
      net saving    = E_shift x (fuel unit cost - solar unit cost)
                      (CEB still pays the feed-in tariff for the solar energy that now has a use)
    Both are given per fuel type and for the average fuel unit cost, per day and per year.
    """
    fuels = {k: v for k, v in costs.items() if k != SOLAR_COST_KEY}
    solar = costs[SOLAR_COST_KEY]
    rows = []
    for _, r in kpi.iterrows():
        e_kwh = max(0.0, r["Extra_Absorbed_vs_BL_MWh"]) * 1000.0
        row = {"Profile": r["Profile"], "Fuel_Displaced_MWh_day": round(e_kwh / 1000.0, 2)}
        for f, c in fuels.items():
            tag = f.replace(AVG_FUEL_KEY, "avg_fuel").replace(" ", "_")
            row[f"Gross_{tag}_Rs_day"] = round(e_kwh * c, 0)
            row[f"Net_{tag}_Rs_day"] = round(e_kwh * (c - solar), 0)
        row["Net_avg_fuel_Rs_year"] = round(e_kwh * (costs[AVG_FUEL_KEY] - solar) * DAYS_PER_YEAR, 0)
        rows.append(row)
    return pd.DataFrame(rows)

def run_scenario(E_val, f_bl, S_res, tariffs, W_sh, W_inf, home_frac):
    D_BL, N_nt, R_nt = a1_baseline(f_bl, E_val)
    profiles = {
        "P1_BL": D_BL,
        "P2_S20": a_shift(D_BL, N_nt, E_val, S_res, ALPHA_S20),
        "P3_S40": a_shift(D_BL, N_nt, E_val, S_res, ALPHA_S40),
        "P4_OPT": a4_optimised(D_BL, E_val, S_res, W_sh, W_inf, tariffs),
    }
    e_bl = D_BL.sum() * DT
    for nm, D in profiles.items():                  # every profile must deliver the same energy
        if abs(D.sum() * DT - e_bl) > 1e-6 * max(1.0, e_bl):
            print(f"[CHECK] {nm}: {D.sum() * DT:.3f} MWh vs baseline {e_bl:.3f} MWh")
    return profiles, a5_kpis(profiles, S_res, tariffs, home_frac), N_nt

# ===========================================================================

COLORS = {"P1_BL": "#1565c0", "P2_S20": "#e07b00",
          "P3_S40": "#7b1fa2", "P4_OPT": "#c62828"}
TITLES = {"P1_BL": "P1: Baseline", "P2_S20": "P2: 20% Night Shift",
          "P3_S40": "P3: 40% Night Shift", "P4_OPT": "P4: Optimized"}

def g1_charging_shape(f_bl):
    t = time_axis()
    plt.figure(figsize=(9, 4.5))
    plt.bar(t, f_bl, width=DT*0.9, color="#1565c0", alpha=0.8)
    plt.axvspan(*T_SW, alpha=0.12, color="gold", label=f"Solar window {int(T_SW[0]):02d}:00-{int(T_SW[1]):02d}:00")
    plt.title("Graph 1 - Baseline Charging Shape f_bl(t)")
    plt.xlabel("Hour of Day"); plt.ylabel("Fraction")
    plt.xticks(range(0, 25, 3)); plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/1_survey_charging_shape.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g2_tariff(tariffs):
    t = time_axis()
    plt.figure(figsize=(9, 4.5))
    for key, lbl in [("domestic", COL_DOMESTIC), ("evcs", COL_EVCS),
                     ("surplus", COL_SURPLUS)]:
        plt.step(np.append(t, 24), np.append(tariffs[key], tariffs[key][-1]),
                 where="post", lw=2, label=lbl)
    plt.axvspan(*T_SW, alpha=0.10, color="gold")
    plt.title("Graph 2 - Tariff Structures (domestic / EVCS / surplus)")
    plt.xlabel("Hour of Day"); plt.ylabel("Rs/kWh")
    plt.xticks(range(0, 25, 3)); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/2_tariff_structure.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g3_solar(S_res):
    t = time_axis()
    plt.figure(figsize=(9, 4.5))
    plt.fill_between(t, 0, S_res, alpha=0.35, color="green")
    plt.plot(t, S_res, color="green", lw=2)
    plt.axvspan(*T_SW, alpha=0.12, color="gold")
    plt.title("Graph 3 - Residual Solar Surplus S_res(t)")
    plt.xlabel("Hour of Day"); plt.ylabel("Power (MW)")
    plt.xticks(range(0, 25, 3)); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/3_solar_surplus_curve.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g4_profiles(profiles, S_res):
    """One image per profile (4a-4d), plus the combined 2x2 panel if KEEP_COMBINED_PROFILE_GRAPH."""
    t = time_axis()
    ymax = max(S_res.max(), max(D.max() for D in profiles.values())) * 1.08
    for letter, (nm, D) in zip("abcd", profiles.items()):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.fill_between(t, 0, S_res, alpha=0.22, color="green", label="Residual solar surplus")
        ax.plot(t, D, color=COLORS[nm], lw=2.4, label="EV charging load")
        ax.axvspan(*T_SW, alpha=0.10, color="gold",
                   label=f"Surplus window {int(T_SW[0]):02d}:00-{int(T_SW[1]):02d}:00")
        ax.set_title(f"Graph 4{letter} - {TITLES[nm]} vs Solar Surplus (mid)", fontweight="bold")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Power (MW)")
        ax.set_xlim(0, 24); ax.set_ylim(0, ymax); ax.set_xticks(range(0, 25, 3))
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/4{letter}_profile_{nm}.png",
                                        dpi=150, bbox_inches="tight"); plt.close()
    if not KEEP_COMBINED_PROFILE_GRAPH:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5)); axes = axes.flatten()
    for ax, (nm, D) in zip(axes, profiles.items()):
        ax.fill_between(t, 0, S_res, alpha=0.22, color="green", label="Solar Surplus")
        ax.plot(t, D, color=COLORS[nm], lw=2.2, label="EV Charging")
        ax.axvspan(*T_SW, alpha=0.10, color="gold")
        ax.set_title(TITLES[nm], fontweight="bold")
        ax.set_xlabel("Hour"); ax.set_ylabel("Power (MW)")
        ax.set_xticks(range(0, 25, 6)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Graph 4 - Charging Profiles vs Solar Surplus (mid)",
                 fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/4_profile_curves.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g5_coverage(kpi):
    plt.figure(figsize=(8, 5))
    b = plt.bar(kpi["Profile"], kpi["Coverage_%"], color=list(COLORS.values()))
    for bar, v in zip(b, kpi["Coverage_%"]):
        plt.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v}%", ha="center", fontweight="bold")
    plt.title("Graph 5 - Solar Surplus Coverage by Profile (mid)")
    plt.ylabel("Coverage (%)"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/5_coverage_by_profile.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g6_cost(kpi, days_per_month=30):
    """
    Consumer charging cost, shown as Rs per EV per month (left) and the saving against the
    baseline in Rs per EV per month, with the % in the bar label (right).
    P1 is costed on the home/EVCS blend; P2-P4 on the surplus tariff.
    """
    per_ev_month = kpi["Cost_Rs"] / N_EV * days_per_month
    saving = per_ev_month.iloc[0] - per_ev_month
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    b = ax1.bar(kpi["Profile"], per_ev_month, color=list(COLORS.values()))
    for bar, v in zip(b, per_ev_month):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"Rs {v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax1.axhline(per_ev_month.iloc[0], color="grey", ls="--", lw=1, label="Baseline")
    ax1.set_ylabel(f"Charging cost (Rs per EV per month, {days_per_month} days)")
    ax1.set_title("Monthly charging cost per EV")
    ax1.set_ylim(0, per_ev_month.max() * 1.15); ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    sub = kpi.iloc[1:]
    y = np.arange(len(sub))
    cols = [COLORS[p] for p in sub["Profile"]]
    ax2.barh(y, saving.iloc[1:], color=cols)
    for yi, sv, pc in zip(y, saving.iloc[1:], sub["Cost_Saving_vs_BL_%"]):
        ax2.text(sv + max(saving.max(), 1) * 0.01, yi, f"Rs {sv:,.0f}  ({pc:+.1f}%)", va="center", fontsize=9)
    ax2.set_yticks(y); ax2.set_yticklabels(sub["Profile"]); ax2.invert_yaxis()
    ax2.set_xlabel("Saving vs baseline (Rs per EV per month)")
    ax2.set_title("Saving against the baseline")
    ax2.set_xlim(0, max(saving.max(), 1) * 1.45); ax2.grid(axis="x", alpha=0.3)
    fig.suptitle("Graph 6 - Consumer Charging Cost (P1: home/EVCS blend | P2-P4: surplus tariff)", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/6_cost_by_profile.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g10_ceb_saving(sav, costs):
    """CEB net generation-cost saving per profile, by displaced fuel, Rs million per day."""
    fuels = [k for k in costs if k != SOLAR_COST_KEY]
    sub = sav.iloc[1:]                             # baseline saves nothing
    x = np.arange(len(sub)); w = 0.8 / len(fuels)
    palette = {"coal": "#4d4d4d", "furnace oil": "#8c6d31", "diesel": "#d95f0e",
               AVG_FUEL_KEY: "#1565c0"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, f in enumerate(fuels):
        tag = f.replace(AVG_FUEL_KEY, "avg_fuel").replace(" ", "_")
        vals = sub[f"Net_{tag}_Rs_day"] / 1e6
        lbl = ("Average fuel" if f == AVG_FUEL_KEY else f.title()) + f" ({costs[f]:.1f} Rs/kWh)"
        bars = ax.bar(x + (i - (len(fuels) - 1) / 2) * w, vals, w, label=lbl,
                      color=palette.get(f, None), edgecolor="black" if f == AVG_FUEL_KEY else "none")
        if f == AVG_FUEL_KEY:
            for bar, v, yr in zip(bars, vals, sub["Net_avg_fuel_Rs_year"] / 1e6):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                        f"{v:.2f}\n(Rs {yr:,.0f} M/yr)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sub["Profile"])
    ax.set_ylabel("Net saving to CEB (Rs million per day)")
    ax.set_title(f"Graph 10 - CEB Generation-Cost Saving from Shifted Load (mid)\n"
                 f"saving = displaced energy x (fuel unit cost - solar {costs[SOLAR_COST_KEY]:.1f} Rs/kWh)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/10_ceb_saving.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g7_shifted(kpi):
    """NEW: energy charged in the surplus window, and how much of that is the SHIFT."""
    x = np.arange(len(kpi)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # total energy in surplus window
    bars = ax.bar(x, kpi["SurplusWindow_Energy_MWh"], w*1.6,
                  color=list(COLORS.values()), alpha=0.55,
                  label="Total charged in surplus window")
    # portion that is the extra shifted-in energy vs baseline
    ax.bar(x, kpi["Shifted_into_Surplus_MWh"], w*1.6,
           color=list(COLORS.values()), edgecolor="black", hatch="//",
           label="Shifted-in vs baseline")
    for xi, tot, sh in zip(x, kpi["SurplusWindow_Energy_MWh"],
                           kpi["Shifted_into_Surplus_MWh"]):
        ax.text(xi, tot + 0.3, f"{tot:.1f}", ha="center", fontweight="bold", fontsize=9)
        if sh > 0.05:
            ax.text(xi, sh/2, f"+{sh:.1f}", ha="center", fontsize=8, color="black")
    ax.set_xticks(x); ax.set_xticklabels(kpi["Profile"])
    ax.set_ylabel("Energy (MWh/day)")
    ax.set_title("Graph 7 - Load Shifted INTO Solar-Surplus Window (MWh)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/7_shifted_into_surplus.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g8_reduction(kpi):
    x = np.arange(len(kpi)); w = 0.26
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, kpi["Night_Reduction_%"], w, label="Night", color="#2c7fb8")
    plt.bar(x,     kpi["Peak_Reduction_%"],  w, label="Peak",  color="#d95f0e")
    plt.bar(x + w, kpi["OffPeak_Reduction_%"], w, label="Off-peak", color="#31a354")
    plt.xticks(x, kpi["Profile"]); plt.ylabel("Reduction vs Baseline (%)")
    plt.title("Graph 8 - Night / Peak / Off-Peak Load Reduction (mid)")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/8_load_reduction.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

def g9_sensitivity(cov):   # FINAL graph
    x = np.arange(len(cov)); w = 0.25
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - w, cov["Coverage_LOW_%"],  w, label="Low",  color="#9ecae1")
    plt.bar(x,     cov["Coverage_MID_%"],  w, label="Mid",  color="#3182bd")
    plt.bar(x + w, cov["Coverage_HIGH_%"], w, label="High", color="#08519c")
    plt.xticks(x, cov["Profile"]); plt.ylabel("Coverage (%)")
    plt.title("Graph 9 (FINAL) - Coverage Sensitivity to Weekly-Distance")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/9_coverage_sensitivity.png",
                                    dpi=150, bbox_inches="tight"); plt.close()

# ===========================================================================

def main():
    f_bl, E_avg, W_sh, W_inf, home_frac = load_survey()
    S_res = load_solar()
    tariffs = load_tariffs()
    costs = load_generation_costs()
    print(f"Night-charging fraction R_nt   : {f_bl[window_mask(T_NT)].sum():.1%}")

    all_kpis, profiles_mid = {}, None
    for scen in ["low", "mid", "high"]:
        profiles, kpi, N_nt = run_scenario(
            E_avg[scen], f_bl, S_res, tariffs, W_sh, W_inf, home_frac)
        all_kpis[scen] = kpi
        if scen == "mid": profiles_mid = profiles
        print("\n" + "#" * 74)
        print(f"SCENARIO {scen.upper()}  (E_avg={E_avg[scen]:.2f} kWh/day, N_nt={N_nt:,.0f})")
        print("#" * 74)
        print(kpi.to_string(index=False))
        kpi.to_csv(f"{RESULTS_DIR}/results_{scen}.csv", index=False)
        sav = ceb_saving(kpi, costs)
        all_kpis[scen + "_saving"] = sav
        print("\nCEB saving (Rs/day; net = fuel cost - solar cost):")
        print(sav.to_string(index=False))
        sav.to_csv(f"{RESULTS_DIR}/ceb_saving_{scen}.csv", index=False)

    cov = pd.DataFrame({
        "Profile": all_kpis["mid"]["Profile"],
        "Coverage_LOW_%":  all_kpis["low"]["Coverage_%"].values,
        "Coverage_MID_%":  all_kpis["mid"]["Coverage_%"].values,
        "Coverage_HIGH_%": all_kpis["high"]["Coverage_%"].values,
    })
    cov.to_csv(f"{RESULTS_DIR}/coverage_sensitivity.csv", index=False)

    print("\nGenerating graphs (major first, sensitivity last)...")
    g1_charging_shape(f_bl);          print("  [1/10] charging shape")
    g2_tariff(tariffs);               print("  [2/10] tariff structures")
    g3_solar(S_res);                  print("  [3/10] solar surplus")
    g4_profiles(profiles_mid, S_res); print("  [4/10] profile curves (4a-4d, one image each)")
    g5_coverage(all_kpis["mid"]);     print("  [5/10] coverage")
    g6_cost(all_kpis["mid"]);         print("  [6/10] cost")
    g7_shifted(all_kpis["mid"]);      print("  [7/10] SHIFTED INTO SURPLUS (MWh)  <-- NEW")
    g8_reduction(all_kpis["mid"]);    print("  [8/10] load reduction")
    g9_sensitivity(cov);              print("  [9/10] coverage sensitivity")
    g10_ceb_saving(all_kpis["mid_saving"], costs); print("  [10/10] CEB generation-cost saving  <-- NEW")

    print("\nDone. Results, CEB saving tables and graphs saved to 'results/'.")


if __name__ == "__main__":
    main()
