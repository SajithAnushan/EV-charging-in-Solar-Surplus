import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import charging_framework as cf

# =============================================================================
# SETTINGS
# =============================================================================
N_REPS = 1000                 # replicates for the main run
SEED = 20261018               # fixed seed = reproducible results

# weekly-km treatment: "uniform" = random inside each respondent's range;
# "low" / "mid" / "high" = the framework's fixed scenarios. Several can be listed.
KM_MODES = ["uniform"]

SURVEY_UNCERTAINTY = True     # Bayesian bootstrap of respondents (small survey)
VEHICLE_UNCERTAINTY = True    # consumption (per model) and charging efficiency
SURPLUS_UNCERTAINTY = True    # block bootstrap of days for the surplus curve

# solar surplus bootstrap (needs the dated excess workbook from solar_excess.py)
EXCESS_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Dataset\New folder\Generation_Potential\Solar_Excess_vs_Mean.xlsx"
EXCESS_SHEET = "Excess_filled"
BLOCK_DAYS = 7                # days per resampled block (keeps weather persistence)

# charging session model
DURATION_MODE = "survey"      # "survey": stated Q_duration, at least energy/power
                              # "power" : energy / charging power
                              # "band"  : whole start band, reproduces the framework's
                              #           spreading (use to check the MC mean = deterministic)
BAND_FROM_SAME_RESPONDENT = True   # True : an EV's start band comes from the same respondent as its
                              #        km and vehicle (heavy users' start times count more)
                              # False: start band drawn independently of energy, as the
                              #        framework's f_bl does (needed for the like-for-like check)
START_JITTER_MIN = 0.0        # optional smoothing of start times (minutes, sd); 0 = off
DC_POWER_KW = 50.0            # power cap for respondents who use fast (DC) charging
BEV_MAX_SESSION_KWH = 60.0    # grid-side energy one BEV session can deliver (assumed);
                              # heavy low-frequency drivers need extra sessions
CONSUMPTION_CV = 0.10         # uncertainty of each model's kWh/km table value
EFFICIENCY_SD = 0.02          # uncertainty of the AC / DC charging efficiency

RUN_ATTRIBUTION = True        # which uncertainty source matters most?
N_REPS_ATTR = 150             # replicates per configuration of that study

BAND = (5, 95)                # percentile band shown in graphs and summaries
RESULTS_DIR_MC = "results_mc"
# =============================================================================

DT = cf.DT
N = cf.N_INTERVALS
N_EV = cf.N_EV
PROFILES = ["P1_BL", "P2_S20", "P3_S40", "P4_OPT"]
MIN_PER_DAY = 1440
SLOT_MIN = MIN_PER_DAY // N


# ----------------------------------------------------------------------------- survey
def parse_duration(label):
    """'3 hours' -> 3.0; anything without a usable number (e.g. 'Overnight') -> NaN."""
    if not isinstance(label, str):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*hours?\s*$", label.lower())
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if 0 < v <= 24 else np.nan


def build_respondents():
    """One row of attributes per survey respondent (same parsing as the framework)."""
    df = pd.read_csv(cf.SURVEY_FILE).dropna(how="all").reset_index(drop=True)
    n = len(df)
    for c in ["Q_weekly_km", "Q_start_band", "Q_willing_tou", "Q_has_facility",
              "Q_daytime_location", "Q_charge_now", "Q_duration"]:
        if c in df.columns:
            df[c] = cf.clean(df[c])

    def col(name):
        return df[name] if name in df.columns else pd.Series([np.nan] * n)

    # weekly km range
    rng_ = col("Q_weekly_km").apply(cf.parse_km_range)
    km = np.array([r if r is not None else (np.nan,) * 3 for r in rng_], dtype=float)
    km_ok = np.isfinite(km[:, 0])
    if not km_ok.any():
        raise ValueError("No valid Q_weekly_km values parsed.")

    # vehicle
    spec = [cf.match_vehicle(m) for m in col("Q_model")]
    names = np.array([s[0] for s in spec])
    is_phev = np.array([s[1] == "PHEV" for s in spec])
    c_tab = np.array([np.nan if s[2] is None else s[2] / 100.0 for s in spec])
    usable = np.array([np.nan if s[3] is None else float(s[3]) for s in spec])
    known_bev = (~is_phev) & np.isfinite(c_tab) & km_ok
    c_default = float(c_tab[known_bev].mean()) if known_bev.any() else cf.ETA_C
    model_names, model_idx = np.unique(names, return_inverse=True)

    n_week = np.array([cf.charges_per_week(v) for v in col("Q_charge_freq")], dtype=float)
    is_dc = np.array([cf.charging_eff(v) == cf.ETA_CHG_DC for v in col("Q_method")])

    # start band
    b_start, b_len = np.full(n, np.nan), np.full(n, np.nan)
    for i, b in enumerate(col("Q_start_band")):
        r = cf.parse_band(b)
        if r is None or r[0] is None or r[1] is None:
            continue
        ln = (r[1] - r[0]) % 24.0
        b_start[i], b_len[i] = r[0], (ln if ln > 0 else 24.0)
    band_ok = np.isfinite(b_start)
    if not band_ok.any():
        raise ValueError("No valid Q_start_band values parsed.")

    dur = np.array([parse_duration(v) for v in col("Q_duration")], dtype=float)

    # willingness / access / where they charge now (same definitions as the framework)
    s = col("Q_willing_tou").dropna().index
    willing = np.full(n, np.nan)
    willing[s] = (col("Q_willing_tou").loc[s].str.lower() == "yes").astype(float)
    loc = col("Q_daytime_location").fillna("").str.lower()
    fac = col("Q_has_facility").fillna("").str.lower()
    access = (loc.str.contains("home") | fac.eq("yes")).values.astype(float)
    cn = col("Q_charge_now").fillna("").str.lower()
    home = np.where(cn.str.len() > 0, cn.str.contains("home").astype(float), np.nan)

    base_w = np.ones(n)
    if cf.FLEET_MIX:                       # optional re-weighting, as in derive_energy
        share = pd.Series(names[km_ok]).value_counts(normalize=True)
        for v, sh in cf.FLEET_MIX.items():
            if v in share.index:
                base_w[names == v] = sh / share[v]
    base_w = base_w / base_w.sum()

    return dict(n=n, km_lo=km[:, 0], km_mid=km[:, 1], km_hi=km[:, 2], km_ok=km_ok,
                names=names, model_idx=model_idx, n_models=len(model_names),
                is_phev=is_phev, c_tab=c_tab, c_default=c_default, usable=usable,
                n_week=n_week, is_dc=is_dc, b_start=b_start, b_len=b_len, band_ok=band_ok,
                dur=dur, dur_ok=np.isfinite(dur), willing=willing,
                will_ok=np.isfinite(willing), access=access, home=home,
                home_ok=np.isfinite(home), base_w=base_w)


# ----------------------------------------------------------------------------- surplus
def to_axis(v_file):
    """Put 96 values in file order (0:00 ... 23:45) on the model time axis, as load_solar does."""
    t_file = np.arange(N) * DT
    if cf.SOLAR_LABEL_IS_END:
        t_file = (t_file - DT) % 24.0
    idx = np.rint(t_file / DT).astype(int) % N
    out = np.zeros(N)
    out[idx] = v_file
    return out


def load_surplus_source():
    """Deterministic curve plus (if available) the dated days used for the bootstrap."""
    S0 = cf.load_solar()
    src = {"X": None}
    if cf.USE_PROXY_SOLAR or not SURPLUS_UNCERTAINTY:
        return S0, src
    if not os.path.isfile(EXCESS_FILE):
        print(f"[MC] {EXCESS_FILE} not found - surplus kept fixed (no day bootstrap)")
        return S0, src
    X = pd.read_excel(EXCESS_FILE, sheet_name=EXCESS_SHEET, index_col=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    if X.shape[1] != N:
        print(f"[MC] excess sheet has {X.shape[1]} columns, expected {N} - surplus kept fixed")
        return S0, src
    sol = pd.read_csv(cf.SOLAR_FILE)
    bess = sol["bess_dispatch_MW"].values.astype(float)
    mean_curve = X.mean(axis=0)
    gap = np.abs(np.round(mean_curve, 2) - sol["surplus_MW"].values).max()
    if gap > 1.0:
        print(f"[MC] WARNING: workbook mean differs from solar_surplus.csv by up to {gap:.1f} MW - "
              f"re-run solar_excess.py and make_solar_surplus.py so both use the same data")
    src.update(X=X, bess=bess, n_days=len(X))
    print(f"[MC] surplus bootstrap uses {len(X)} dated days (blocks of {BLOCK_DAYS})")
    return S0, src


# ----------------------------------------------------------------------------- masks
def masks():
    return {k: cf.window_mask(v) for k, v in
            {"NT": cf.T_NT, "PK": cf.T_PK, "OP": cf.T_OP, "SW": cf.T_SW}.items()}


MASK = None


# ----------------------------------------------------------------------------- random inputs
def pick_cdf(u, w):
    cw = np.cumsum(w / w.sum())
    return np.minimum(np.searchsorted(cw, u, side="right"), len(w) - 1)


def fallback_pick(valid, w, base, u):
    """Keep the sampled respondent if the answer exists, else draw another who answered."""
    if valid.all() or not valid.any():
        return base
    idx = base.copy()
    bad = ~valid[base]
    if bad.any():
        idx[bad] = pick_cdf(u[bad], np.where(valid, w, 0.0))
    return idx


def draw_randoms(rng, R, src):
    """All random numbers of one replicate. Always drawn in the same order, so the
    uncertainty-source study uses common random numbers across configurations."""
    d = {"g": rng.exponential(1.0, R["n"]),
         "z_model": rng.normal(size=R["n_models"]),
         "z_eta": rng.normal(size=2)}
    nb = int(np.ceil(src["n_days"] / BLOCK_DAYS)) if src["X"] is not None else 1
    hi = max(1, src["n_days"] - BLOCK_DAYS + 1) if src["X"] is not None else 1
    d["block_start"] = rng.integers(0, hi, nb)
    for k in ("base", "core", "band", "band2", "will", "home", "km", "charge", "start"):
        d["u_" + k] = rng.random(N_EV)
    d["jitter"] = rng.normal(size=N_EV)
    return d


# ----------------------------------------------------------------------------- one simulated day
def build_load(start_h, dur_h, energy_kwh):
    """Fleet load (MW per slot) from sessions with constant power over [start, start+duration]."""
    m0 = np.floor(start_h * 60.0).astype(int) % MIN_PER_DAY
    n = np.clip(np.rint(dur_h * 60.0).astype(int), 1, MIN_PER_DAY)
    p = energy_kwh / (n / 60.0)                                  # kW, energy is conserved exactly
    size = 2 * MIN_PER_DAY + 1
    diff = np.bincount(m0, weights=p, minlength=size) - np.bincount(m0 + n, weights=p, minlength=size)
    kw = np.cumsum(diff)[:2 * MIN_PER_DAY]
    day = kw[:MIN_PER_DAY] + kw[MIN_PER_DAY:]                    # sessions running past midnight wrap
    return day.reshape(N, SLOT_MIN).mean(axis=1) / 1000.0


def simulate_day(R, d, km_mode, flags):
    n = R["n"]
    w = R["base_w"] * d["g"] if flags["survey"] else R["base_w"].copy()
    w = w / w.sum()

    base = pick_cdf(d["u_base"], w)
    core = fallback_pick(R["km_ok"], w, base, d["u_core"])      # km, vehicle, frequency, method
    b_base = base if BAND_FROM_SAME_RESPONDENT else pick_cdf(d["u_band"], w)
    band = fallback_pick(R["band_ok"], w, b_base, d["u_band2"])  # start band, stated duration
    will = fallback_pick(R["will_ok"], w, base, d["u_will"])
    home = fallback_pick(R["home_ok"], w, base, d["u_home"])

    # weekly km
    lo, mid, hi = R["km_lo"][core], R["km_mid"][core], R["km_hi"][core]
    if km_mode == "uniform":
        km = lo + d["u_km"] * (hi - lo)
    else:
        km = {"low": lo, "mid": mid, "high": hi}[km_mode]

    # consumption (battery side) and charging efficiency
    c = np.where(np.isfinite(R["c_tab"][core]), R["c_tab"][core], R["c_default"])
    if flags["vehicle"] and CONSUMPTION_CV > 0:
        s = np.sqrt(np.log(1 + CONSUMPTION_CV ** 2))
        c = c * np.exp(s * d["z_model"][R["model_idx"][core]] - 0.5 * s * s)
    eta_ac, eta_dc = cf.ETA_CHG_AC, cf.ETA_CHG_DC
    if flags["vehicle"]:
        eta_ac = float(np.clip(eta_ac + EFFICIENCY_SD * d["z_eta"][0], 0.75, 0.97))
        eta_dc = float(np.clip(eta_dc + EFFICIENCY_SD * d["z_eta"][1], 0.75, 0.98))
    is_dc = R["is_dc"][core]
    eta = np.where(is_dc, eta_dc, eta_ac)

    # daily grid energy per EV (same formula as the framework, PHEV utility factor included)
    n_w = R["n_week"][core]
    e_km = km * c
    if cf.ETA_C_MODE == "fixed":
        uf = np.ones(N_EV); grid_km = np.full(N_EV, cf.ETA_C)
    else:
        cap_w = n_w * np.nan_to_num(R["usable"][core], nan=0.0)
        uf = np.where(R["is_phev"][core] & (e_km > 0), np.minimum(1.0, cap_w / np.maximum(e_km, 1e-9)), 1.0)
        grid_km = c * cf.AUX_FACTOR / eta
    E = km / 7.0 * uf * grid_km                                  # kWh/day (average over days)

    # does the EV charge today? probability = sessions per week / 7
    cap_s = np.where(R["is_phev"][core], np.nan_to_num(R["usable"][core], nan=BEV_MAX_SESSION_KWH) / eta,
                     BEV_MAX_SESSION_KWH)
    p = np.clip(np.maximum(n_w, 7.0 * E / cap_s) / 7.0, 1e-6, 1.0)
    sel = d["u_charge"] < p
    e_s = (E / p)[sel]                                           # energy of today's session

    # session start and length
    b0, bl = R["b_start"][band][sel], R["b_len"][band][sel]
    p_cap = np.where(is_dc[sel], DC_POWER_KW, cf.P_MAX_KW)
    d_min = e_s / p_cap
    if DURATION_MODE == "band":
        start, dur = b0, bl
    else:
        start = b0 + d["u_start"][sel] * bl
        if START_JITTER_MIN > 0:
            start = start + d["jitter"][sel] * START_JITTER_MIN / 60.0
        if DURATION_MODE == "survey":
            stated = R["dur"][band][sel]
            dur = np.minimum(np.maximum(np.where(np.isfinite(stated), stated, d_min), d_min), 24.0)
        else:
            dur = d_min
    D_BL = build_load(start % 24.0, dur, e_s)

    return dict(D_BL=D_BL, W_sh=float(np.nanmean(R["willing"][will])),
                W_inf=float(R["access"][base].mean()), home_frac=float(np.nanmean(R["home"][home])),
                energy_mwh=float(e_s.sum() / 1000.0), n_sessions=int(sel.sum()),
                E_avg=float(E.mean()))


def draw_surplus(S0, src, d, flags):
    if not flags["surplus"] or src["X"] is None:
        return S0
    X, B = src["X"], BLOCK_DAYS
    idx = (d["block_start"][:, None] + np.arange(B)).ravel()[:src["n_days"]]
    mean_curve = X[idx].mean(axis=0)
    return to_axis(np.maximum(0.0, mean_curve - src["bess"]))


# ----------------------------------------------------------------------------- profiles and KPIs
def build_profiles(fleet, S, tariffs):
    D_BL = fleet["D_BL"]
    e_tot = D_BL.sum()
    E_avg = e_tot * DT * 1000.0 / N_EV                           # realised kWh per EV
    R_nt = D_BL[MASK["NT"]].sum() / e_tot
    N_nt = N_EV * R_nt
    prof = {"P1_BL": D_BL,
            "P2_S20": cf.a_shift(D_BL, N_nt, E_avg, S, cf.ALPHA_S20),
            "P3_S40": cf.a_shift(D_BL, N_nt, E_avg, S, cf.ALPHA_S40),
            "P4_OPT": cf.a4_optimised(D_BL, E_avg, S, fleet["W_sh"], fleet["W_inf"], tariffs)}
    return prof, R_nt


def kpi_rows(rep, prof, S, tariffs, home_frac, R_nt):
    nt, pk, op, sw = MASK["NT"], MASK["PK"], MASK["OP"], MASK["SW"]
    D0 = prof["P1_BL"]
    cbl = cf.profile_cost("P1_BL", D0, tariffs, home_frac)
    ntbl, pkbl, opbl = D0[nt].sum() * DT, D0[pk].sum() * DT, D0[op].sum() * DT
    swbl = D0[sw].sum() * DT
    rows = []
    for nm, D in prof.items():
        Et = D.sum() * DT
        Ea = np.minimum(D, S).sum() * DT
        cost = cf.profile_cost(nm, D, tariffs, home_frac)
        swe = D[sw].sum() * DT
        rows.append({"rep": rep, "Profile": nm,
                     "Coverage_%": Ea / Et * 100 if Et else 0.0,
                     "Curtail_Absorbed_MWh": Ea,
                     "SurplusWindow_Energy_MWh": swe,
                     "Shifted_into_Surplus_MWh": swe - swbl,
                     "Cost_Rs": cost,
                     "Cost_Saving_vs_BL_%": (cbl - cost) / cbl * 100 if cbl else 0.0,
                     "Night_Reduction_%": (ntbl - D[nt].sum() * DT) / ntbl * 100 if ntbl else 0.0,
                     "Peak_Reduction_%": (pkbl - D[pk].sum() * DT) / pkbl * 100 if pkbl else 0.0,
                     "OffPeak_Reduction_%": (opbl - D[op].sum() * DT) / opbl * 100 if opbl else 0.0,
                     "Peak_Load_MW": D.max(),
                     "Peak_Hour": float(np.argmax(D) * DT),
                     "Energy_MWh": Et})
    return rows


# ----------------------------------------------------------------------------- the Monte Carlo loop
def run_mc(R, S0, src, tariffs, km_mode, flags, n_reps, seed, label=""):
    rng = np.random.default_rng(seed)
    curves = {p: np.zeros((n_reps, N)) for p in PROFILES}
    S_all = np.zeros((n_reps, N))
    rows, meta = [], []
    t0 = time.time()
    for r in range(n_reps):
        d = draw_randoms(rng, R, src)
        fleet = simulate_day(R, d, km_mode, flags)
        S = draw_surplus(S0, src, d, flags)
        prof, R_nt = build_profiles(fleet, S, tariffs)
        for p in PROFILES:
            curves[p][r] = prof[p]
        S_all[r] = S
        rows += kpi_rows(r, prof, S, tariffs, fleet["home_frac"], R_nt)
        meta.append({"rep": r, "E_avg_kWh": fleet["E_avg"], "Energy_MWh": fleet["energy_mwh"],
                     "Sessions": fleet["n_sessions"], "W_sh": fleet["W_sh"], "W_inf": fleet["W_inf"],
                     "R_nt": R_nt, "home_frac": fleet["home_frac"]})
        if (r + 1) % max(1, n_reps // 10) == 0:
            print(f"  {label} {r + 1}/{n_reps} replicates  ({time.time() - t0:,.0f} s)")
    return {"curves": curves, "S": S_all, "kpi": pd.DataFrame(rows), "meta": pd.DataFrame(meta)}


def summarise(kpi):
    cols = [c for c in kpi.columns if c not in ("rep", "Profile")]
    lo, hi = BAND
    out = []
    for nm, g in kpi.groupby("Profile", sort=False):
        for c in cols:
            v = g[c].values
            out.append({"Profile": nm, "KPI": c, "Mean": v.mean(), "SD": v.std(ddof=1),
                        f"P{lo}": np.percentile(v, lo), "P50": np.percentile(v, 50),
                        f"P{hi}": np.percentile(v, hi)})
    return pd.DataFrame(out)


def band_table(curves, S_all):
    lo, hi = BAND
    t = cf.time_axis()
    out = {"hour": t}
    for p, A in {**curves, "Surplus_net": S_all}.items():
        out[f"{p}_mean"] = A.mean(axis=0)
        out[f"{p}_P{lo}"] = np.percentile(A, lo, axis=0)
        out[f"{p}_P50"] = np.percentile(A, 50, axis=0)
        out[f"{p}_P{hi}"] = np.percentile(A, hi, axis=0)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- validation
def validate(res, det, mode):
    lines = []
    m = res["meta"]
    det_e = det["E_avg"] * N_EV / 1000.0
    mc_e = m["Energy_MWh"].mean()
    se = m["Energy_MWh"].std(ddof=1) / np.sqrt(len(m))
    lines.append(f"[{mode}] fleet energy per day: MC mean {mc_e:,.2f} MWh (SE {se:.2f}) vs "
                 f"deterministic {det_e:,.2f} MWh  ->  difference {100 * (mc_e / det_e - 1):+.2f}%")
    lines.append(f"[{mode}] realised energy of P1..P4 equal within each replicate: "
                 f"max spread {res['kpi'].groupby('rep')['Energy_MWh'].agg(lambda x: x.max() - x.min()).max():.2e} MWh")
    lines.append(f"[{mode}] night fraction R_nt: MC {m['R_nt'].mean():.3f} vs deterministic {det['R_nt']:.3f}")
    lines.append(f"[{mode}] W_sh: MC {m['W_sh'].mean():.3f} (sd {m['W_sh'].std():.3f}) vs {det['W_sh']:.3f};  "
                 f"W_inf: MC {m['W_inf'].mean():.3f} (sd {m['W_inf'].std():.3f}) vs {det['W_inf']:.3f}")
    p1 = res["curves"]["P1_BL"].mean(axis=0)
    rmse = np.sqrt(np.mean((p1 - det["profiles"]["P1_BL"]) ** 2))
    lines.append(f"[{mode}] P1 shape: RMSE MC mean vs deterministic {rmse:.2f} MW, "
                 f"peak {p1.max():.1f} vs {det['profiles']['P1_BL'].max():.1f} MW")
    return lines


# ----------------------------------------------------------------------------- graphs
def g_bands(res, det_prof, mode):
    lo, hi = BAND
    t = cf.time_axis()
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    Sm = res["S"].mean(axis=0)
    for ax, p in zip(axs.ravel(), PROFILES):
        A = res["curves"][p]
        ax.fill_between(t, np.percentile(A, lo, axis=0), np.percentile(A, hi, axis=0),
                        color=cf.COLORS[p], alpha=0.25, label=f"P{lo}-P{hi} band")
        ax.plot(t, A.mean(axis=0), color=cf.COLORS[p], lw=2, label="MC mean")
        ax.plot(t, det_prof[p], color="black", lw=1, ls="--", label="Deterministic")
        ax.axvspan(*cf.T_SW, alpha=0.10, color="gold")
        ax.set_title(cf.TITLES[p]); ax.grid(alpha=0.3)
        ax.set_xticks(range(0, 25, 3))
    for ax in axs[1]:
        ax.set_xlabel("Hour of day")
    for ax in axs[:, 0]:
        ax.set_ylabel("Fleet load (MW)")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Monte Carlo charging profiles ({len(res['meta'])} replicates, km mode: {mode})")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_MC}/mc_1_profile_bands.png", dpi=150, bbox_inches="tight"); plt.close()


def g_kpis(res, mode):
    k = res["kpi"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    spec = [("Coverage_%", PROFILES, "Coverage of surplus (%)"),
            ("Cost_Saving_vs_BL_%", PROFILES[1:], "Cost saving vs baseline (%)"),
            ("Shifted_into_Surplus_MWh", PROFILES[1:], "Shifted into surplus window (MWh)")]
    for ax, (c, profs, lab) in zip(axs, spec):
        data = [k.loc[k.Profile == p, c].values for p in profs]
        bp = ax.boxplot(data, tick_labels=[p.split("_")[0] for p in profs], patch_artist=True,
                        whis=BAND, showfliers=False)
        for patch, p in zip(bp["boxes"], profs):
            patch.set_facecolor(cf.COLORS[p]); patch.set_alpha(0.55)
        ax.set_title(lab); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Distribution of KPIs (box = quartiles, whiskers = P{BAND[0]}-P{BAND[1]}), km mode: {mode}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_MC}/mc_2_kpi_distributions.png", dpi=150, bbox_inches="tight"); plt.close()


def g_peaks(res, mode):
    k = res["kpi"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [k.loc[k.Profile == p, "Peak_Load_MW"].values for p in PROFILES]
    bp = axs[0].boxplot(data, tick_labels=[p.split("_")[0] for p in PROFILES], patch_artist=True,
                        whis=BAND, showfliers=False)
    for patch, p in zip(bp["boxes"], PROFILES):
        patch.set_facecolor(cf.COLORS[p]); patch.set_alpha(0.55)
    axs[0].set_title("Peak fleet load (MW)"); axs[0].grid(axis="y", alpha=0.3)
    h = k.loc[k.Profile == "P1_BL", "Peak_Hour"].values
    axs[1].hist(h, bins=np.arange(0, 24.5, 0.5), color=cf.COLORS["P1_BL"], alpha=0.7)
    axs[1].set_title("Hour of the baseline (P1) peak"); axs[1].set_xlabel("Hour of day")
    axs[1].set_ylabel("Replicates"); axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_MC}/mc_3_peak_load.png", dpi=150, bbox_inches="tight"); plt.close()


def g_attribution(att):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, c, lab in zip(axs, ["Coverage_%", "Cost_Saving_vs_BL_%"],
                          ["SD of P4 coverage (% points)", "SD of P4 cost saving (% points)"]):
        d = att[att.KPI == c]
        ax.barh(d["Configuration"], d["SD"], color="#c62828", alpha=0.7)
        ax.invert_yaxis(); ax.set_title(lab); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_MC}/mc_4_uncertainty_sources.png", dpi=150, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------- main
def deterministic_reference(S0, tariffs):
    """The framework's own 'mid' run, for comparison."""
    f_bl, E_avg, W_sh, W_inf, home_frac = cf.load_survey()
    prof, kpi, _ = cf.run_scenario(E_avg["mid"], f_bl, S0, tariffs, W_sh, W_inf, home_frac)
    return {"E_avg": E_avg["mid"], "R_nt": float(f_bl[MASK["NT"]].sum()), "W_sh": W_sh,
            "W_inf": W_inf, "profiles": prof, "kpi": kpi}


def main():
    global MASK, N_REPS
    if len(sys.argv) > 1:
        N_REPS = int(sys.argv[1])
    os.makedirs(RESULTS_DIR_MC, exist_ok=True)
    MASK = masks()

    R = build_respondents()
    S0, src = load_surplus_source()
    tariffs = cf.load_tariffs()
    print(f"[MC] {R['n']} survey respondents, {N_EV:,} EVs per replicate, {N_REPS} replicates, "
          f"duration mode '{DURATION_MODE}'")
    det = deterministic_reference(S0, tariffs)

    full = {"survey": SURVEY_UNCERTAINTY, "vehicle": VEHICLE_UNCERTAINTY, "surplus": SURPLUS_UNCERTAINTY}
    report = []
    for k_i, mode in enumerate(KM_MODES):
        print(f"\n=== Monte Carlo, weekly-km mode '{mode}' ===")
        res = run_mc(R, S0, src, tariffs, mode, full, N_REPS, SEED + k_i, label=mode)
        summ = summarise(res["kpi"])
        res["kpi"].to_csv(f"{RESULTS_DIR_MC}/mc_kpis_{mode}.csv", index=False)
        res["meta"].to_csv(f"{RESULTS_DIR_MC}/mc_replicate_inputs_{mode}.csv", index=False)
        summ.to_csv(f"{RESULTS_DIR_MC}/mc_kpi_summary_{mode}.csv", index=False)
        band_table(res["curves"], res["S"]).to_csv(f"{RESULTS_DIR_MC}/mc_profile_bands_{mode}.csv", index=False)
        np.savez_compressed(f"{RESULTS_DIR_MC}/mc_profiles_{mode}.npz", surplus=res["S"], **res["curves"])
        report += validate(res, det, mode)

        lo, hi = BAND
        show = summ[summ.KPI.isin(["Coverage_%", "Cost_Saving_vs_BL_%", "Shifted_into_Surplus_MWh", "Peak_Load_MW"])]
        print(f"\nKPI summary, mode '{mode}' (mean [P{lo}, P{hi}])")
        for kpi_name, g in show.groupby("KPI", sort=False):
            txt = ",  ".join(f"{r.Profile.split('_')[0]} {r.Mean:,.1f} [{r[f'P{lo}']:,.1f}, {r[f'P{hi}']:,.1f}]"
                             for _, r in g.iterrows())
            print(f"  {kpi_name:26s} {txt}")
        if k_i == 0:
            g_bands(res, det["profiles"], mode); g_kpis(res, mode); g_peaks(res, mode)

    if RUN_ATTRIBUTION:
        print("\n=== Which uncertainty source matters? (P4) ===")
        mode0 = KM_MODES[0]
        off = {"survey": False, "vehicle": False, "surplus": False}
        configs = [("Individual randomness only", off, "mid"),
                   ("+ survey sampling", {**off, "survey": True}, "mid"),
                   ("+ weekly km range", off, "uniform"),
                   ("+ consumption / efficiency", {**off, "vehicle": True}, "mid"),
                   ("+ surplus day sampling", {**off, "surplus": True}, "mid"),
                   ("All sources", full, mode0)]
        rows = []
        for name, fl, km_mode in configs:
            r_ = run_mc(R, S0, src, tariffs, km_mode, fl, N_REPS_ATTR, SEED + 99, label=name[:18])
            k4 = r_["kpi"][r_["kpi"].Profile == "P4_OPT"]
            for c in ["Coverage_%", "Cost_Saving_vs_BL_%", "Shifted_into_Surplus_MWh"]:
                rows.append({"Configuration": name, "KPI": c, "Mean": k4[c].mean(), "SD": k4[c].std(ddof=1)})
        att = pd.DataFrame(rows)
        att.to_csv(f"{RESULTS_DIR_MC}/mc_uncertainty_sources.csv", index=False)
        g_attribution(att)
        print(att.pivot(index="Configuration", columns="KPI", values="SD").round(3)
              .reindex([c[0] for c in configs]).to_string())

    with open(f"{RESULTS_DIR_MC}/mc_validation.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\nValidation against the deterministic framework")
    print("\n".join("  " + x for x in report))
    print(f"\nDone. Results and graphs saved to '{RESULTS_DIR_MC}/'.")


if __name__ == "__main__":
    main()
