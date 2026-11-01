import re
import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
INPUT_FILE = r"C:\Users\SAAULK\Desktop\Masters\Research\Charging patterns of EV owners in Sri Lanka (Responses).xlsx"
OUTPUT_FILE = r"data\survey_responses.csv"      # the framework reads data/survey_responses.csv
KEEP_CONTACT = True    # False blanks name and e-mail (for sharing the file)
# =============================================================================

# CSV column -> phrase that appears in the form question (case-insensitive)
COLUMNS = {
    "share_contact":          "like to share your name",
    "name":                   "name",                       # matched exactly (see below)
    "email":                  "e-mail address",
    "Q_model":                "vehicle model",
    "Q_weekly_km":            "weekly commute distance",
    "Q_charge_freq":          "how often do you charge your vehicle?",
    "Q_method":               "what is your preferred charging method",
    "Q_duration":             "how long do you usually charge",
    "Q_start_band":           "mostly start the charging",
    "Q_daytime_location":     "where does your vehicle stay",
    "Q_has_facility":         "charging facilities at the parking location",
    "Q_facility_freq":        "charge your vehicle from charging facility",
    "Q_reasons":              "reasons to never or rarely charge",
    "Q_charge_now":           "where do you mostly charge",
    "Q_willing_parking_day":  "would you charge the vehicle from parking facility",
    "Q_rooftop_solar":        "rooftop solar at your home",
    "Q_grid_connected":       "connected to the national grid",
    "Q_plan_solar":           "invest on rooftop solar",
    "Q_plan_bess":            "battery energy storage system",
    "Q_willing_tou":          "time of use tariff",
    "Q_infra_expect":         "infrastructure improvements",
    "Q_can_charge_day":       "if your charge is low",
}


def find_column(cols, key, phrase):
    norm = {c: " ".join(str(c).lower().split()) for c in cols}
    if key == "name":
        hits = [c for c, n in norm.items() if n == "name"]
    else:
        hits = [c for c, n in norm.items() if phrase in n]
    if len(hits) != 1:
        raise ValueError(f"{key}: expected one form column containing '{phrase}', found {len(hits)}")
    return hits[0]


def tidy(v):
    if pd.isna(v):
        return ""
    return " ".join(str(v).split())


def tidy_duration(v):
    """'4 hours with', '3 hrs', '2hrs', '5' -> 'N hours'; a range '9 to 10' -> upper value.
    Text answers without a number (e.g. 'Overnight') and values above 24 h
    (e.g. '80-90', probably a battery %) are kept as written."""
    t = tidy(v)
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", t)]
    return f"{max(nums):g} hours" if nums and max(nums) <= 24 else t


def main():
    inp = Path(sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE)
    out = Path(sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE)
    if not inp.is_file():
        print(f"Input file not found: {inp}")
        return

    df = pd.read_excel(inp, sheet_name=0)
    df = df.dropna(how="all")
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp", kind="stable")
    df = df.reset_index(drop=True)

    res = pd.DataFrame({"id": range(1, len(df) + 1)})
    for key, phrase in COLUMNS.items():
        res[key] = df[find_column(df.columns, key, phrase)].map(tidy).values
    res["Q_duration"] = df[find_column(df.columns, "Q_duration", COLUMNS["Q_duration"])].map(tidy_duration).values
    if not KEEP_CONTACT:
        res["name"] = ""
        res["email"] = ""

    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8")

    print(f"{len(res)} responses written to {out}")
    for col in ("Q_weekly_km", "Q_start_band", "Q_willing_tou"):
        counts = res[col].replace("", "(blank)").value_counts()
        print(f"  {col}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
