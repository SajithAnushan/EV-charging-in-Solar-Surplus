 EV Charging Framework: How to Run It

## Before you start

**1. Install the libraries (once).** Open Command Prompt and run:
```
pip install pandas numpy matplotlib scipy pvlib openpyxl requests
```

**2. Put all the scripts in one folder**, for example `C:\Users\SAAULK\Desktop\EV_Framework`, and create a subfolder called `data` inside it.

**3. Open Command Prompt inside that folder.** In File Explorer, open the folder, click the address bar, type `cmd`, press Enter. Run every command below from there. If `python` is not recognised, type `py` instead.

**4. Check the file paths.** Steps 1 to 8 have paths at the top of each script, in the block marked `SETTINGS` (lines such as `INPUT_FILE` and `OUTPUT_DIR`). Open each script once and make sure they point to your folders.

**5. Files you must already have:**
- `Solar_Radiation_All_Stations.xlsx`: the raw station radiation data
- The survey export (`.xlsx`) from the Google Form
- `tariff.csv`: copy it into the `data` folder

---

## Part A. Solar potential (do once)

**Step 1. Station curves**
- Run: `python solar_daily_curves.py`
- Needs: `Solar_Radiation_All_Stations.xlsx`
- Creates: one `<Station>_daily_curves.xlsx` per station, in the `Daily_Curves` folder
- Check: it prints "usable days" for each of the ten stations

**Step 2. National curve**
- Run: `python national_curve.py`
- Needs: `Solar_Radiation_All_Stations.xlsx`, and `solar_daily_curves.py` in the same folder
- Creates: `National_daily_curves.xlsx`, in the `Daily_Curves` folder
- Check: it prints "national usable days"

**Step 3. Potential in MW (10-minute)**
- Run: `python generation_10min_excel.py`
- Needs: the files from steps 1 and 2
- Creates: `Solar_Generation_10min_MW.xlsx`. This takes a few minutes. You do not need to open it.

**Step 4. Potential in MW (15-minute)**
- Run: `python convert_mean_to_15min.py`
- Needs: the file from step 3
- Creates: `National_Mean_15min_MW.xlsx`
- Check: the two annual energy figures it prints are the same


---

## Part B. Surplus

**Step 7. Excess = potential minus generation**
- Run: `python solar_excess.py`
- Needs: the files from steps 4 and 5
- Creates: `Solar_Excess_vs_Mean.xlsx`
- Check: it prints the share of intervals refilled, the excess energy and the peak. Write these down for the thesis.

**Step 8. Surplus table for the model**
- Run: `python make_solar_surplus.py`
- Needs: the files from steps 6 and 7
- Creates: `data\solar_surplus.csv`
- Check: it prints the surplus energy and the BESS energy per average day. Write these down too.

---

## Part C. Survey

**Step 9. Survey to CSV**
- Run: `python build_survey_csv.py`
- Needs: the survey export `.xlsx`
- Creates: `data\survey_responses.csv`
- Check: it prints counts for distance, start time and willingness
- To share the file, set `KEEP_CONTACT = False` in the script and run it again. This removes names and emails.

---

## Part D. The model

Before running, confirm that the `data` folder contains `solar_surplus.csv`, `survey_responses.csv` and `tariff.csv`.

**Step 10. Deterministic results**
- Run: `python charging_framework.py`
- Creates: a `results` folder with the KPI tables, the CEB saving tables and graphs 1 to 10

**Step 11. Monte Carlo**
- First open `monte_carlo.py` and set `EXCESS_FILE` (near the top) to the full path of `Solar_Excess_vs_Mean.xlsx`.
- Quick test first: `python monte_carlo.py 20`
- Full run: `python monte_carlo.py`. This takes much longer.
- Creates: a `results_mc` folder with the KPI summaries, profile bands, uncertainty study and four graphs
- Check: open `results_mc\mc_validation.txt`. It compares the simulation with the deterministic results.

**Step 12. Optional: export the Monte Carlo curves to CSV**
- Run: `python npz_to_csv.py`
- Needs: `results_mc\mc_profiles_uniform.npz`
- Creates: `results_mc\csv\`

---

## If something changes
- New survey data: run step 9, then steps 10 and 11.
- New tariff or model settings: run steps 10 and 11 only.
- New solar or BESS data: run from step 5 or 6 onwards.
- Steps 1 to 4 do not need repeating unless the station data change.

## Common errors
| Message | What to do |
|---|---|
| `ModuleNotFoundError` | Run the `pip install` line again |
| `KeyError: 'Start'` | `data\tariff.csv` is the wrong file or has different headers |
| `OSError: [Errno 22]` on a PNG | Close the image, delete the `results` folder, run again |
| Monte Carlo says "surplus kept fixed" | `EXCESS_FILE` path is wrong |
