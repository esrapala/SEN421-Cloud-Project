"""
SEN421 Topic 12 - data preparation (Huawei Public Cloud Trace 2025, Region 2).

Steps
  1. Download the Region 2 trigger file and per-minute time series (skipped if already done)
  2. Extract 5 metrics for HTTP (APIG-S) functions only and save them as Parquet
  3. Build a per-function summary table
  4. Classify functions (>= 1,000 requests/day) as steady / diurnal / bursty and plot examples
  5. Compute statistics of the three representative functions

Usage in Colab:  %run prepare_data.py
Data source: https://github.com/sir-lab/data-release (CC BY 4.0)
"""
import os, zipfile, subprocess
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------------- Paths ----------------
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass
OUT = os.environ.get("SEN421_OUT", "/content/drive/MyDrive/SEN421/data")
os.makedirs(OUT, exist_ok=True)
ZIP = "/content/region2_quantiles.zip"
TRIG = f"{OUT}/df_funcID_runtime_triggerType.csv"
GDRIVE_IDS = {"zip": "1iZk-n_5IT7j3M_pHpRzhwSPTuJ2vxvlR",       # Region 2 quantiles (2.7 GB)
              "trig": "1dUg_yxeLR5OyldivPZGW5djYF31166EG"}      # Region 2 runtimes and trigger types
METRICS = ["requests", "totalCost_avg", "memory_usage_avg", "num_pods", "num_cold_starts"]
DAYS, MIN_PER_DAY = 31, 1440

# ---------------- Step 1-2: download and extract (skipped if Parquet files exist) ----------------
if not all(os.path.exists(f"{OUT}/{m}.parquet") for m in METRICS):
    if not os.path.exists(TRIG):
        subprocess.run(["gdown", GDRIVE_IDS["trig"], "-O", TRIG], check=True)
    if not os.path.exists(ZIP):
        subprocess.run(["gdown", GDRIVE_IDS["zip"], "-O", ZIP], check=True)
    trig = pd.read_csv(TRIG)
    http_funcs = set(trig.loc[trig["triggerType-invocationType"] == "APIG-S", "funcID"])
    print("HTTP (APIG-S) functions:", len(http_funcs))
    keep = lambda c: c in ("day", "time") or c in http_funcs
    z = zipfile.ZipFile(ZIP)
    for m in METRICS:
        parts = []
        for d in range(DAYS):
            df = pd.read_csv(z.open(f"R2/{m}/day_{d:02d}.csv"), usecols=keep)
            df.index = df["day"] * MIN_PER_DAY + df["time"] // 60          # minute index 0..44639
            parts.append(df.drop(columns=["day", "time"]).astype("float32"))
        full = pd.concat(parts)
        if m in ("requests", "num_cold_starts"):
            full = full.fillna(0)                                            # NaN = no request in that minute
        full.to_parquet(f"{OUT}/{m}.parquet")
        print(m, full.shape, "saved")

# ---------------- Step 3: per-function summary ----------------
req = pd.read_parquet(f"{OUT}/requests.parquet")
dur = pd.read_parquet(f"{OUT}/totalCost_avg.parquet")        # seconds
mem = pd.read_parquet(f"{OUT}/memory_usage_avg.parquet")     # MB

s = pd.DataFrame(index=req.columns)
s["user"] = s.index.str.split("---").str[1]
s["mem_limit_mb"] = s.index.str.extract(r"-(\d+)$", expand=False).astype(int).values
s["cpu_limit_cores"] = s.index.str.extract(r"-(\d+)-\d+$", expand=False).astype(int).values / 1000
s["total_req"] = req.sum()
s["req_per_day"] = s["total_req"] / DAYS
s["mean_rpm"] = req.mean()                                  # mean requests per minute
s["peak_rpm"] = req.max()                                   # busiest minute
s["peak_to_mean"] = s["peak_rpm"] / s["mean_rpm"]
s["active_share"] = (req > 0).mean()                        # share of minutes with >= 1 request
s["avg_dur_ms"] = (req * dur).sum() / s["total_req"] * 1000 # request-weighted mean execution time
s["avg_mem_mb"] = (req * mem).sum() / s["total_req"]        # request-weighted mean memory usage
s["daily_autocorr"] = req.apply(lambda c: c.autocorr(lag=MIN_PER_DAY) if c.std() > 0 else np.nan)
s.to_csv(f"{OUT}/function_summary.csv")

act = s[s.total_req > 0]
cols = ["req_per_day", "peak_to_mean", "active_share", "avg_dur_ms", "avg_mem_mb", "daily_autocorr"]
print("HTTP functions with requests:", len(act), "/", len(s), "| users:", act.user.nunique())
print(act[cols].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string(), "\n")

# ---------------- Step 4: classification ----------------
cand = s[s.req_per_day >= 1000].copy()
hourly = req[cand.index].groupby((req.index % MIN_PER_DAY) // 60).mean()     # mean hour-of-day profile
cand["diurnal_ratio"] = (hourly.max() / hourly.min().clip(lower=1e-9)).clip(upper=1000)


def label(r):
    if r.daily_autocorr >= 0.6 and r.diurnal_ratio >= 2:  return "diurnal"   # checked first
    if r.active_share < 0.5 or r.peak_to_mean > 50:        return "bursty"
    return "steady"


cand["pattern"] = cand.apply(label, axis=1)
cand.to_csv(f"{OUT}/candidates.csv")
print("Functions with >= 1,000 requests/day:", len(cand), "| users:", cand.user.nunique())
print(cand.pattern.value_counts().to_string(), "\n")

show = ["user", "mem_limit_mb", "req_per_day", "peak_to_mean", "active_share", "diurnal_ratio", "daily_autocorr", "avg_dur_ms"]
for p in ["steady", "diurnal", "bursty"]:
    print(f"=== {p} ===")
    print(cand[cand.pattern == p].sort_values("req_per_day", ascending=False)[show].head(8).round(2).to_string(), "\n")

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
for ax, p in zip(axes, ["steady", "diurnal", "bursty"]):
    pick = cand[cand.pattern == p].sort_values("req_per_day", ascending=False).drop_duplicates("user").head(3).index
    for f in pick:
        h = req[f].groupby(req.index // 60).sum()
        ax.plot(h.values / h.values.max(), label=f, lw=0.8)
    ax.set_title(p); ax.legend(fontsize=7); ax.set_ylabel("normalised requests")
axes[-1].set_xlabel("hour (31 days)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig0_pattern_examples.png", dpi=200); plt.show()

# ---------------- Step 5: representative functions ----------------
# Chosen manually from the lists above: different users, full 31 days of data.
# Excluded: 577---631 (exactly 8,640 req/day = one request every 10 s, likely a health check)
#           1111---818 (no traffic during the first ~5.6 days)
REPS = {"steady": "1249---849---custom-300-128",
        "diurnal": "1411---741---pool24-600-512",
        "bursty": "1749---4---pool24-600-512"}
rows = []
for p, f in REPS.items():
    rps = req[f] / 60
    d = s.loc[f, "avg_dur_ms"] / 1000
    rows.append({"pattern": p, "funcID": f,
                 "mean_rps": rps.mean(), "p99_rps": rps.quantile(.99), "max_rps": rps.max(),
                 "mean_concurrency": rps.mean() * d,       # Little's law: arrival rate x time in system
                 "max_concurrency": rps.max() * d,
                 "avg_dur_ms": d * 1000, "mem_limit_mb": s.loc[f, "mem_limit_mb"],
                 "cpu_limit_cores": s.loc[f, "cpu_limit_cores"]})
reps = pd.DataFrame(rows).set_index("pattern")
reps.to_csv(f"{OUT}/representatives.csv")
print(reps.round(3).to_string())
