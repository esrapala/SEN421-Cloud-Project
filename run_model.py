"""
SEN421 Topic 12 - run the cost model on the three representative functions.
Usage in Colab (after uploading cost_model.py and run_model.py to /content):
    %run run_model.py
Outputs (tables + figures) are saved to OUT.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from cost_model import workload, costs, sweep, breakeven

OUT = os.environ.get("SEN421_OUT", "/content/drive/MyDrive/SEN421/data")
RES = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)

req = pd.read_parquet(f"{OUT}/requests.parquet")
summ = pd.read_csv(f"{OUT}/function_summary.csv", index_col=0)
REPS = {"steady": "1249---849---custom-300-128",
        "diurnal": "1411---741---pool24-600-512",
        "bursty": "1749---4---pool24-600-512"}
cpu_of = lambda f: int(f.split("-")[-2]) / 1000          # e.g. ...pool24-600-512 -> 0.6 cores
W = {k: workload(req[f], summ.loc[f, "avg_dur_ms"], summ.loc[f, "mem_limit_mb"], cpu_of(f)) for k, f in REPS.items()}

# ---------- Table 1: cost at the observed volume (s = 1), baseline ----------
t1 = pd.DataFrame({k: costs(w) for k, w in W.items()}).T
cols = ["req_month", "FaaS", "faas_compute", "faas_gateway", "gw_edition", "VM", "vm_n", "vm_flavor", "CONT", "cont_pods"]
t1 = t1[cols]
t1.to_csv(f"{RES}/table1_observed_volume.csv")
print("TABLE 1 - monthly cost (USD) at observed volume, baseline\n", t1.to_string(), "\n")

# ---------- Figure 1: cost vs monthly requests ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (k, w) in zip(axes, W.items()):
    base = sweep(w); ideal = sweep(w, k=10**9)
    ax.loglog(base.req_month, base.FaaS, label="FaaS (gateway on 1 API)", c="tab:red")
    ax.loglog(ideal.req_month, ideal.FaaS, "--", label="FaaS (gateway fully shared)", c="tab:red", alpha=.6)
    ax.loglog(base.req_month, base.VM, label="VM (ECS)", c="tab:blue")
    ax.loglog(base.req_month, base.CONT, label="Container (CCE Autopilot)", c="tab:green")
    ax.axvline(w["total"] * 730 / 744, c="grey", ls=":", lw=1, label="observed volume")
    pm = w["max_rps"] / (w["total"] / (len(req) * 60))
    ax.set_title(f"{k} (peak/mean = {pm:.0f})"); ax.set_xlabel("requests / month"); ax.grid(True, which="both", alpha=.3)
axes[0].set_ylabel("USD / month"); axes[0].set_ylim(bottom=1); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RES}/fig1_cost_curves.png", dpi=200); plt.show()

# ---------- Table 2: break-even, one-at-a-time sensitivity ----------
SCEN = {"baseline": {}, "k=5": {"k": 5}, "k=20": {"k": 20}, "k=100": {"k": 100},
        "gateway fully shared": {"k": 10**9}, "billing 100 ms": {"gran_ms": 100},
        "sizing p99": {"sizing": "p99"}, "overhead 0.5 ms": {"overhead_ms": .5},
        "overhead 5 ms": {"overhead_ms": 5}, "HA (2 AZ)": {"ha": True},
        "shared gateway + 100 ms": {"k": 10**9, "gran_ms": 100},
        "k=20 + sizing p99": {"k": 20, "sizing": "p99"}}
rows = []
for sname, over in SCEN.items():
    for k, w in W.items():
        df = sweep(w, **over)
        for b in ("VM", "CONT"):
            v, note = breakeven(df, "FaaS", b)
            rows.append({"scenario": sname, "workload": k, "vs": b, "breakeven_req_month": v, "note": note})
t2 = pd.DataFrame(rows)
t2.to_csv(f"{RES}/table2_breakeven_scenarios.csv", index=False)
show = t2.assign(be=lambda d: d.apply(lambda r: f"{r.breakeven_req_month:.3g}" if r.note == "ok" else r.note, axis=1))
print("TABLE 2 - break-even (requests/month) where FaaS stops being cheaper\n",
      show.pivot_table(index=["scenario"], columns=["workload", "vs"], values="be", aggfunc="first").to_string(), "\n")

# ---------- Table 3 / Figure 2: memory x duration sensitivity (Topic 12 requirement) ----------
MEM = [128, 256, 512, 1024, 2048]
DUR = [1, 10, 50, 100, 200, 500, 1000]
shape = pd.Series(np.ones(len(req)))          # constant-rate synthetic baseline (shape only; volume is swept)
ABBR = {"FaaS never cheaper": "never", "FaaS always cheaper": "always", "FaaS cheaper only at high volume": "high-only"}
for kval, label in [(1, "k1"), (20, "k20"), (10**9, "kinf")]:
    grid = pd.DataFrame(index=MEM, columns=DUR, dtype=float)
    notes = pd.DataFrame(index=MEM, columns=DUR, dtype=object)
    for m in MEM:
        for d in DUR:
            v, note = breakeven(sweep(workload(shape, d, m), k=kval), "FaaS", "VM")
            grid.loc[m, d] = v
            notes.loc[m, d] = f"{v:.2g}" if note == "ok" else ABBR.get(note, note)
    grid.index.name = notes.index.name = "memory_MB"; grid.columns.name = notes.columns.name = "duration_ms"
    grid.to_csv(f"{RES}/table3_mem_dur_breakeven_{label}.csv")
    print(f"TABLE 3 ({label}) - FaaS vs VM break-even (requests/month), constant-rate baseline\n", notes.to_string(), "\n")
    if label == "kinf":
        plt.figure(figsize=(7, 4))
        plt.imshow(np.log10(grid.values.astype(float)), aspect="auto", cmap="viridis")
        plt.xticks(range(len(DUR)), DUR); plt.yticks(range(len(MEM)), MEM)
        plt.xlabel("execution time (ms)"); plt.ylabel("memory (MB)")
        plt.colorbar(label="log10 break-even requests/month")
        for i in range(len(MEM)):
            for j in range(len(DUR)):
                txt = notes.iloc[i, j]
                plt.text(j, i, txt, ha="center", va="center", fontsize=8,
                         color="black" if (txt in ("always", "never", "high-only")
                                           or np.log10(grid.iloc[i, j]) > 8.3) else "white")
        plt.title("FaaS vs VM break-even, constant load, gateway fully shared\n(always = FaaS cheaper at every volume)", fontsize=10)
        plt.tight_layout(); plt.savefig(f"{RES}/fig2_mem_dur_heatmap.png", dpi=200); plt.show()

print("Saved to", RES)
