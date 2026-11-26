"""
SEN421 - Topic 12: Serverless vs Always-On cost model (Huawei Cloud, TR-Istanbul)

Three options per workload:
  FaaS  : FunctionGraph + dedicated API Gateway (APIG)
  VM    : always-on ECS S7n instances (+ EIP, + ELB when >1 instance)
  CONT  : always-on CCE Autopilot pods (+ cluster mgmt, + ELB, + EIP)

Traffic SHAPE comes from the Huawei 2025 trace; traffic VOLUME is scaled by a factor s.
All costs are USD per month (730 h). The 31-day trace (744 h) is normalised to 730 h.
"""
import numpy as np
import pandas as pd

HOURS = 730.0
TRACE_HOURS = 31 * 24.0          # 744 h
MONTH_FACTOR = HOURS / TRACE_HOURS

# ---------------- Prices (TR-Istanbul, accessed 2026-11-26) ----------------
P = {
    "fg_req_per_m": 0.25,  "fg_req_free": 1e6,          # per 1M requests, free 1M/month
    "fg_gbs": 0.0000206667, "fg_gbs_free": 4e5,         # per GB-s, free 400k GB-s/month
    # APIG dedicated editions: (name, max TPS, USD/hour)
    "apig": [("Basic", 2000, 0.76), ("Professional", 4000, 3.46),
             ("Enterprise", 6000, 5.19), ("Platinum", 10000, 8.65)],
    "bw_mbps_h": 0.0178,                                 # public bandwidth per Mbit/s-hour (APIG and EIP)
    # ECS S7n pay-per-use: (vCPU, GiB, USD/hour)
    "ecs": [(1, 1, 0.0311), (1, 2, 0.0341), (1, 4, 0.0401), (2, 4, 0.0682),
            (2, 8, 0.0802), (4, 8, 0.1364), (8, 16, 0.2728)],
    "disk_gib_h": 0.000143, "disk_gib": 40,              # General Purpose SSD system disk
    "eip_h": 0.005,                                      # EIP reservation
    "elb_1az_h": 0.0695, "elb_2az_h": 0.139,             # application LB Small I
    "cce_cluster_h": 0.1, "cce_vcpu_h": 0.04032, "cce_gib_h": 0.004428,
}

# ---------------- Baseline assumptions (change one at a time for sensitivity) ----------------
BASE = {
    "k": 1,              # number of APIs sharing platform/entry components (gateway, ELB, EIP, cluster)
    "gran_ms": 1,        # FunctionGraph billing granularity (1 ms per billing docs; 100 ms per price page)
    "sizing": "max",     # size always-on capacity for 'max' or 'p99' per-minute load
    "overhead_ms": 1.0,  # fixed per-request HTTP/network overhead on VM/container
    "util": 0.70,        # target CPU utilisation of always-on capacity
    "ha": False,         # True = at least 2 instances/pods in 2 AZs + 2-AZ ELB
    "bw_mbps": 5,        # public bandwidth bought for the entry point (same for all options)
    "pod_vcpu": 1, "pod_gib": 2,   # CCE Autopilot pod size
}


# Huawei resource pools in the trace: memory (MB) -> CPU limit (cores), e.g. pool22-300-128
POOL_CPU = {128: 0.3, 256: 0.4, 512: 0.6, 768: 0.8, 1024: 1.0, 1280: 1.2, 1536: 1.4,
            1792: 1.6, 2048: 1.8, 2560: 2.2, 3072: 2.6, 3584: 3.0, 4096: 3.4}


def workload(req_per_min: pd.Series, dur_ms: float, mem_mb: float, cpu_cores: float = None) -> dict:
    """Summarise one function from its per-minute request series (31 days).
    cpu_cores = CPU limit of the function (from its pool name); CPU work per request is
    assumed to be execution time x CPU limit (an upper bound)."""
    r = req_per_min.fillna(0).values.astype(float)
    if cpu_cores is None:
        cpu_cores = POOL_CPU.get(int(mem_mb), mem_mb / 1024 * 0.9)
    return {"total": r.sum(), "max_rps": r.max() / 60.0,
            "p99_rps": np.quantile(r, 0.99) / 60.0,
            "dur_s": dur_ms / 1000.0, "mem_gb": mem_mb / 1024.0, "cpu": cpu_cores}


def faas_cost(w, s, a):
    N = s * w["total"] * MONTH_FACTOR
    billed = np.ceil(w["dur_s"] * 1000 / a["gran_ms"]) * a["gran_ms"] / 1000.0
    gbs = N * billed * w["mem_gb"]
    c_req = max(N - P["fg_req_free"], 0) / 1e6 * P["fg_req_per_m"]
    c_dur = max(gbs - P["fg_gbs_free"], 0) * P["fg_gbs"]
    peak = s * w["max_rps"]
    if peak <= P["apig"][-1][1]:
        name, _, price = next(e for e in P["apig"] if e[1] >= peak)
        n_gw = 1
    else:                                   # beyond Platinum: several Platinum gateways
        name, cap, price = P["apig"][-1]
        n_gw = int(np.ceil(peak / cap))
    c_gw = (n_gw * price + a["bw_mbps"] * P["bw_mbps_h"]) * HOURS / a["k"]
    return {"FaaS": c_req + c_dur + c_gw, "faas_compute": c_req + c_dur,
            "faas_gateway": c_gw, "gw_edition": f"{n_gw}x{name}"}


def _needed_vcpu(w, s, a):
    peak = s * (w["max_rps"] if a["sizing"] == "max" else w["p99_rps"])
    return peak * (w["dur_s"] * w["cpu"] + a["overhead_ms"] / 1000.0) / a["util"]


def vm_cost(w, s, a):
    need = _needed_vcpu(w, s, a)
    mem_need = max(1.0, need * a["util"] * w["mem_gb"])      # GiB, rough
    largest = P["ecs"][-1]
    fit = [e for e in P["ecs"] if e[0] >= need and e[1] >= mem_need]
    if fit:
        n, flav = 1, min(fit, key=lambda e: e[2])
    else:
        flav = largest
        n = int(np.ceil(max(need / largest[0], mem_need / largest[1])))
    if a["ha"]:
        n = max(n, 2)
    c_comp = n * (flav[2] + P["disk_gib"] * P["disk_gib_h"]) * HOURS
    elb = (P["elb_2az_h"] if a["ha"] else P["elb_1az_h"]) if n > 1 else 0.0
    c_entry = (P["eip_h"] + a["bw_mbps"] * P["bw_mbps_h"] + elb) * HOURS / a["k"]
    return {"VM": c_comp + c_entry, "vm_n": n, "vm_flavor": f"{flav[0]}vCPU/{flav[1]}GiB"}


def cont_cost(w, s, a):
    need = _needed_vcpu(w, s, a)
    n = max(1, int(np.ceil(need / a["pod_vcpu"])))
    if a["ha"]:
        n = max(n, 2)
    c_pods = n * (a["pod_vcpu"] * P["cce_vcpu_h"] + a["pod_gib"] * P["cce_gib_h"]) * HOURS
    elb = P["elb_2az_h"] if a["ha"] else P["elb_1az_h"]
    c_shared = (P["cce_cluster_h"] + elb + P["eip_h"] + a["bw_mbps"] * P["bw_mbps_h"]) * HOURS / a["k"]
    return {"CONT": c_pods + c_shared, "cont_pods": n}


def costs(w, s=1.0, **over):
    a = {**BASE, **over}
    out = {"s": s, "req_month": s * w["total"] * MONTH_FACTOR}
    out.update(faas_cost(w, s, a)); out.update(vm_cost(w, s, a)); out.update(cont_cost(w, s, a))
    return out


def sweep(w, vol_grid=None, **over):
    """Sweep monthly request volume (shape fixed, volume scaled)."""
    if vol_grid is None:
        vol_grid = np.logspace(3, 11, 400)
    base = w["total"] * MONTH_FACTOR
    return pd.DataFrame([costs(w, v / base, **over) for v in vol_grid])


def breakeven(df, a="FaaS", b="VM"):
    """Break-even = lowest monthly request volume where option a stops being cheaper than b.
    Returns (value, note). value is NaN when there is no such point in the swept range."""
    d = (df[a] - df[b]).values
    x = np.log10(df["req_month"].values)
    if (d >= 0).all():
        return np.nan, f"{a} never cheaper"
    if (d < 0).all():
        return np.nan, f"{a} always cheaper"
    for i in range(len(d) - 1):
        if d[i] < 0 <= d[i + 1]:
            t = d[i] / (d[i] - d[i + 1])
            return 10 ** (x[i] + t * (x[i + 1] - x[i])), "ok"
    return np.nan, f"{a} cheaper only at high volume"


def crossings(df, a="FaaS", b="VM"):
    """Monthly request volumes where option a and b have equal cost (log-interpolated)."""
    d = (df[a] - df[b]).values
    x = np.log10(df["req_month"].values)
    out = []
    for i in range(len(d) - 1):
        if d[i] == 0:
            out.append(10 ** x[i])
        elif d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            out.append(10 ** (x[i] + t * (x[i + 1] - x[i])))
    return out
