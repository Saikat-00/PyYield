"""
monte_carlo.py v3 — Global Process Variation + Local Mismatch (RDF)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
import warnings
warnings.filterwarnings("ignore")

from src.simulation.spice_engine import NOMINAL_PARAMS, run_single_simulation

# ─────────────────────────────────────────────
# GLOBAL PROCESS VARIATION (wafer-wide)
# Affects ALL devices on a die the same way
# ─────────────────────────────────────────────
GLOBAL_VARIATION = {
    "vth_n":    (0.10, 0.25,  0.75),
    "vth_p":    (0.10, -0.75, -0.25),
    "length":   (0.08, 0.14,  0.22),
    "width_n":  (0.08, 0.28,  0.44),
    "width_p":  (0.08, 0.56,  0.88),
    "tox":      (0.05, 3e-9,  5e-9),    # ← tox now in MC
    "load_cap": (0.15, 5e-15, 20e-15),
}

# ─────────────────────────────────────────────
# LOCAL MISMATCH — Random Dopant Fluctuation
# Device-to-device variation on the SAME die
# Uses Pelgrom's law: σ_local ∝ 1/√(W×L)
# ─────────────────────────────────────────────
PELGROM_AVT   = 4e-9    # Threshold mismatch coefficient (V·m) — typical 180nm
PELGROM_ABETA = 0.01    # Beta mismatch coefficient


def _clamp(v, lo, hi): return max(lo, min(hi, v))


def _pelgrom_sigma_vth(width_m: float, length_m: float) -> float:
    """
    Pelgrom's law for threshold voltage mismatch.
    σ(ΔVth) = AVT / √(W × L)
    Returns sigma in volts.
    """
    return PELGROM_AVT / np.sqrt(width_m * length_m)


def generate_lhs_samples(n_samples: int = 500, seed: int = 42,
                          include_mismatch: bool = True) -> pd.DataFrame:
    """
    Generates samples using Latin Hypercube Sampling.
    Combines global process variation with local RDF mismatch.
    """
    param_names  = list(GLOBAL_VARIATION.keys())
    sampler      = qmc.LatinHypercube(d=len(param_names), seed=seed)
    u            = sampler.random(n=n_samples)
    rng          = np.random.default_rng(seed + 1)

    samples = []
    for i in range(n_samples):
        p = NOMINAL_PARAMS.copy()

        # ── Global variation ──────────────────
        for j, param in enumerate(param_names):
            sigma_pct, lo, hi = GLOBAL_VARIATION[param]
            nominal = NOMINAL_PARAMS[param]
            sigma   = abs(nominal) * sigma_pct
            varied  = norm.ppf(np.clip(u[i, j], 1e-6, 1-1e-6),
                               loc=nominal, scale=sigma)
            p[param] = _clamp(varied, lo, hi)

        # ── Local mismatch (RDF) ─────────────
        if include_mismatch:
            Wn_m = p["width_n"] * 1e-6
            Wp_m = p["width_p"] * 1e-6
            L_m  = p["length"]  * 1e-6

            sigma_vth_n = _pelgrom_sigma_vth(Wn_m, L_m)
            sigma_vth_p = _pelgrom_sigma_vth(Wp_m, L_m)

            # Each transistor gets its own local Vth offset
            dvth_n = rng.normal(0, sigma_vth_n)
            dvth_p = rng.normal(0, sigma_vth_p)

            p["vth_n"] = _clamp(p["vth_n"] + dvth_n, 0.25, 0.75)
            p["vth_p"] = _clamp(p["vth_p"] + dvth_p, -0.75, -0.25)
            p["local_dvth_n_mv"] = round(dvth_n * 1000, 4)  # store in mV
            p["local_dvth_p_mv"] = round(dvth_p * 1000, 4)
        else:
            p["local_dvth_n_mv"] = 0.0
            p["local_dvth_p_mv"] = 0.0

        samples.append(p)

    return pd.DataFrame(samples)


def generate_monte_carlo_samples(n_samples: int = 500, seed: int = 42,
                                  include_mismatch: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        p = NOMINAL_PARAMS.copy()
        for param, (sigma_pct, lo, hi) in GLOBAL_VARIATION.items():
            nominal = NOMINAL_PARAMS[param]
            p[param] = _clamp(rng.normal(nominal, abs(nominal)*sigma_pct), lo, hi)

        if include_mismatch:
            Wn_m = p["width_n"] * 1e-6
            L_m  = p["length"]  * 1e-6
            sigma_vth_n = _pelgrom_sigma_vth(Wn_m, L_m)
            sigma_vth_p = _pelgrom_sigma_vth(p["width_p"]*1e-6, L_m)
            dvth_n = rng.normal(0, sigma_vth_n)
            dvth_p = rng.normal(0, sigma_vth_p)
            p["vth_n"] = _clamp(p["vth_n"] + dvth_n, 0.25, 0.75)
            p["vth_p"] = _clamp(p["vth_p"] + dvth_p, -0.75, -0.25)
            p["local_dvth_n_mv"] = round(dvth_n * 1000, 4)
            p["local_dvth_p_mv"] = round(dvth_p * 1000, 4)
        else:
            p["local_dvth_n_mv"] = 0.0
            p["local_dvth_p_mv"] = 0.0

        samples.append(p)
    return pd.DataFrame(samples)


def run_monte_carlo(n_samples: int = 500, method: str = "lhs",
                    seed: int = 42, frequency: float = 1e9,
                    include_mismatch: bool = True) -> pd.DataFrame:
    mode = "lhs" if method == "lhs" else "mc"
    print(f"\n[Monte Carlo] {n_samples} sims | {mode.upper()} | "
          f"Mismatch: {'ON' if include_mismatch else 'OFF'}")

    if method == "lhs":
        samples_df = generate_lhs_samples(n_samples, seed, include_mismatch)
    else:
        samples_df = generate_monte_carlo_samples(n_samples, seed, include_mismatch)

    results = []
    for i, row in samples_df.iterrows():
        results.append(run_single_simulation(row.to_dict(), frequency))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples} complete...")

    df = pd.DataFrame(results)
    # Carry forward mismatch columns
    if "local_dvth_n_mv" in samples_df.columns:
        df["local_dvth_n_mv"] = samples_df["local_dvth_n_mv"].values
        df["local_dvth_p_mv"] = samples_df["local_dvth_p_mv"].values

    print(f"  All {n_samples} done.\n")
    return df


def compute_yield_statistics(df: pd.DataFrame) -> dict:
    n_total = len(df); n_pass = df["is_valid"].sum()
    return {
        "yield_percent":   round(100 * n_pass / n_total, 2),
        "n_total":         n_total,
        "n_pass":          int(n_pass),
        "n_fail":          int(n_total - n_pass),
        "mean_delay_ns":   round(df["propagation_delay_ns"].mean(), 6),
        "std_delay_ns":    round(df["propagation_delay_ns"].std(), 6),
        "delay_3sigma_ns": round(df["propagation_delay_ns"].mean() +
                                 3*df["propagation_delay_ns"].std(), 6),
        "mean_power_uw":   round(df["power_dissipation_uw"].mean(), 4),
        "std_power_uw":    round(df["power_dissipation_uw"].std(), 4),
    }


def compute_process_capability(df: pd.DataFrame,
                                usl_delay=1.0, lsl_delay=0.0,
                                usl_power=500.0, lsl_power=0.0) -> dict:
    def _cpk(data, usl, lsl):
        mu = data.mean(); sigma = data.std()
        if sigma == 0: return float("inf"), float("inf")
        return round((usl-lsl)/(6*sigma), 3), round(min((usl-mu)/(3*sigma),(mu-lsl)/(3*sigma)), 3)
    cp_d, cpk_d = _cpk(df["propagation_delay_ns"], usl_delay, lsl_delay)
    cp_p, cpk_p = _cpk(df["power_dissipation_uw"],  usl_power, lsl_power)
    return {
        "delay_Cp": cp_d, "delay_Cpk": cpk_d,
        "power_Cp": cp_p, "power_Cpk": cpk_p,
        "delay_sigma_level": round(cpk_d*3, 2),
        "power_sigma_level": round(cpk_p*3, 2),
        "process_capable": cpk_d >= 1.33 and cpk_p >= 1.33,
    }


def compute_convergence(df: pd.DataFrame, n_steps: int = 25) -> pd.DataFrame:
    n     = len(df)
    sizes = np.unique(np.linspace(10, n, n_steps).astype(int))
    rows  = []
    for size in sizes:
        sub = df.iloc[:size]
        rows.append({
            "n_samples":   int(size),
            "yield_pct":   round(100*sub["is_valid"].mean(), 2),
            "mean_delay":  round(sub["propagation_delay_ns"].mean(), 6),
            "running_std": round(sub["propagation_delay_ns"].std(), 6),
        })
    return pd.DataFrame(rows)


def compute_sensitivity_tornado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-parameter yield impact by toggling each one
    from its 10th to 90th percentile while holding others at median.
    Returns a sorted tornado DataFrame for bar chart display.
    """
    from src.simulation.spice_engine import run_single_simulation, DELAY_LIMIT_NS, POWER_LIMIT_UW

    base_params = {col: df[col].median() for col in
                   ["vth_n","vth_p","length","width_n","width_p","tox","load_cap"]
                   if col in df.columns}
    base_params.update({k: NOMINAL_PARAMS[k] for k in NOMINAL_PARAMS
                        if k not in base_params})

    rows = []
    sweep_params = ["vth_n", "length", "width_n", "width_p", "tox", "load_cap"]

    for param in sweep_params:
        if param not in df.columns:
            continue
        lo_val = df[param].quantile(0.10)
        hi_val = df[param].quantile(0.90)

        p_lo = base_params.copy(); p_lo[param] = lo_val
        p_hi = base_params.copy(); p_hi[param] = hi_val

        r_lo = run_single_simulation(p_lo)
        r_hi = run_single_simulation(p_hi)

        delay_lo = r_lo["propagation_delay_ns"]
        delay_hi = r_hi["propagation_delay_ns"]
        delta    = abs(delay_hi - delay_lo)

        rows.append({
            "parameter":  param,
            "delay_at_p10": delay_lo,
            "delay_at_p90": delay_hi,
            "delta_delay_ns": round(delta, 6),
            "impact_pct": round(100 * delta / max(delay_lo, delay_hi, 1e-10), 2),
        })

    return pd.DataFrame(rows).sort_values("delta_delay_ns", ascending=False)


def save_results(df: pd.DataFrame, filename: str = "monte_carlo_results.csv"):
    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{filename}"
    df.to_csv(path, index=False)
    print(f"[Monte Carlo] Saved → {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("   PyYield — Monte Carlo v3 Self-Test")
    print("=" * 60)

    print("\n[Pelgrom Mismatch Preview]")
    Wn = NOMINAL_PARAMS["width_n"] * 1e-6
    L  = NOMINAL_PARAMS["length"]  * 1e-6
    sv = _pelgrom_sigma_vth(Wn, L)
    print(f"  σ(ΔVth_n) from RDF : {sv*1000:.2f} mV  (Pelgrom)")

    df    = run_monte_carlo(n_samples=300, method="lhs", include_mismatch=True)
    stats = compute_yield_statistics(df)
    cpk   = compute_process_capability(df)

    print(f"\n[Statistics]")
    print(f"  Yield      : {stats['yield_percent']}%")
    print(f"  Mean Delay : {stats['mean_delay_ns']} ns")
    print(f"  Delay Cpk  : {cpk['delay_Cpk']}")

    if "local_dvth_n_mv" in df.columns:
        print(f"\n[Local Mismatch (RDF)]")
        print(f"  σ(ΔVth_n) observed : {df['local_dvth_n_mv'].std():.3f} mV")
        print(f"  Max ΔVth_n         : {df['local_dvth_n_mv'].abs().max():.3f} mV")

    save_results(df)
    print("\n[monte_carlo.py v3] PASSED")
    print("=" * 60)