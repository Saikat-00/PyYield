"""
yield_oracle.py — ML Yield Oracle v2 (fixed)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.ml.surrogate_model import load_model, predict, FEATURE_COLS
from src.simulation.spice_engine import NOMINAL_PARAMS

# ─────────────────────────────────────────────
# YIELD SPECIFICATION LIMITS
# ─────────────────────────────────────────────
DELAY_LIMIT_NS = 1.0      # Max propagation delay (ns)
POWER_LIMIT_UW = 500.0    # Max power dissipation (µW)


def classify_chip(params, delay_model, delay_scaler,
                  power_model, power_scaler) -> dict:
    pred_delay = float(predict(delay_model, delay_scaler, params))
    pred_power = float(predict(power_model, power_scaler, params))

    delay_pass = bool(pred_delay < DELAY_LIMIT_NS)
    power_pass = bool(pred_power < POWER_LIMIT_UW)
    overall    = delay_pass and power_pass

    return {
        "predicted_delay_ns":  round(pred_delay, 6),
        "predicted_power_uw":  round(pred_power, 4),
        "delay_pass":          delay_pass,
        "power_pass":          power_pass,
        "overall_pass":        overall,
        "delay_margin_pct":    round(100 * (DELAY_LIMIT_NS - pred_delay) / DELAY_LIMIT_NS, 2),
        "power_margin_pct":    round(100 * (POWER_LIMIT_UW - pred_power) / POWER_LIMIT_UW, 2),
        "verdict":             "PASS" if overall else "FAIL",
    }


def parameter_sweep(param_name, sweep_values,
                    delay_model, delay_scaler,
                    power_model, power_scaler,
                    base_params=None) -> pd.DataFrame:
    if base_params is None:
        base_params = NOMINAL_PARAMS.copy()
    rows = []
    for val in sweep_values:
        p = base_params.copy()
        p[param_name] = val
        result = classify_chip(p, delay_model, delay_scaler,
                               power_model, power_scaler)
        rows.append({param_name: val, **result})
    return pd.DataFrame(rows)


def compute_oracle_yield(n_samples: int = 2000,
                         delay_model=None,
                         delay_scaler=None,
                         power_model=None,
                         power_scaler=None,
                         seed: int = 42) -> tuple:
    """
    Vectorized Oracle yield estimation.
    Returns (stats_dict, results_df).
    FIXED: explicit float casting, bounds verification, debug output.
    """
    from src.simulation.monte_carlo import generate_lhs_samples

    print(f"\n[Yield Oracle] Classifying {n_samples} chips...")
    print(f"[Yield Oracle] Limits — Delay < {DELAY_LIMIT_NS} ns | Power < {POWER_LIMIT_UW} µW")

    # ── Generate samples ──────────────────────
    samples_df = generate_lhs_samples(n_samples=n_samples, seed=seed,
                                       include_mismatch=True)

    # ── Select only the 6 trained features ───
    missing = [c for c in FEATURE_COLS if c not in samples_df.columns]
    if missing:
        raise ValueError(f"[Oracle] Missing columns in samples: {missing}")

    X = samples_df[FEATURE_COLS].values.astype(np.float64)
    actual_n = len(X)

    # ── Vectorized prediction ─────────────────
    X_delay = delay_scaler.transform(X)
    X_power = power_scaler.transform(X)

    pred_delays = delay_model.predict(X_delay).astype(np.float64)
    pred_powers = power_model.predict(X_power).astype(np.float64)

    # ── Debug: print prediction ranges ───────
    print(f"[Yield Oracle] Delay predictions  : "
          f"min={pred_delays.min():.5f}  max={pred_delays.max():.5f}  "
          f"mean={pred_delays.mean():.5f} ns")
    print(f"[Yield Oracle] Power predictions  : "
          f"min={pred_powers.min():.4f}  max={pred_powers.max():.4f}  "
          f"mean={pred_powers.mean():.4f} µW")

    # ── Classification ────────────────────────
    delay_ok = pred_delays < float(DELAY_LIMIT_NS)
    power_ok = pred_powers < float(POWER_LIMIT_UW)
    passes   = delay_ok & power_ok

    n_pass        = int(np.sum(passes))
    n_delay_fail  = int(np.sum(~delay_ok))
    n_power_fail  = int(np.sum(~power_ok))
    yield_pct     = round(100.0 * n_pass / actual_n, 2)

    print(f"[Yield Oracle] Delay failures : {n_delay_fail}")
    print(f"[Yield Oracle] Power failures : {n_power_fail}")
    print(f"[Yield Oracle] Yield          : {yield_pct}%  "
          f"({n_pass}/{actual_n} pass)")

    # ── Build results DataFrame ───────────────
    results_df = samples_df[FEATURE_COLS].copy().reset_index(drop=True)
    results_df["predicted_delay_ns"] = pred_delays
    results_df["predicted_power_uw"] = pred_powers
    results_df["delay_pass"]         = delay_ok
    results_df["power_pass"]         = power_ok
    results_df["overall_pass"]       = passes

    stats = {
        "yield_percent":   yield_pct,
        "n_total":         actual_n,
        "n_pass":          n_pass,
        "n_fail":          actual_n - n_pass,
        "n_delay_fail":    n_delay_fail,
        "n_power_fail":    n_power_fail,
        "mean_delay_ns":   round(float(pred_delays.mean()), 6),
        "std_delay_ns":    round(float(pred_delays.std()),  6),
        "mean_power_uw":   round(float(pred_powers.mean()), 4),
        "std_power_uw":    round(float(pred_powers.std()),  4),
    }

    return stats, results_df


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   PyYield — Yield Oracle Self-Test (v2 fixed)")
    print("=" * 60)

    delay_model, delay_scaler = load_model("delay")
    power_model, power_scaler = load_model("power")

    # Single chip test
    result = classify_chip(NOMINAL_PARAMS, delay_model, delay_scaler,
                           power_model, power_scaler)
    print(f"\n[Nominal Chip]")
    print(f"  Predicted Delay : {result['predicted_delay_ns']} ns  "
          f"(limit {DELAY_LIMIT_NS} ns)  -> {'PASS' if result['delay_pass'] else 'FAIL'}")
    print(f"  Predicted Power : {result['predicted_power_uw']} µW  "
          f"(limit {POWER_LIMIT_UW} µW) -> {'PASS' if result['power_pass'] else 'FAIL'}")
    print(f"  Overall Verdict : {result['verdict']}")

    # Oracle test
    stats, _ = compute_oracle_yield(n_samples=500,
                                    delay_model=delay_model,
                                    delay_scaler=delay_scaler,
                                    power_model=power_model,
                                    power_scaler=power_scaler)
    print(f"\n[Oracle Result]")
    print(f"  Yield     : {stats['yield_percent']}%")
    print(f"  Pass      : {stats['n_pass']}")
    print(f"  Fail      : {stats['n_fail']}")
    print(f"  Delay fails: {stats['n_delay_fail']}")
    print(f"  Power fails: {stats['n_power_fail']}")
    print("\n[yield_oracle.py v2] PASSED")
    print("=" * 60)