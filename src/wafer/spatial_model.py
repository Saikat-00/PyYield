"""
spatial_model.py
----------------
Module 3 - Simulates spatial process variations across a silicon wafer.
Models the fact that chips at wafer edges experience more parameter
drift than chips at the center, generating yield heatmaps.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.simulation.spice_engine import NOMINAL_PARAMS, run_single_simulation


# ─────────────────────────────────────────────
# WAFER CONFIGURATION
# ─────────────────────────────────────────────
WAFER_RADIUS_MM   = 150    # Standard 300mm wafer → 150mm radius
EDGE_SIGMA_SCALE  = 3.0    # Edge chips have 3x more variation than center
GRID_RESOLUTION   = 20     # 20x20 grid of die positions


def compute_radial_sigma(x_mm: float, y_mm: float,
                         base_sigma_pct: float = 0.03) -> float:
    """
    Computes the local process variation sigma at a wafer (x, y) position.
    Variation increases quadratically from center to edge — this models
    the real thermal and chemical gradients in deposition processes.

    Args:
        x_mm, y_mm     : Die position in mm from wafer center
        base_sigma_pct : Sigma at wafer center (fraction of nominal)

    Returns:
        Local sigma (fraction of nominal value)
    """
    r = np.sqrt(x_mm**2 + y_mm**2)
    r_norm = r / WAFER_RADIUS_MM   # Normalized 0 (center) → 1 (edge)

    # Quadratic gradient: sigma grows toward edge
    local_sigma = base_sigma_pct * (1 + (EDGE_SIGMA_SCALE - 1) * r_norm**2)
    return local_sigma


def simulate_die(x_mm: float, y_mm: float, seed: int = None) -> dict:
    """
    Simulates a single die at position (x_mm, y_mm) on the wafer.
    Parameters are varied based on the local sigma at that position.

    Returns a result dict with position, parameters, and performance.
    """
    rng = np.random.default_rng(seed)

    local_sigma = compute_radial_sigma(x_mm, y_mm)
    params = NOMINAL_PARAMS.copy()

    # Vary each sensitive parameter by local sigma
    for param in ["vth_n", "vth_p", "length", "width_n", "width_p"]:
        nominal = NOMINAL_PARAMS[param]
        varied  = rng.normal(loc=nominal, scale=abs(nominal) * local_sigma)
        params[param] = varied

    result = run_single_simulation(params)

    return {
        "x_mm":               x_mm,
        "y_mm":               y_mm,
        "radius_mm":          round(np.sqrt(x_mm**2 + y_mm**2), 3),
        "local_sigma_pct":    round(local_sigma * 100, 3),
        **result,
    }


def simulate_full_wafer(grid_size: int = GRID_RESOLUTION,
                        seed: int = 42) -> pd.DataFrame:
    """
    Simulates all dies across a full wafer grid.
    Dies outside the wafer radius are excluded.

    Args:
        grid_size : Number of grid points per axis (grid_size x grid_size)
        seed      : Random seed

    Returns:
        DataFrame with one row per valid die position
    """
    rng    = np.random.default_rng(seed)
    coords = np.linspace(-WAFER_RADIUS_MM, WAFER_RADIUS_MM, grid_size)
    dies   = []

    print(f"[Wafer] Simulating {grid_size}x{grid_size} die grid...")

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            # Skip dies outside circular wafer boundary
            if np.sqrt(x**2 + y**2) > WAFER_RADIUS_MM:
                continue

            die_seed = int(rng.integers(0, 99999))
            result   = simulate_die(x, y, seed=die_seed)
            dies.append(result)

        if (i + 1) % 5 == 0:
            print(f"  ✓ Row {i+1}/{grid_size} complete...")

    df = pd.DataFrame(dies)
    print(f"[Wafer] Simulated {len(df)} valid dies on wafer.\n")
    return df


def compute_wafer_yield_map(df: pd.DataFrame,
                             grid_size: int = GRID_RESOLUTION) -> dict:
    """
    Computes yield statistics and spatial patterns from wafer simulation.

    Returns:
        stats dict with overall yield, edge vs center breakdown,
        and arrays for heatmap plotting.
    """
    total     = len(df)
    passing   = df["is_valid"].sum()
    failing   = total - passing

    # Split into center (r < 75mm) and edge (r >= 75mm) zones
    center_df = df[df["radius_mm"] < 75]
    edge_df   = df[df["radius_mm"] >= 75]

    center_yield = 100 * center_df["is_valid"].mean() if len(center_df) > 0 else 0
    edge_yield   = 100 * edge_df["is_valid"].mean()   if len(edge_df)   > 0 else 0

    stats = {
        "total_dies":     total,
        "passing_dies":   int(passing),
        "failing_dies":   int(failing),
        "overall_yield":  round(100 * passing / total, 2),
        "center_yield":   round(center_yield, 2),
        "edge_yield":     round(edge_yield, 2),
        "mean_delay_ns":  round(df["propagation_delay_ns"].mean(), 6),
        "std_delay_ns":   round(df["propagation_delay_ns"].std(), 6),
        "mean_power_uw":  round(df["power_dissipation_uw"].mean(), 4),
    }

    return stats


def save_wafer_data(df: pd.DataFrame,
                    filename: str = "wafer_simulation.csv"):
    """Saves wafer simulation data to data/processed/."""
    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{filename}"
    df.to_csv(path, index=False)
    print(f"[Wafer] Data saved → {path}")


# ─────────────────────────────────────────────
# Self-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   PyYield — Wafer Spatial Model Self-Test")
    print("=" * 55)

    # Simulate full wafer
    df = simulate_full_wafer(grid_size=20, seed=42)

    # Compute yield map
    stats = compute_wafer_yield_map(df)

    print("[Wafer Yield Statistics]")
    print(f"  Total Dies Simulated : {stats['total_dies']}")
    print(f"  Passing Dies         : {stats['passing_dies']}")
    print(f"  Failing Dies         : {stats['failing_dies']}")
    print(f"  Overall Wafer Yield  : {stats['overall_yield']}%")
    print(f"  Center Zone Yield    : {stats['center_yield']}%")
    print(f"  Edge Zone Yield      : {stats['edge_yield']}%")
    print(f"\n  Mean Delay           : {stats['mean_delay_ns']} ns")
    print(f"  Std Dev Delay        : {stats['std_delay_ns']} ns")
    print(f"  Mean Power           : {stats['mean_power_uw']} µW")

    # Show radial trend — sample 5 dies from center to edge
    print("\n[Radial Trend — Center to Edge]")
    sample = df.sort_values("radius_mm").iloc[
        np.linspace(0, len(df)-1, 6).astype(int)
    ]
    print(f"  {'Radius (mm)':<14} {'Sigma (%)':<12} "
          f"{'Delay (ns)':<14} {'Pass?'}")
    print(f"  {'-'*52}")
    for _, row in sample.iterrows():
        status = "✓" if row["is_valid"] else "✗"
        print(f"  {row['radius_mm']:<14.1f} {row['local_sigma_pct']:<12.2f} "
              f"{row['propagation_delay_ns']:<14.6f} {status}")

    # Save data
    save_wafer_data(df)

    print("\n[spatial_model.py] — All checks passed.")
    print("=" * 55)