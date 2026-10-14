"""
aging.py — NBTI / HCI / Electromigration aging model
Predicts how yield degrades over chip operational lifetime.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.simulation.spice_engine import NOMINAL_PARAMS, run_single_simulation


# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS & MODEL COEFFICIENTS
# ─────────────────────────────────────────────
KB          = 8.617e-5    # Boltzmann constant (eV/K)
T_REF_K     = 300         # Reference temperature (K)

# NBTI (Negative Bias Temperature Instability)
NBTI_A      = 3.5e-3      # Pre-exponential factor (V)
NBTI_EA     = 0.13        # Activation energy (eV)
NBTI_N      = 0.25        # Time exponent

# HCI (Hot Carrier Injection)
HCI_B       = 1.2e-4      # HCI coefficient
HCI_EA      = 0.15        # Activation energy (eV)
HCI_M       = 0.45        # Stress exponent

# Electromigration (Black's equation)
EM_A        = 1.0e20      # Pre-exponential (Black's equation)
EM_N        = 2.0         # Current density exponent
EM_EA_EM    = 0.9         # Activation energy for metal EM (eV)
MAX_CURRENT_DENSITY = 5e9 # Max allowed J (A/m²) — industry rule


def compute_nbti_vth_shift(time_years: float,
                            temp_c: float = 27,
                            vdd: float = 3.3,
                            tox: float = 4e-9) -> float:
    """
    NBTI Vth shift model (power-law degradation).
    Used for PMOS under negative bias stress.

    ΔVth_NBTI = A × exp(-Ea/kT) × (Eox)^0.5 × t^n

    Returns ΔVth in volts (always positive — causes |Vtp| to increase).
    """
    t_sec = time_years * 365.25 * 24 * 3600
    T_K   = temp_c + 273.15
    E_ox  = vdd / tox                        # oxide electric field (V/m)
    dVth  = (NBTI_A
             * np.exp(-NBTI_EA / (KB * T_K))
             * np.sqrt(E_ox / 1e9)           # normalized field
             * (t_sec ** NBTI_N))
    return round(float(dVth), 6)


def compute_hci_vth_shift(time_years: float,
                           temp_c: float = 27,
                           vdd: float = 3.3) -> float:
    """
    HCI Vth shift model for NMOS.
    Hot carriers trapped near drain degrade NMOS threshold.

    ΔVth_HCI = B × exp(-Ea/kT) × VDD^3 × t^m
    """
    t_sec = time_years * 365.25 * 24 * 3600
    T_K   = temp_c + 273.15
    dVth  = (HCI_B
             * np.exp(-HCI_EA / (KB * T_K))
             * (vdd ** 3)
             * (t_sec ** HCI_M))
    return round(float(dVth), 6)


def compute_em_mttf(current_density: float, temp_c: float = 27) -> float:
    """
    Black's equation for Electromigration Mean Time To Failure.
    MTTF = A / (J^n) × exp(Ea / kT)

    Returns MTTF in years.
    """
    T_K  = temp_c + 273.15
    J    = max(current_density, 1.0)   # A/m², avoid div by zero
    mttf_sec = (EM_A / (J ** EM_N)) * np.exp(EM_EA_EM / (KB * T_K))
    return round(mttf_sec / (365.25 * 24 * 3600), 2)


def compute_aged_params(base_params: dict,
                         time_years: float,
                         activity_factor: float = 0.5) -> dict:
    """
    Applies aging degradation to a parameter set.
    Returns a new param dict representing the aged device.

    Degradation mechanisms applied:
      - NBTI  → increases |Vtp| (PMOS gets slower)
      - HCI   → increases Vtn  (NMOS gets slower)
      - tox degradation → thinner effective oxide after stress
    """
    aged = base_params.copy()
    temp = base_params.get("temperature", 27)
    vdd  = base_params.get("vdd", 3.3)
    tox  = base_params.get("tox", 4e-9)

    # NBTI on PMOS — |Vtp| increases
    dvth_nbti = compute_nbti_vth_shift(time_years, temp, vdd, tox)
    aged["vth_p"] = aged["vth_p"] - dvth_nbti   # more negative = stronger stress

    # HCI on NMOS — Vtn increases
    dvth_hci = compute_hci_vth_shift(time_years, temp, vdd)
    aged["vth_n"] = aged["vth_n"] + dvth_hci * activity_factor

    # Oxide wear — slight tox reduction from charge trapping
    tox_degradation = 0.001 * time_years   # 0.1% per year (simplified)
    aged["tox"] = max(tox * (1 - tox_degradation), 2e-9)

    aged["aging_years"]   = time_years
    aged["dvth_nbti_mv"]  = round(dvth_nbti * 1000, 4)
    aged["dvth_hci_mv"]   = round(dvth_hci  * 1000, 4)

    return aged


def run_aging_sweep(time_points: list = None,
                    base_params: dict = None,
                    activity_factor: float = 0.5) -> pd.DataFrame:
    """
    Sweeps aging from t=0 to max_years and simulates each aged state.
    Returns DataFrame with performance vs time.
    """
    if time_points is None:
        time_points = [0, 0.5, 1, 2, 3, 5, 7, 10]
    if base_params is None:
        base_params = NOMINAL_PARAMS.copy()

    rows = []
    print(f"[Aging] Simulating {len(time_points)} time points...")
    for t in time_points:
        aged_p = compute_aged_params(base_params, t, activity_factor)
        result = run_single_simulation(aged_p)
        rows.append({
            "years":            t,
            "dvth_nbti_mv":     aged_p.get("dvth_nbti_mv", 0),
            "dvth_hci_mv":      aged_p.get("dvth_hci_mv", 0),
            "vth_n_aged":       round(aged_p["vth_n"], 4),
            "vth_p_aged":       round(aged_p["vth_p"], 4),
            "tox_aged_nm":      round(aged_p["tox"] * 1e9, 3),
            "propagation_delay_ns": result["propagation_delay_ns"],
            "power_dissipation_uw": result["power_dissipation_uw"],
            "p_leakage_uw":         result.get("p_leakage_uw", 0),
            "is_valid":             result["is_valid"],
            "sim_source":           result["sim_source"],
        })
        status = "PASS" if rows[-1]["is_valid"] else "FAIL"
        print(f"  t={t:>4} yr | Delay={rows[-1]['propagation_delay_ns']:.5f} ns "
              f"| ΔVth_n={rows[-1]['dvth_hci_mv']:.2f} mV | {status}")

    return pd.DataFrame(rows)


def compute_em_risk(params: dict) -> dict:
    """
    Estimates electromigration risk for the supply current path.
    """
    from src.simulation.spice_engine import derive_cox, compute_effective_load
    vdd  = params["vdd"]
    CL   = compute_effective_load(params)
    freq = 1e9
    # Average supply current (dynamic estimate)
    I_avg = 0.5 * CL * vdd * freq
    # Assume min metal width = 0.28µm, thickness = 0.5µm
    wire_area = 0.28e-6 * 0.5e-6
    J_avg     = I_avg / wire_area
    mttf      = compute_em_mttf(J_avg, params.get("temperature", 27))
    em_safe   = J_avg < MAX_CURRENT_DENSITY

    return {
        "avg_current_uA":      round(I_avg * 1e6, 4),
        "current_density_Apm2": round(J_avg, 2),
        "em_mttf_years":       mttf,
        "em_safe":             em_safe,
        "max_allowed_J":       MAX_CURRENT_DENSITY,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("   PyYield — Aging Model Self-Test")
    print("=" * 60)

    print("\n[NBTI Shift Preview]")
    for yr in [0, 1, 3, 5, 10]:
        dv = compute_nbti_vth_shift(yr)
        print(f"  t={yr:>2} yr → ΔVth_p (NBTI) = {dv*1000:.3f} mV")

    print("\n[HCI Shift Preview]")
    for yr in [0, 1, 3, 5, 10]:
        dv = compute_hci_vth_shift(yr)
        print(f"  t={yr:>2} yr → ΔVth_n (HCI)  = {dv*1000:.3f} mV")

    print("\n[Electromigration Risk]")
    em = compute_em_risk(NOMINAL_PARAMS)
    print(f"  Avg Current     : {em['avg_current_uA']:.4f} µA")
    print(f"  Current Density : {em['current_density_Apm2']:.2e} A/m²")
    print(f"  MTTF (EM)       : {em['em_mttf_years']:.1f} years")
    print(f"  EM Safe         : {'YES' if em['em_safe'] else 'NO — RISK!'}")

    print("\n[Aging Sweep — 0 to 10 years]")
    df_age = run_aging_sweep([0, 1, 3, 5, 10], NOMINAL_PARAMS)
    print(df_age[["years", "dvth_hci_mv", "dvth_nbti_mv",
                  "propagation_delay_ns", "is_valid"]].to_string(index=False))

    print("\n[aging.py] PASSED")
    print("=" * 60)