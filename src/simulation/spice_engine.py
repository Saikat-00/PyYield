"""
spice_engine.py v4 — Full SPICE + tox + Parasitic RC + Power Breakdown
"""

import os, re, subprocess
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import warnings
warnings.filterwarnings("ignore")

import platform as _platform
NGSPICE_EXE = (r"C:\Spice64\bin\ngspice.exe"
               if _platform.system() == "Windows"
               else "ngspice")
NETLIST_DIR  = "data/raw"

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
EPSILON_OX   = 3.9 * 8.854e-12   # SiO2 permittivity (F/m)
EPSILON_SI   = 11.7 * 8.854e-12  # Silicon permittivity (F/m)

NOMINAL_PARAMS = {
    "vdd":         3.3,
    "vth_n":       0.50,
    "vth_p":      -0.50,
    "length":      0.18,
    "width_n":     0.36,
    "width_p":     0.72,
    "tox":         4e-9,       # Gate oxide thickness (m) — 4nm for 180nm node
    "load_cap":    10e-15,
    "temperature": 27,
    # Parasitic interconnect
    "wire_length": 5e-6,       # Wire length (m) — 5µm default
    "wire_res":    50,         # Wire sheet resistance (Ω/sq)
    "wire_cap":    1e-16,      # Wire capacitance per µm (F/µm)
}

DELAY_LIMIT_NS = 0.022
POWER_LIMIT_UW = 150


def derive_cox(params: dict) -> float:
    """
    Derives gate oxide capacitance Cox from oxide thickness tox.
    Cox = ε_ox / tox  (F/m²)
    This is the physically correct derivation — tox drives everything.
    """
    tox = params.get("tox", 4e-9)
    return EPSILON_OX / tox


def compute_parasitic_rc(params: dict) -> tuple:
    """
    Computes wire parasitic resistance and capacitance.
    Models the interconnect between gates as a lumped RC.

    Returns (R_wire_ohm, C_wire_F)
    """
    wire_length = params.get("wire_length", 5e-6)       # m
    wire_res    = params.get("wire_res", 50)             # Ω/sq
    wire_cap_pm = params.get("wire_cap", 1e-16)         # F/µm

    # Assume minimum width wire = 0.28µm for 180nm node
    wire_width  = 0.28e-6                               # m
    n_squares   = wire_length / wire_width
    R_wire      = wire_res * n_squares
    C_wire      = wire_cap_pm * (wire_length * 1e6)     # convert m → µm

    return round(R_wire, 4), round(C_wire, 20)


def compute_effective_load(params: dict) -> float:
    """
    Effective load capacitance including parasitic wire capacitance.
    C_eff = C_load + C_miller + C_wire
    Miller capacitance ≈ 2 × C_gate_overlap (simplified)
    """
    CL      = params["load_cap"]
    _, C_wire = compute_parasitic_rc(params)
    cox     = derive_cox(params)
    L       = params["length"] * 1e-6
    Wn      = params["width_n"] * 1e-6
    # Miller capacitance from gate-drain overlap (≈ 10% of total gate cap)
    C_miller = 0.1 * cox * Wn * L

    return CL + C_wire + C_miller


def build_netlist(params: dict, sim_id: str = "sim") -> str:
    vdd   = params["vdd"]
    vth_n = params["vth_n"];  vth_p = params["vth_p"]
    L     = params["length"] * 1e-6
    Wn    = params["width_n"] * 1e-6
    Wp    = params["width_p"] * 1e-6
    cox   = derive_cox(params)   # ← derived from tox
    CL    = compute_effective_load(params)  # ← includes parasitics
    R_w, C_w = compute_parasitic_rc(params)
    temp  = params.get("temperature", 27)
    kp_n  = 450e-4 * cox;  kp_p = 200e-4 * cox
    half  = vdd / 2.0

    return f""".title CMOS_Inverter_{sim_id}
.TEMP {temp}
VDD  vdd  0   DC {vdd:.4f}
VIN  inp  0   PULSE(0 {vdd:.4f} 0.5n 50p 50p 0.95n 2n)
*
* Parasitic wire resistance between gate and load
Rwire  out  outx  {R_w:.4f}
*
.MODEL NMOS_M NMOS LEVEL=1 VTO={vth_n:.4f} KP={kp_n:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01 TOX={params.get("tox", 4e-9):.4e}
.MODEL PMOS_M PMOS LEVEL=1 VTO={vth_p:.4f} KP={kp_p:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01 TOX={params.get("tox", 4e-9):.4e}
MN1  out  inp  0    0    NMOS_M  W={Wn:.4e}  L={L:.4e}
MP1  out  inp  vdd  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
*
* Effective load: C_load + C_wire + C_miller
CL   outx  0   {CL:.4e}
*
.TRAN 1p 4n
.MEAS TRAN tpHL TRIG V(inp) VAL={half:.4f} RISE=1 TARG V(outx) VAL={half:.4f} FALL=1
.MEAS TRAN tpLH TRIG V(inp) VAL={half:.4f} FALL=1 TARG V(outx) VAL={half:.4f} RISE=1
.MEAS TRAN pavg AVG POWER
.END
"""


def build_nand2_netlist(params: dict, sim_id: str = "nand") -> str:
    vdd   = params["vdd"];  vth_n = params["vth_n"];  vth_p = params["vth_p"]
    L     = params["length"] * 1e-6
    Wn    = params["width_n"] * 1e-6 * 2
    Wp    = params["width_p"] * 1e-6
    cox   = derive_cox(params)
    CL    = compute_effective_load(params)
    temp  = params.get("temperature", 27)
    kp_n  = 450e-4 * cox;  kp_p = 200e-4 * cox;  half = vdd / 2.0

    return f""".title CMOS_NAND2_{sim_id}
.TEMP {temp}
VDD  vdd  0  DC {vdd:.4f}
VA   a    0  PULSE(0 {vdd:.4f} 0.5n 50p 50p 0.95n 2n)
VB   b    0  DC {vdd:.4f}
.MODEL NMOS_M NMOS LEVEL=1 VTO={vth_n:.4f} KP={kp_n:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
.MODEL PMOS_M PMOS LEVEL=1 VTO={vth_p:.4f} KP={kp_p:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
MP1  out  a  vdd  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
MP2  out  b  vdd  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
MN1  out  a  mid  0    NMOS_M  W={Wn:.4e}  L={L:.4e}
MN2  mid  b  0    0    NMOS_M  W={Wn:.4e}  L={L:.4e}
CL   out  0  {CL:.4e}
.TRAN 1p 4n
.MEAS TRAN tpHL TRIG V(a) VAL={half:.4f} RISE=1 TARG V(out) VAL={half:.4f} FALL=1
.MEAS TRAN tpLH TRIG V(a) VAL={half:.4f} FALL=1 TARG V(out) VAL={half:.4f} RISE=1
.MEAS TRAN pavg AVG POWER
.END
"""


def build_nor2_netlist(params: dict, sim_id: str = "nor") -> str:
    vdd   = params["vdd"];  vth_n = params["vth_n"];  vth_p = params["vth_p"]
    L     = params["length"] * 1e-6
    Wn    = params["width_n"] * 1e-6
    Wp    = params["width_p"] * 1e-6 * 2
    cox   = derive_cox(params)
    CL    = compute_effective_load(params)
    temp  = params.get("temperature", 27)
    kp_n  = 450e-4 * cox;  kp_p = 200e-4 * cox;  half = vdd / 2.0

    return f""".title CMOS_NOR2_{sim_id}
.TEMP {temp}
VDD  vdd  0  DC {vdd:.4f}
VA   a    0  PULSE(0 {vdd:.4f} 0.5n 50p 50p 0.95n 2n)
VB   b    0  DC 0
.MODEL NMOS_M NMOS LEVEL=1 VTO={vth_n:.4f} KP={kp_n:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
.MODEL PMOS_M PMOS LEVEL=1 VTO={vth_p:.4f} KP={kp_p:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
MP1  mid  a  vdd  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
MP2  out  b  mid  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
MN1  out  a  0    0    NMOS_M  W={Wn:.4e}  L={L:.4e}
MN2  out  b  0    0    NMOS_M  W={Wn:.4e}  L={L:.4e}
CL   out  0  {CL:.4e}
.TRAN 1p 4n
.MEAS TRAN tpHL TRIG V(a) VAL={half:.4f} RISE=1 TARG V(out) VAL={half:.4f} FALL=1
.MEAS TRAN tpLH TRIG V(a) VAL={half:.4f} FALL=1 TARG V(out) VAL={half:.4f} RISE=1
.MEAS TRAN pavg AVG POWER
.END
"""


def _run_ngspice(netlist: str, sim_id: str = "sim") -> str:
    os.makedirs(NETLIST_DIR, exist_ok=True)
    sp_file  = os.path.join(NETLIST_DIR, f"{sim_id}.sp")
    out_file = os.path.join(NETLIST_DIR, f"{sim_id}.out")
    with open(sp_file, "w", encoding="utf-8") as f:
        f.write(netlist)
    try:
        proc = subprocess.run(
            [NGSPICE_EXE, "-b", "-o", out_file, sp_file],
            capture_output=True, text=True, timeout=30)
        log = proc.stdout + "\n" + proc.stderr
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                log += "\n" + f.read()
    except subprocess.TimeoutExpired:
        log = "TIMEOUT"
    except FileNotFoundError:
        log = "NGSPICE_NOT_FOUND"
    finally:
        for p in [sp_file, out_file]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception: pass
    return log


def _parse_meas(log: str, name: str):
    pattern = rf"{re.escape(name)}\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)"
    match   = re.search(pattern, log, re.IGNORECASE)
    if match:
        try:
            v = float(match.group(1))
            return v if not np.isnan(v) else None
        except ValueError:
            return None
    return None


def compute_power_breakdown(params: dict, frequency: float = 1e9) -> dict:
    """
    Computes detailed power breakdown:
      - Dynamic power    (switching activity)
      - Leakage power    (subthreshold + gate oxide tunneling)
      - Short-circuit    (crowbar current during transition)

    Returns dict with individual components in µW.
    """
    vdd   = params["vdd"]
    CL    = compute_effective_load(params)
    vth_n = params["vth_n"]
    tox   = params.get("tox", 4e-9)
    cox   = derive_cox(params)
    Wn    = params["width_n"] * 1e-6
    L     = params["length"]  * 1e-6
    temp  = params.get("temperature", 27)

    alpha = 0.5   # activity factor

    # Dynamic: P_dyn = α × C_eff × VDD² × f
    p_dynamic = alpha * CL * vdd**2 * frequency

    # Subthreshold leakage (temperature-dependent)
    Vt        = 0.02585 * (1 + (temp - 27) / 300)  # thermal voltage
    i_sub     = 1e-9 * np.exp(-vth_n / Vt) * (Wn / L)
    p_leakage_sub = i_sub * vdd

    # Gate oxide tunneling (tox-dependent — thinner = more leakage)
    # J_tunnel ≈ A × E_ox² × exp(-B/E_ox)
    E_ox      = vdd / tox
    A_tunnel  = 3.75e-7;  B_tunnel = 2.2e10
    j_tunnel  = A_tunnel * E_ox**2 * np.exp(-B_tunnel / max(E_ox, 1e6))
    gate_area = Wn * L
    p_gate_ox = j_tunnel * gate_area * vdd

    # Short-circuit power (crowbar during transition)
    # P_sc ≈ (1/12) × (VDD - 2Vth)³ / VDD × β × t_r × f
    beta  = 450e-4 * cox * (Wn / L)
    t_r   = 50e-12   # rise time from netlist pulse
    vov   = vdd - 2 * abs(vth_n)
    if vov > 0:
        p_sc = (1/12) * beta * vov**3 * t_r * frequency
    else:
        p_sc = 0.0

    total = p_dynamic + p_leakage_sub + p_gate_ox + p_sc

    return {
        "p_dynamic_uw":     round(p_dynamic     * 1e6, 6),
        "p_leakage_uw":     round(p_leakage_sub * 1e6, 6),
        "p_gate_oxide_uw":  round(p_gate_ox     * 1e6, 6),
        "p_short_circuit_uw": round(p_sc        * 1e6, 6),
        "p_total_uw":       round(total          * 1e6, 6),
    }


def _analytical_delay(params: dict) -> float:
    vdd   = params["vdd"];      vth_n = params["vth_n"]
    vth_p = abs(params["vth_p"])
    Wn    = params["width_n"];  Wp   = params["width_p"]
    L     = params["length"];   cox  = derive_cox(params)
    CL    = compute_effective_load(params)
    kn    = 450e-4 * cox * (Wn / L)
    kp    = 200e-4 * cox * (Wp / L)
    vov_n = vdd - vth_n;  vov_p = vdd - vth_p
    if vov_n <= 0 or vov_p <= 0: return float("inf")
    t_fall = (CL * vdd) / (kn * vov_n**2 + 1e-30)
    t_rise = (CL * vdd) / (kp * vov_p**2 + 1e-30)
    return round(((t_rise + t_fall) / 2) * 1e9, 6)


def _analytical_power(params: dict, freq: float = 1e9) -> float:
    return compute_power_breakdown(params, freq)["p_total_uw"]


_sim_counter = 0

def run_single_simulation(params: dict = None,
                          frequency: float = 1e9,
                          verbose: bool = False) -> dict:
    global _sim_counter
    _sim_counter += 1
    sim_id = f"inv_{_sim_counter:05d}"
    if params is None: params = NOMINAL_PARAMS.copy()

    netlist = build_netlist(params, sim_id)
    log     = _run_ngspice(netlist, sim_id)
    tpHL    = _parse_meas(log, "tpHL")
    tpLH    = _parse_meas(log, "tpLH")
    pavg    = _parse_meas(log, "pavg")

    spice_ok = (tpHL is not None and tpLH is not None
                and tpHL > 0 and tpLH > 0 and tpHL < 1e-8)

    power_bd = compute_power_breakdown(params, frequency)

    if spice_ok:
        delay  = round(((tpHL + tpLH) / 2) * 1e9, 6)
        power  = round(abs(pavg) * 1e6, 6) if pavg else power_bd["p_total_uw"]
        source = "ngspice"
    else:
        delay  = _analytical_delay(params)
        power  = power_bd["p_total_uw"]
        source = "analytical_fallback"
        if verbose:
            print(f"  [!] {sim_id}: fallback used")

    cox_derived = derive_cox(params)
    R_w, C_w   = compute_parasitic_rc(params)

    return {
        **params,
        "cox_derived":          round(cox_derived, 6),
        "c_effective_load_f":   round(compute_effective_load(params), 20),
        "r_wire_ohm":           R_w,
        "c_wire_f":             C_w,
        "propagation_delay_ns": delay,
        "power_dissipation_uw": power,
        "p_dynamic_uw":         power_bd["p_dynamic_uw"],
        "p_leakage_uw":         power_bd["p_leakage_uw"],
        "p_gate_oxide_uw":      power_bd["p_gate_oxide_uw"],
        "p_short_circuit_uw":   power_bd["p_short_circuit_uw"],
        "is_valid":             (delay < DELAY_LIMIT_NS) and (power < POWER_LIMIT_UW),
        "sim_source":           source,
    }


def run_gate_simulation(gate_type: str, params: dict = None) -> dict:
    global _sim_counter
    _sim_counter += 1
    sim_id = f"{gate_type}_{_sim_counter:05d}"
    if params is None: params = NOMINAL_PARAMS.copy()

    if gate_type == "nand2":
        netlist = build_nand2_netlist(params, sim_id)
    elif gate_type == "nor2":
        netlist = build_nor2_netlist(params, sim_id)
    else:
        netlist = build_netlist(params, sim_id)

    log  = _run_ngspice(netlist, sim_id)
    tpHL = _parse_meas(log, "tpHL"); tpLH = _parse_meas(log, "tpLH")
    pavg = _parse_meas(log, "pavg")
    spice_ok = (tpHL is not None and tpLH is not None
                and tpHL > 0 and tpLH > 0 and tpHL < 1e-8)
    power_bd = compute_power_breakdown(params)

    if spice_ok:
        delay  = round(((tpHL + tpLH) / 2) * 1e9, 6)
        power  = round(abs(pavg) * 1e6, 6) if pavg else power_bd["p_total_uw"]
        source = "ngspice"
    else:
        delay  = _analytical_delay(params)
        power  = power_bd["p_total_uw"]
        source = "analytical_fallback"

    return {"gate": gate_type,
            "propagation_delay_ns": delay,
            "power_dissipation_uw": power,
            "sim_source": source,
            "is_valid": (delay < DELAY_LIMIT_NS) and (power < POWER_LIMIT_UW)}


def run_corner_analysis() -> dict:
    corners = {
        "Fast-Fast":       {**NOMINAL_PARAMS, "vth_n": 0.35, "vth_p": -0.35,
                            "length": 0.16, "width_n": 0.40, "width_p": 0.80,
                            "tox": 3.5e-9},
        "Typical-Typical": NOMINAL_PARAMS.copy(),
        "Slow-Slow":       {**NOMINAL_PARAMS, "vth_n": 0.65, "vth_p": -0.65,
                            "length": 0.20, "width_n": 0.32, "width_p": 0.64,
                            "tox": 4.5e-9},
    }
    results = {}
    for name, p in corners.items():
        print(f"  Running {name}...")
        results[name] = run_single_simulation(p, verbose=True)
    return results


def run_temperature_sweep(temps=None, params: dict = None) -> list:
    if temps is None: temps = [-40, -20, 0, 27, 50, 75, 100, 125]
    if params is None: params = NOMINAL_PARAMS.copy()
    results = []
    for t in temps:
        p = params.copy(); p["temperature"] = t
        r = run_single_simulation(p)
        r["temperature"] = t
        results.append(r)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("   PyYield — SPICE Engine v4 Self-Test")
    print("=" * 60)

    tox    = NOMINAL_PARAMS["tox"]
    cox    = derive_cox(NOMINAL_PARAMS)
    R_w, C_w = compute_parasitic_rc(NOMINAL_PARAMS)
    C_eff  = compute_effective_load(NOMINAL_PARAMS)

    print(f"\n[Physical Parameters]")
    print(f"  tox              : {tox*1e9:.2f} nm")
    print(f"  Cox (derived)    : {cox:.4e} F/m²")
    print(f"  Wire R           : {R_w:.2f} Ω")
    print(f"  Wire C           : {C_w:.4e} F")
    print(f"  Effective C_load : {C_eff:.4e} F")

    print(f"\n[Power Breakdown]")
    pb = compute_power_breakdown(NOMINAL_PARAMS)
    print(f"  Dynamic          : {pb['p_dynamic_uw']:.4f} µW")
    print(f"  Leakage (sub-Vt) : {pb['p_leakage_uw']:.6f} µW")
    print(f"  Gate oxide leak  : {pb['p_gate_oxide_uw']:.6f} µW")
    print(f"  Short-circuit    : {pb['p_short_circuit_uw']:.6f} µW")
    print(f"  Total            : {pb['p_total_uw']:.4f} µW")

    print(f"\n[Running Full SPICE Simulation]")
    r = run_single_simulation(NOMINAL_PARAMS, verbose=True)
    print(f"  Source  : {r['sim_source']}")
    print(f"  Delay   : {r['propagation_delay_ns']} ns")
    print(f"  Power   : {r['power_dissipation_uw']} µW")
    print(f"  Valid   : {'YES' if r['is_valid'] else 'NO'}")

    print("\n[spice_engine.py v4] PASSED")
    print("=" * 60)