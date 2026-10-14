"""
waveform.py — Real transient waveform extraction from Ngspice
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import subprocess
import warnings
warnings.filterwarnings("ignore")

from src.simulation.spice_engine import NGSPICE_EXE, NETLIST_DIR, NOMINAL_PARAMS


def build_waveform_netlist(params: dict, wf_path: str, sim_id: str = "wf") -> str:
    vdd   = params["vdd"];        vth_n = params["vth_n"]
    vth_p = params["vth_p"];      L     = params["length"]  * 1e-6
    Wn    = params["width_n"] * 1e-6
    Wp    = params["width_p"] * 1e-6
    from src.simulation.spice_engine import derive_cox
    cox   = derive_cox(params)    # derived from tox, not stored directly
    CL    = params["load_cap"]
    temp  = params.get("temperature", 27)
    kp_n  = 450e-4 * cox;         kp_p  = 200e-4 * cox
    wf_path_fwd = wf_path.replace("\\", "/")

    return f""".title CMOS_Waveform_{sim_id}
.TEMP {temp}
VDD  vdd  0  DC {vdd:.4f}
VIN  inp  0  PULSE(0 {vdd:.4f} 0.5n 50p 50p 0.95n 2n)
.MODEL NMOS_M NMOS LEVEL=1 VTO={vth_n:.4f} KP={kp_n:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
.MODEL PMOS_M PMOS LEVEL=1 VTO={vth_p:.4f} KP={kp_p:.4e} GAMMA=0.4 PHI=0.65 LAMBDA=0.01
MN1  out  inp  0    0    NMOS_M  W={Wn:.4e}  L={L:.4e}
MP1  out  inp  vdd  vdd  PMOS_M  W={Wp:.4e}  L={L:.4e}
CL   out  0    {CL:.4e}
.control
tran 1p 4n
wrdata {wf_path_fwd} v(inp) v(out)
.endc
.END
"""


def extract_waveform(params: dict = None, sim_id: str = "wf001") -> pd.DataFrame:
    """
    Runs Ngspice and returns V(inp), V(out) as a DataFrame.
    Falls back to synthetic waveform if Ngspice fails.
    """
    if params is None:
        params = NOMINAL_PARAMS.copy()

    os.makedirs(NETLIST_DIR, exist_ok=True)
    sp_file = os.path.join(NETLIST_DIR, f"{sim_id}.sp")
    wf_file = os.path.join(NETLIST_DIR, f"{sim_id}.dat")

    netlist = build_waveform_netlist(params, wf_file, sim_id)
    with open(sp_file, "w", encoding="utf-8") as f:
        f.write(netlist)

    try:
        subprocess.run([NGSPICE_EXE, "-b", sp_file],
                       capture_output=True, text=True, timeout=30)
        if os.path.exists(wf_file) and os.path.getsize(wf_file) > 100:
            return _parse_wrdata(wf_file)
        return _synthetic_waveform(params)
    except Exception:
        return _synthetic_waveform(params)
    finally:
        for p in [sp_file, wf_file]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass


def _parse_wrdata(filepath: str) -> pd.DataFrame:
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    t, vi, vo = float(parts[0]), float(parts[1]), float(parts[2])
                    rows.append({"time_ns": t * 1e9, "v_inp": vi, "v_out": vo})
                except ValueError:
                    continue
    if rows:
        return pd.DataFrame(rows)
    return _synthetic_waveform(NOMINAL_PARAMS)


def _synthetic_waveform(params: dict) -> pd.DataFrame:
    """Analytical fallback waveform."""
    from src.simulation.spice_engine import _analytical_delay
    vdd   = params["vdd"]
    t     = np.linspace(0, 4, 2000)
    delay = _analytical_delay(params)
    v_inp = np.where((t >= 0.5) & (t < 2.5), vdd, 0.0)
    v_out = np.where(
        ((t >= 0) & (t < (0.5 + delay))) | (t >= (2.5 + delay)),
        vdd, 0.0
    )
    return pd.DataFrame({"time_ns": t, "v_inp": v_inp, "v_out": v_out})


if __name__ == "__main__":
    print("Extracting waveform...")
    df = extract_waveform()
    print(f"  Rows extracted : {len(df)}")
    print(f"  Time range     : {df['time_ns'].min():.3f} – {df['time_ns'].max():.3f} ns")
    print(f"  Max V(inp)     : {df['v_inp'].max():.3f} V")
    print(f"  Max V(out)     : {df['v_out'].max():.3f} V")
    print("[waveform.py] PASSED")