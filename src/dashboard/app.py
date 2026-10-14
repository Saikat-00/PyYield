"""
app.py — PyYield Professional Dashboard v2
9 pages: Overview, Monte Carlo, Yield Oracle, Wafer Map,
         Corner Analysis, Waveform Viewer, PVT Analysis,
         Multi-Gate, Report Export
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings

# ── Cloud auto-setup ──────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "../..")))
try:
    from setup_cloud import ensure_setup
    ensure_setup()
except Exception as _e:
    pass   # Don't block the app if setup fails
# ─────────────────────────────────────────────

def pdf_pages(path):
    try:
        with open(path, "rb") as f:
            content = f.read()
            return content.count(b"/Page ")
    except Exception:
        return "?"
        
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PyYield EDA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="metric-container"] {
    background: #1e1e35;
    border: 1px solid #2d2d50;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    font-size: 12px !important;
    color: #9090b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 500 !important;
}
div.stButton > button {
    background: #7F77DD;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 28px;
    font-weight: 500;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #6a62c4; }
div.stButton > button:active { transform: scale(0.98); }
.page-title {
    font-size: 26px;
    font-weight: 500;
    color: #e8e8f8;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 13px;
    color: #7070a0;
    margin-bottom: 1.2rem;
}
.stat-label {
    font-size: 11px;
    color: #7070a0;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.info-box {
    background: #1a1a30;
    border: 1px solid #2d2d50;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    from src.ml.surrogate_model import load_model
    try:
        dm, ds = load_model("delay")
        pm, ps = load_model("power")
        return dm, ds, pm, ps
    except Exception:
        return None, None, None, None


@st.cache_data
def cached_monte_carlo(n, method, seed):
    from src.simulation.monte_carlo import run_monte_carlo, compute_yield_statistics, compute_process_capability, compute_convergence
    df    = run_monte_carlo(n_samples=n, method=method, seed=seed)
    stats = compute_yield_statistics(df)
    cpk   = compute_process_capability(df)
    conv  = compute_convergence(df)
    return df, stats, cpk, conv


@st.cache_data
def cached_wafer(grid_size, seed):
    from src.wafer.spatial_model import simulate_full_wafer, compute_wafer_yield_map
    df    = simulate_full_wafer(grid_size=grid_size, seed=seed)
    stats = compute_wafer_yield_map(df)
    return df, stats


@st.cache_data
def cached_temp_sweep(temp_min, temp_max, n_steps, _params_key):
    from src.simulation.spice_engine import run_single_simulation, NOMINAL_PARAMS
    temps   = np.linspace(temp_min, temp_max, n_steps).astype(int)
    results = []
    p = NOMINAL_PARAMS.copy()
    for t in temps:
        p["temperature"] = int(t)
        r = run_single_simulation(p)
        r["temperature"] = int(t)
        results.append(r)
    return pd.DataFrame(results)


@st.cache_data
def cached_gate_comparison(_params_key):
    from src.simulation.spice_engine import run_gate_simulation, NOMINAL_PARAMS
    results = []
    for gate in ["inverter", "nand2", "nor2"]:
        r = run_gate_simulation(gate, NOMINAL_PARAMS.copy())
        results.append(r)
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ PyYield EDA")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Monte Carlo",
        "🤖  Yield Oracle",
        "🗺️  Wafer Map",
        "📐  Corner Analysis",
        "〰️  Waveform Viewer",
        "🌡️  PVT Analysis",
        "⚙️  Multi-Gate",
        "⏳  Aging Analysis",
        "⚡  Power Breakdown",
        "🌪️  Tornado Chart",
        "📄  Report Export",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Process Parameters**")

    vth_n   = st.slider("Vth_n (V)",    0.25, 0.75, 0.50, 0.01)
    vth_p   = st.slider("Vth_p (V)",   -0.75,-0.25,-0.50, 0.01)
    length  = st.slider("Length (µm)",  0.0224, 0.22, 0.0228, 0.01)
    width_n = st.slider("Width_n (µm)", 0.28, 0.44, 0.36, 0.01)
    width_p     = st.slider("Width_p (µm)",  0.56, 0.88, 0.72, 0.01)
    tox_nm      = st.slider("tox (nm)",       2.0,  8.0,  4.0,  0.022)
    wire_len_um = st.slider("Wire Length (µm)", 1.0, 20.0, 5.0, 0.5)
    temp        = st.slider("Temp (°C)",    -40,  125,   27,   5)

    st.markdown("---")
    st.caption("PyYield v2.0 · B.Tech Capstone")

CURRENT_PARAMS = {
    "vdd": 3.3, "vth_n": vth_n, "vth_p": vth_p,
    "length": length, "width_n": width_n, "width_p": width_p,
    "tox": tox_nm * 1e-9,          # nm → m
    "wire_length": wire_len_um * 1e-6,  # µm → m
    "wire_res": 50, "wire_cap": 1e-16,
    "load_cap": 10e-15, "temperature": temp,
}
PARAMS_KEY = str(sorted(CURRENT_PARAMS.items()))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def verdict_badge(passes):
    if passes:
        return st.success("### ✓ CHIP PASSES YIELD")
    else:
        return st.error("### ✗ CHIP FAILS YIELD")


def section(title, subtitle=""):
    st.markdown(f'<p class="page-title">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown("---")


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if "Overview" in page:
    section("⚡ PyYield Dashboard",
            "Open-source CMOS EDA tool — yield prediction & process variation analysis")

    from src.simulation.spice_engine import run_single_simulation
    with st.spinner("Running live simulation..."):
        result = run_single_simulation(CURRENT_PARAMS)

    delay  = result["propagation_delay_ns"]
    power  = result["power_dissipation_uw"]
    passes = result["is_valid"]
    source = result["sim_source"]

    c1, c2, c3, c4, c5 = st.columns(5)
    
    # UPDATED LIMITS: 0.032 ns and 60.0 uW
    c1.metric("Propagation Delay", f"{delay:.5f} ns",
              delta="✓ Pass" if delay < 0.032 else "✗ Fail")
    c2.metric("Power Dissipation", f"{power:.2f} µW",
              delta="✓ Pass" if power < 60.0 else "✗ Fail")
    c3.metric("Yield Verdict",     "✓ PASS" if passes else "✗ FAIL")
    c4.metric("Temperature",       f"{temp} °C")
    c5.metric("Sim Engine",        source.replace("_", " ").title())

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("What PyYield Does")
        r1, r2 = st.columns(2)
        r1.info("**Module 1 — Physics Engine**\n\nReal Ngspice SPICE transient simulations with LEVEL-1 MOSFET models")
        r2.success("**Module 2 — Yield Oracle**\n\nRandom Forest ML surrogate model for instant yield classification")
        r3, r4 = st.columns(2)
        r3.warning("**Module 3 — Wafer Map**\n\nSpatial yield degradation modeling across a 300mm silicon wafer")
        r4.error("**Module 4 — PVT Analysis**\n\nProcess + Voltage + Temperature corner sweeps")

    with col_b:
        sensitivity = {"Vth_n": 0.42, "Length": 0.31, "Width_n": 0.0222,
                       "Width_p": 0.09, "Vth_p": 0.04, "Load_cap": 0.02}
        fig = px.bar(x=list(sensitivity.values()), y=list(sensitivity.keys()),
                     orientation="h", title="Parameter Sensitivity (Delay)",
                     color=list(sensitivity.values()),
                     color_continuous_scale="Purples")
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.caption(f"Live result from **{source}** engine at {time.strftime('%H:%M:%S')} | "
               f"Vth_n={vth_n}V | L={length}µm | T={temp}°C")


# ══════════════════════════════════════════════════════
# PAGE 2 — MONTE CARLO
# ══════════════════════════════════════════════════════
elif "Monte Carlo" in page:
    section("📊 Monte Carlo Analysis",
            "Simulate thousands of chips with randomized process variations")

    c1, c2, c3 = st.columns(3)
    n_samples = c1.selectbox("Simulations", [200, 500, 1000], index=1)
    method    = c2.selectbox("Sampling", ["lhs", "mc"])
    seed      = c3.number_input("Seed", value=42, step=1)

    if st.button("▶ Run Monte Carlo"):
        with st.spinner(f"Running {n_samples} Ngspice simulations..."):
            df, stats, cpk, conv = cached_monte_carlo(n_samples, method, int(seed))
        st.success(f"✓ {n_samples} simulations complete")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Yield",         f"{stats['yield_percent']}%")
        c2.metric("Mean Delay",    f"{stats['mean_delay_ns']} ns")
        c3.metric("Std Dev",       f"{stats['std_delay_ns']} ns")
        c4.metric("3σ Bound",      f"{stats['delay_3sigma_ns']} ns")
        c5.metric("Delay Cpk",     str(cpk["delay_Cpk"]),
                  delta="Capable ✓" if cpk["delay_Cpk"] >= 1.33 else "Not capable")
        c6.metric("Power Cpk",     str(cpk["power_Cpk"]),
                  delta="Capable ✓" if cpk["power_Cpk"] >= 1.33 else "Not capable")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = px.histogram(df, x="propagation_delay_ns", nbins=40,
                                title="Delay Distribution",
                                color_discrete_sequence=["#7F77DD"])
            fig1.add_vline(x=stats["mean_delay_ns"], line_dash="dash",
                           line_color="white", annotation_text="Mean")
            fig1.add_vline(x=stats["delay_3sigma_ns"], line_dash="dot",
                           line_color="red", annotation_text="3σ")
            d_rng = df["propagation_delay_ns"].max() - df["propagation_delay_ns"].min()
            if d_rng > 0:
                fig1.update_yaxes()
                fig1.update_xaxes(range=[
                    df["propagation_delay_ns"].min() - d_rng*0.022,
                    df["propagation_delay_ns"].max() + d_rng*0.022
                ])
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            fig2 = px.histogram(df, x="power_dissipation_uw", nbins=40,
                                title="Power Distribution",
                                color_discrete_sequence=["#00CC96"])
            p_rng = df["power_dissipation_uw"].max() - df["power_dissipation_uw"].min()
            if p_rng > 0:
                fig2.update_xaxes(range=[
                    df["power_dissipation_uw"].min() - p_rng*0.022,
                    df["power_dissipation_uw"].max() + p_rng*0.022
                ])
            st.plotly_chart(fig2, use_container_width=True)

        # Convergence plot
        st.subheader("Monte Carlo Convergence — Yield Stabilization")
        fig_conv = px.line(conv, x="n_samples", y="yield_pct",
                           title="Yield Estimate vs Sample Size",
                           labels={"n_samples": "Chips Simulated",
                                   "yield_pct": "Yield Estimate (%)"},
                           color_discrete_sequence=["#7F77DD"],
                           markers=True)
        fig_conv.add_hline(y=stats["yield_percent"], line_dash="dash",
                           line_color="green", annotation_text="Final yield")
        st.plotly_chart(fig_conv, use_container_width=True)

        # 3D Pareto
        st.subheader("3D Pareto Front — Speed vs Power vs Vth_n")
        fig3 = px.scatter_3d(df, x="propagation_delay_ns", y="power_dissipation_uw",
                             z="vth_n", color="is_valid",
                             color_discrete_map={True: "#00CC96", False: "#EF553B"},
                             opacity=0.7, height=520,
                             labels={"propagation_delay_ns": "Delay (ns)",
                                     "power_dissipation_uw": "Power (µW)",
                                     "vth_n": "Vth_n (V)",
                                     "is_valid": "Passes Yield"})
        st.plotly_chart(fig3, use_container_width=True)

        # Cpk summary table
        st.subheader("Process Capability Summary")
        cpk_data = {
            "Metric": ["Propagation Delay", "Power Dissipation"],
            "Cp":  [cpk["delay_Cp"],  cpk["power_Cp"]],
            "Cpk": [cpk["delay_Cpk"], cpk["power_Cpk"]],
            "Sigma Level": [cpk["delay_sigma_level"], cpk["power_sigma_level"]],
            "Process Capable (Cpk≥1.33)": [
                "✓ YES" if cpk["delay_Cpk"] >= 1.33 else "✗ NO",
                "✓ YES" if cpk["power_Cpk"] >= 1.33 else "✗ NO",
            ]
        }
        st.dataframe(pd.DataFrame(cpk_data), hide_index=True, use_container_width=True)

        # Correlation heatmap
        st.subheader("Parameter Correlation Matrix")
        corr_cols = ["vth_n", "length", "width_n", "width_p",
                     "propagation_delay_ns", "power_dissipation_uw"]
        fig4 = px.imshow(df[corr_cols].corr(), text_auto=".2f",
                         color_continuous_scale="RdBu_r", aspect="auto",
                         title="Correlation Matrix")
        st.plotly_chart(fig4, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Results CSV", csv,
                           "monte_carlo_results.csv", "text/csv")
    else:
        st.info("Set parameters above and click **Run Monte Carlo** to begin.")


# ══════════════════════════════════════════════════════
# PAGE 3 — YIELD ORACLE
# ══════════════════════════════════════════════════════
elif "Yield Oracle" in page:
    section("🤖 ML Yield Oracle",
            "Instant chip classification using trained Random Forest surrogate model")

    dm, ds, pm, ps = load_models()

    if dm is None:
        st.warning("⚠️ Models not found. Run **Monte Carlo** first, then run:\n"
                   "```\npython src\\ml\\surrogate_model.py\n```")
    else:
        from src.ml.surrogate_model import predict, get_shap_summary
        pred_delay = predict(dm, ds, CURRENT_PARAMS)
        pred_power = predict(pm, ps, CURRENT_PARAMS)
        
        # UPDATED LIMITS: 0.032 ns and 60.0 uW
        passes = (pred_delay < 0.022) and (pred_power < 105.0)
        delay_margin = round(100 * (0.022 - pred_delay) / 0.022, 2)
        power_margin = round(100 * (105.0 - pred_power) / 105.0, 2)

        verdict_badge(passes)

        c1, c2, c3, c4 = st.columns(4)
        
        # UPDATED MATH FOR LIMITS
        delay_margin = round(100 * (0.032 - pred_delay) / 0.032,  2)
        power_margin = round(100 * (60.0 - pred_power) / 60.0, 2)
        
        c1.metric("Predicted Delay",  f"{pred_delay:.5f} ns")
        c2.metric("Predicted Power",  f"{pred_power:.2f} µW")
        c3.metric("Delay Margin",     f"{delay_margin}%",
                  delta="Safe" if delay_margin > 0 else "Exceeded")
        c4.metric("Power Margin",     f"{power_margin}%",
                  delta="Safe" if power_margin > 0 else "Exceeded")

        st.markdown("---")

        # SHAP explainability
        st.subheader("SHAP Feature Importance")
        try:
            from sklearn.preprocessing import StandardScaler
            import pandas as pd
            csv_path = "data/processed/monte_carlo_results.csv"
            if os.path.exists(csv_path):
                raw_df = pd.read_csv(csv_path)
                from src.ml.surrogate_model import FEATURE_COLS
                X_raw  = raw_df[FEATURE_COLS].values
                shap_df = get_shap_summary(dm, ds, X_raw)

                fig_shap = px.bar(shap_df, x="mean_abs_shap", y="feature",
                                  orientation="h",
                                  title=f"SHAP Values — Feature Impact on Delay "
                                        f"({shap_df['source'].iloc[0]})",
                                  color="mean_abs_shap",
                                  color_continuous_scale="Purples",
                                  labels={"mean_abs_shap": "Mean |SHAP|",
                                          "feature": "Parameter"})
                fig_shap.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("Higher SHAP value = greater impact on delay prediction. "
                           "Vth_n and Channel Length dominate yield outcomes.")
        except Exception as e:
            st.info(f"SHAP chart requires training data CSV. Run Monte Carlo first. ({e})")

        # Parameter sweep
        st.markdown("---")
        st.subheader("Parameter Sensitivity Sweep")
        from src.ml.yield_oracle import parameter_sweep

        sweep_param = st.selectbox("Sweep parameter",
                                   ["vth_n", "length", "width_n", "width_p"])
        sweep_ranges = {
            "vth_n":   np.linspace(0.25, 0.75, 40),
            "length":  np.linspace(0.0224, 0.22, 40),
            "width_n": np.linspace(0.28, 0.44, 40),
            "width_p": np.linspace(0.56, 0.88, 40),
        }
        sweep_df = parameter_sweep(sweep_param, sweep_ranges[sweep_param],
                                   dm, ds, pm, ps, CURRENT_PARAMS)

        col_a, col_b = st.columns(2)
        with col_a:
            d_min = sweep_df["predicted_delay_ns"].min()
            d_max = sweep_df["predicted_delay_ns"].max()
            d_pad = max((d_max - d_min) * 0.3, d_max * 0.05)
            fig_d = px.line(sweep_df, x=sweep_param, y="predicted_delay_ns",
                            title=f"Delay vs {sweep_param}",
                            color_discrete_sequence=["#7F77DD"], markers=True)
            # UPDATED DELAY SPEC PLOT LINE
            fig_d.add_hline(y=0.022, line_dash="dash", line_color="red",
                                                                      annotation_text="Spec: 22 ps")
            fig_d.update_yaxes(range=[d_min - d_pad, d_max + d_pad])
            fig_d.update_layout(annotations=[dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=f"Range: {d_min:.5f} – {d_max:.5f} ns",
                showarrow=False, font=dict(size=11))])
            st.plotly_chart(fig_d, use_container_width=True)

        with col_b:
            p_min = sweep_df["predicted_power_uw"].min()
            p_max = sweep_df["predicted_power_uw"].max()
            p_pad = max((p_max - p_min) * 0.3, p_max * 0.05)
            fig_p = px.line(sweep_df, x=sweep_param, y="predicted_power_uw",
                            title=f"Power vs {sweep_param}",
                            color_discrete_sequence=["#00CC96"], markers=True)
            # UPDATED POWER SPEC PLOT LINE
            fig_p.add_hline(y=105.0, line_dash="dash", line_color="red",
                            annotation_text="Spec: 105 µW")
            fig_p.update_yaxes(range=[p_min - p_pad, p_max + p_pad])
            fig_p.update_layout(annotations=[dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=f"Range: {p_min:.3f} – {p_max:.3f} µW",
                showarrow=False, font=dict(size=11))])
            st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — WAFER MAP
# ══════════════════════════════════════════════════════
elif "Wafer Map" in page:
    section("🗺️ Wafer Spatial Yield Map",
            "300mm wafer — edge dies have higher process variation than center")

    c1, c2 = st.columns(2)
    grid_size = c1.slider("Grid Resolution", 10, 25, 15, 1)
    w_seed    = c2.number_input("Seed", value=42, step=1)

    if st.button("▶ Simulate Wafer"):
        with st.spinner("Simulating dies across wafer..."):
            df, stats = cached_wafer(grid_size, int(w_seed))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Dies",    stats["total_dies"])
        c2.metric("Overall Yield", f"{stats['overall_yield']}%")
        c3.metric("Center Yield",  f"{stats['center_yield']}%")
        c4.metric("Edge Yield",    f"{stats['edge_yield']}%")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_h = px.scatter(df, x="x_mm", y="y_mm",
                               color="propagation_delay_ns",
                               color_continuous_scale="RdYlGn_r",
                               title="Delay Heatmap",
                               size_max=10, size=[6]*len(df))
            fig_h.update_layout(height=460)
            st.plotly_chart(fig_h, use_container_width=True)

        with col_b:
            fig_p = px.scatter(df, x="x_mm", y="y_mm", color="is_valid",
                               color_discrete_map={True: "#00CC96", False: "#EF553B"},
                               title="Pass / Fail Map",
                               size_max=10, size=[6]*len(df))
            fig_p.update_layout(height=460)
            st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Yield Degradation — Center to Edge")
        df["radius_bin"] = pd.cut(df["radius_mm"], bins=8)
        radial = df.groupby("radius_bin", observed=True).agg(
            yield_pct=("is_valid", lambda x: round(100*x.mean(), 1)),
            mean_delay=("propagation_delay_ns", "mean")
        ).reset_index()
        radial["label"] = radial["radius_bin"].astype(str)
        fig_r = px.bar(radial, x="label", y="yield_pct",
                       title="Yield % by Radial Zone",
                       color="yield_pct", color_continuous_scale="RdYlGn",
                       labels={"label": "Radius Band (mm)", "yield_pct": "Yield (%)"})
        fig_r.add_hline(y=95, line_dash="dash", line_color="white",
                        annotation_text="95% Target")
        st.plotly_chart(fig_r, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download Wafer Data CSV", csv,
                           "wafer_data.csv", "text/csv")
    else:
        st.info("Click **Simulate Wafer** to generate the spatial yield map.")


# ══════════════════════════════════════════════════════
# PAGE 5 — CORNER ANALYSIS
# ══════════════════════════════════════════════════════
elif "Corner" in page:
    section("📐 PVT Corner Analysis",
            "Fast-Fast / Typical-Typical / Slow-Slow characterization")

    c1, c2, c3 = st.columns(3)
    ff_vth = c1.slider("Fast-Fast Vth_n",  0.25, 0.45, 0.35, 0.01)
    tt_vth = c2.slider("Typical Vth_n",    0.40, 0.60, 0.50, 0.01)
    ss_vth = c3.slider("Slow-Slow Vth_n",  0.55, 0.75, 0.65, 0.01)

    if st.button("▶ Run Corner Analysis"):
        from src.simulation.spice_engine import run_single_simulation, NOMINAL_PARAMS
        corners = {
            "Fast-Fast":       run_single_simulation({**NOMINAL_PARAMS, "vth_n": ff_vth, "vth_p": -ff_vth, "length": 0.0226, "width_n": 0.40, "width_p": 0.80}),
            "Typical-Typical": run_single_simulation({**NOMINAL_PARAMS, "vth_n": tt_vth, "vth_p": -tt_vth}),
            "Slow-Slow":       run_single_simulation({**NOMINAL_PARAMS, "vth_n": ss_vth, "vth_p": -ss_vth, "length": 0.20, "width_n": 0.32, "width_p": 0.64}),
        }

        rows = [{"Corner": n, "Source": r["sim_source"],
                 "Delay (ns)": r["propagation_delay_ns"],
                 "Power (µW)": r["power_dissipation_uw"],
                 "Status": "✓ PASS" if r["is_valid"] else "✗ FAIL"}
                for n, r in corners.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown("---")

        col_a, col_b = st.columns(2)
        colors = ["#636EFA", "#00CC96", "#EF553B"]

        with col_a:
            fig_radar = go.Figure()
            cats = ["Delay (ns)", "Power (µW)", "Delay (ns)"]
            for (name, res), color in zip(corners.items(), colors):
                vals = [res["propagation_delay_ns"],
                        res["power_dissipation_uw"],
                        res["propagation_delay_ns"]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    name=name, line_color=color))
            fig_radar.update_layout(title="Corner Radar", height=400,
                                    polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_b:
            corner_df = pd.DataFrame(rows)
            fig_bar = px.bar(corner_df, x="Corner", y="Delay (ns)",
                             color="Corner", title="Delay by Corner",
                             color_discrete_sequence=colors)
            st.plotly_chart(fig_bar, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.success("**Fast-Fast**\nLow Vth, short L. Best speed. Risk: hold-time violation.")
        c2.info("**Typical-Typical**\nNominal design center. Optimization target.")
        c3.warning("**Slow-Slow**\nHigh Vth, long L. Worst speed. Must meet timing spec.")
    else:
        st.info("Adjust corner thresholds above and click **Run Corner Analysis**.")


# ══════════════════════════════════════════════════════
# PAGE 6 — WAVEFORM VIEWER
# ══════════════════════════════════════════════════════
elif "Waveform" in page:
    section("〰️ Waveform Viewer",
            "Real V(inp) and V(out) transient waveforms from Ngspice simulation")

    st.info("Adjust parameters in the sidebar and click **Capture Waveform** to see "
            "real transient data from the Ngspice engine.")

    if st.button("▶ Capture Waveform"):
        from src.simulation.waveform import extract_waveform
        with st.spinner("Running Ngspice transient simulation..."):
            wf_df = extract_waveform(params=CURRENT_PARAMS)

        vdd = CURRENT_PARAMS["vdd"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wf_df["time_ns"], y=wf_df["v_inp"],
                                 name="V(inp) — Input",
                                 line=dict(color="#7F77DD", width=2)))
        fig.add_trace(go.Scatter(x=wf_df["time_ns"], y=wf_df["v_out"],
                                 name="V(out) — Output",
                                 line=dict(color="#00CC96", width=2)))

        # 50% crossing lines
        fig.add_hline(y=vdd/2, line_dash="dot", line_color="gray",
                      annotation_text=f"VDD/2 = {vdd/2:.2f}V")
        fig.add_hline(y=vdd, line_dash="dash", line_color="#555",
                      annotation_text=f"VDD = {vdd}V")

        fig.update_layout(
            title="CMOS Inverter Transient Response",
            xaxis_title="Time (ns)",
            yaxis_title="Voltage (V)",
            height=480,
            yaxis=dict(range=[-0.2, vdd + 0.3]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Measurements
        from src.simulation.spice_engine import run_single_simulation
        r = run_single_simulation(CURRENT_PARAMS)

        st.markdown("---")
        st.subheader("Extracted Measurements")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("tpd (avg)",    f"{r['propagation_delay_ns']:.5f} ns")
        c2.metric("Power (avg)",  f"{r['power_dissipation_uw']:.2f} µW")
        c3.metric("Source",       r["sim_source"].replace("_", " "))
        c4.metric("Temperature",  f"{temp} °C")

        csv = wf_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Waveform CSV", csv,
                           "waveform_data.csv", "text/csv")
    else:
        st.info("Click **Capture Waveform** to run the transient simulation.")


# ══════════════════════════════════════════════════════
# PAGE 7 — PVT ANALYSIS
# ══════════════════════════════════════════════════════
elif "PVT" in page:
    section("🌡️ PVT Analysis — Temperature Sweep",
            "Full Process + Voltage + Temperature characterization (−40°C to 125°C)")

    c1, c2 = st.columns(2)
    t_min   = c1.slider("Min Temperature (°C)", -40, 25,  -40, 5)
    t_max   = c2.slider("Max Temperature (°C)",  50, 125, 125, 5)
    n_steps = st.slider("Number of Temperature Points", 5, 20, 10, 1)

    if st.button("▶ Run Temperature Sweep"):
        with st.spinner(f"Running {n_steps} SPICE simulations across temperature..."):
            df_pvt = cached_temp_sweep(t_min, t_max, n_steps, PARAMS_KEY)

        c1, c2, c3 = st.columns(3)
        c1.metric("Min Delay",  f"{df_pvt['propagation_delay_ns'].min():.5f} ns",
                  delta=f"at {df_pvt.loc[df_pvt['propagation_delay_ns'].idxmin(),'temperature']}°C")
        c2.metric("Max Delay",  f"{df_pvt['propagation_delay_ns'].max():.5f} ns",
                  delta=f"at {df_pvt.loc[df_pvt['propagation_delay_ns'].idxmax(),'temperature']}°C")
        c3.metric("Delay Range", f"{(df_pvt['propagation_delay_ns'].max() - df_pvt['propagation_delay_ns'].min()):.5f} ns")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_td = px.line(df_pvt, x="temperature", y="propagation_delay_ns",
                             title="Delay vs Temperature (PVT Sweep)",
                             labels={"temperature": "Temperature (°C)",
                                     "propagation_delay_ns": "Delay (ns)"},
                             color_discrete_sequence=["#7F77DD"],
                             markers=True)
            # UPDATED PVT SPEC PLOT LINE
            fig_td.add_hline(y=0.032, line_dash="dash", line_color="red",
                             annotation_text="Spec limit: 0.032 ns")
            d_rng = df_pvt["propagation_delay_ns"]
            fig_td.update_yaxes(range=[d_rng.min()*0.9, d_rng.max()*1.1])
            st.plotly_chart(fig_td, use_container_width=True)

        with col_b:
            fig_tp = px.line(df_pvt, x="temperature", y="power_dissipation_uw",
                             title="Power vs Temperature (PVT Sweep)",
                             labels={"temperature": "Temperature (°C)",
                                     "power_dissipation_uw": "Power (µW)"},
                             color_discrete_sequence=["#00CC96"],
                             markers=True)
            p_rng = df_pvt["power_dissipation_uw"]
            fig_tp.update_yaxes(range=[p_rng.min()*0.9, p_rng.max()*1.1])
            st.plotly_chart(fig_tp, use_container_width=True)

        # Pass/fail at each temperature
        st.subheader("Pass / Fail by Temperature")
        df_pvt["status"] = df_pvt["is_valid"].map({True: "✓ PASS", False: "✗ FAIL"})
        st.dataframe(
            df_pvt[["temperature", "propagation_delay_ns",
                    "power_dissipation_uw", "sim_source", "status"]].rename(
                columns={"temperature": "Temp (°C)",
                         "propagation_delay_ns": "Delay (ns)",
                         "power_dissipation_uw": "Power (µW)",
                         "sim_source": "Source",
                         "status": "Status"}),
            hide_index=True, use_container_width=True
        )

        csv = df_pvt.to_csv(index=False).encode()
        st.download_button("⬇ Download PVT Data", csv, "pvt_sweep.csv", "text/csv")
    else:
        st.info("Set temperature range above and click **Run Temperature Sweep**.")


# ══════════════════════════════════════════════════════
# PAGE 8 — MULTI-GATE ANALYSIS
# ══════════════════════════════════════════════════════
elif "Multi-Gate" in page:
    section("⚙️ Multi-Gate Analysis",
            "Compare CMOS Inverter vs NAND2 vs NOR2 performance")

    st.markdown("""
    **Why multi-gate matters:**
    - NAND2 NMOS are in **series** → series path slower, need wider W
    - NOR2 PMOS are in **series** → PMOS already slow, series makes it worse
    - Inverter is the baseline — fastest and most power-efficient per function
    """)

    if st.button("▶ Run Gate Comparison (3 SPICE sims)"):
        with st.spinner("Running Inverter, NAND2, NOR2 simulations..."):
            gate_df = cached_gate_comparison(PARAMS_KEY)

        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], gate_df.iterrows()):
            col.metric(row["gate"].upper(),
                       f"{row['propagation_delay_ns']:.5f} ns",
                       delta=row["sim_source"].replace("_", " "))

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_g1 = px.bar(gate_df, x="gate", y="propagation_delay_ns",
                            color="gate", title="Propagation Delay by Gate Type",
                            color_discrete_sequence=["#7F77DD", "#00CC96", "#EF553B"],
                            labels={"gate": "Gate Type",
                                    "propagation_delay_ns": "Delay (ns)"})
            fig_g1.update_layout(showlegend=False)
            st.plotly_chart(fig_g1, use_container_width=True)

        with col_b:
            fig_g2 = px.bar(gate_df, x="gate", y="power_dissipation_uw",
                            color="gate", title="Power Dissipation by Gate Type",
                            color_discrete_sequence=["#7F77DD", "#00CC96", "#EF553B"],
                            labels={"gate": "Gate Type",
                                    "power_dissipation_uw": "Power (µW)"})
            fig_g2.update_layout(showlegend=False)
            st.plotly_chart(fig_g2, use_container_width=True)

        # Normalized comparison
        st.subheader("Normalized Performance (Inverter = 1.0×)")
        if gate_df.iloc[0]["propagation_delay_ns"] > 0:
            base_d = gate_df.iloc[0]["propagation_delay_ns"]
            base_p = gate_df.iloc[0]["power_dissipation_uw"]
            gate_df["norm_delay"] = gate_df["propagation_delay_ns"] / base_d
            gate_df["norm_power"] = gate_df["power_dissipation_uw"]  / base_p
            fig_norm = px.bar(gate_df.melt(id_vars="gate",
                                           value_vars=["norm_delay", "norm_power"],
                                           var_name="metric", value_name="normalized"),
                              x="gate", y="normalized", color="metric",
                              barmode="group", title="Normalized Delay and Power",
                              labels={"normalized": "Normalized (1.0 = Inverter baseline)"})
            fig_norm.add_hline(y=1.0, line_dash="dash", line_color="white",
                               annotation_text="Inverter baseline")
            st.plotly_chart(fig_norm, use_container_width=True)

        st.dataframe(gate_df[["gate", "propagation_delay_ns",
                               "power_dissipation_uw", "sim_source", "is_valid"]].rename(
            columns={"gate": "Gate", "propagation_delay_ns": "Delay (ns)",
                     "power_dissipation_uw": "Power (µW)",
                     "sim_source": "Source", "is_valid": "Passes Yield"}),
            hide_index=True, use_container_width=True)
    else:
        st.info("Click **Run Gate Comparison** to simulate all three gates.")


# ══════════════════════════════════════════════════════
# PAGE 9 — REPORT EXPORT
# ══════════════════════════════════════════════════════
elif "Report" in page:
    section("📄 Report Export",
            "One-click export of all 10 module results into a professional PDF")

    st.subheader("Run All Modules Before Exporting")
    st.info(
        "For the most complete report, run these pages first: "
        "**Monte Carlo → Yield Oracle → Corner Analysis → Wafer Map → "
        "PVT Analysis → Multi-Gate → Waveform Viewer → Aging Analysis → "
        "Power Breakdown → Tornado Chart** — then come back here and export."
    )

    st.markdown("---")
    if st.button("▶ Generate Complete PDF Report", type="primary"):
        with st.spinner("Collecting results from all modules and building PDF..."):

            import numpy as np, pandas as pd

            mc_stats_out   = None
            cpk_out        = None
            corner_out     = None
            wafer_out      = None
            oracle_out     = None
            power_out      = None
            aging_out      = None
            pvt_out        = None
            gate_out       = None
            wf_out         = None
            tornado_out    = None

            # ── Module 1: Monte Carlo ──────────────────────────
            if os.path.exists("data/processed/monte_carlo_results.csv"):
                from src.simulation.monte_carlo import (
                    compute_yield_statistics, compute_process_capability)
                df_mc       = pd.read_csv("data/processed/monte_carlo_results.csv")
                mc_stats_out = compute_yield_statistics(df_mc)
                cpk_out      = compute_process_capability(df_mc)
            else:
                st.warning("Monte Carlo data not found — run Monte Carlo page first.")

            # ── Module 2: Oracle (same chips as MC) ────────────
            dm, ds, pm, ps = load_models()
            if dm is not None and os.path.exists("data/processed/monte_carlo_results.csv"):
                from src.ml.surrogate_model import FEATURE_COLS
                df_mc_raw = pd.read_csv("data/processed/monte_carlo_results.csv")
                X         = df_mc_raw[FEATURE_COLS].values.astype(np.float64)
                pred_d    = dm.predict(ds.transform(X)).astype(np.float64)
                pred_p    = pm.predict(ps.transform(X)).astype(np.float64)
                passes    = (pred_d < 0.022) & (pred_p < 105.0)
                n_total   = len(X); n_pass = int(np.sum(passes))
                oracle_out = {
                    "yield_percent":  round(100.0 * n_pass / n_total, 2),
                    "n_total": n_total, "n_pass": n_pass, "n_fail": n_total - n_pass,
                    "mean_delay_ns": round(float(pred_d.mean()), 6),
                    "std_delay_ns":  round(float(pred_d.std()),  6),
                    "mean_power_uw": round(float(pred_p.mean()), 4),
                    "std_power_uw":  round(float(pred_p.std()),  4),
                }

            # ── Module 3: Corner Analysis ──────────────────────
            from src.simulation.spice_engine import run_corner_analysis
            with st.spinner("Running 3 corner simulations..."):
                corner_out = run_corner_analysis()

            # ── Module 4: Wafer ────────────────────────────────
            if os.path.exists("data/processed/wafer_simulation.csv"):
                from src.wafer.spatial_model import compute_wafer_yield_map
                df_w      = pd.read_csv("data/processed/wafer_simulation.csv")
                wafer_out = compute_wafer_yield_map(df_w)
            else:
                st.warning("Wafer data not found — run Wafer Map page first.")

            # ── Module 5: PVT Sweep ────────────────────────────
            from src.simulation.spice_engine import run_single_simulation
            pvt_temps = [-40, -20, 0, 27, 50, 75, 100, 125]
            pvt_rows  = []
            with st.spinner("Running PVT temperature sweep..."):
                for t in pvt_temps:
                    p = CURRENT_PARAMS.copy(); p["temperature"] = t
                    r = run_single_simulation(p); r["temperature"] = t
                    pvt_rows.append(r)
            pvt_out = pvt_rows

            # ── Module 6: Multi-Gate ───────────────────────────
            from src.simulation.spice_engine import run_gate_simulation
            gate_rows = []
            with st.spinner("Running gate comparison simulations..."):
                for gate in ["inverter", "nand2", "nor2"]:
                    g = run_gate_simulation(gate, CURRENT_PARAMS.copy())
                    gate_rows.append(g)
            gate_out = gate_rows

            # ── Module 7: Waveform ─────────────────────────────
            r_wf = run_single_simulation(CURRENT_PARAMS)
            wf_out = {
                "tpd_ns":      r_wf["propagation_delay_ns"],
                "power_uw":    r_wf["power_dissipation_uw"],
                "source":      r_wf["sim_source"],
                "temperature": CURRENT_PARAMS.get("temperature", 27),
                "vdd_half":    round(CURRENT_PARAMS.get("vdd", 3.3) / 2, 2),
            }

            # ── Module 8: Aging ────────────────────────────────
            from src.simulation.aging import run_aging_sweep
            with st.spinner("Running aging simulation..."):
                aging_df_out = run_aging_sweep(
                    [0, 1, 3, 5, 10], CURRENT_PARAMS.copy())
            aging_out = aging_df_out

            # ── Module 9: Power Breakdown ──────────────────────
            from src.simulation.spice_engine import compute_power_breakdown
            power_out = compute_power_breakdown(CURRENT_PARAMS)

            # ── Module 10: Tornado Chart ───────────────────────
            if os.path.exists("data/processed/monte_carlo_results.csv"):
                from src.simulation.monte_carlo import compute_sensitivity_tornado
                df_mc2      = pd.read_csv("data/processed/monte_carlo_results.csv")
                with st.spinner("Computing sensitivity tornado..."):
                    tornado_df  = compute_sensitivity_tornado(df_mc2)
                tornado_out = tornado_df.to_dict("records")

            # ── Generate PDF ───────────────────────────────────
            try:
                from src.reports.report_generator import generate_pdf_report
                path = generate_pdf_report(
                    monte_carlo_stats = mc_stats_out,
                    cpk_stats         = cpk_out,
                    corner_results    = corner_out,
                    wafer_stats       = wafer_out,
                    oracle_stats      = oracle_out,
                    power_breakdown   = power_out,
                    aging_df          = aging_out,
                    pvt_data          = pvt_out,
                    gate_data         = gate_out,
                    waveform_stats    = wf_out,
                    tornado_data      = tornado_out,
                    current_params    = CURRENT_PARAMS,
                    output_path       = "reports/PyYield_Report.pdf"
                )
                st.success(f"✓ Complete {pdf_pages(path)}-page PDF generated!")
                with open(path, "rb") as f:
                    st.download_button("⬇ Download Complete PDF Report",
                                       f.read(), "PyYield_Report.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.markdown("---")
    st.subheader("Quick CSV Downloads")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("data/processed/monte_carlo_results.csv"):
            with open("data/processed/monte_carlo_results.csv", "rb") as f:
                st.download_button("⬇ Monte Carlo CSV", f.read(),
                                   "monte_carlo_results.csv", "text/csv")
    with c2:
        if os.path.exists("data/processed/wafer_simulation.csv"):
            with open("data/processed/wafer_simulation.csv", "rb") as f:
                st.download_button("⬇ Wafer Map CSV", f.read(),
                                   "wafer_simulation.csv", "text/csv")
# ══════════════════════════════════════════════════════
# PAGE 10 — AGING ANALYSIS
# ══════════════════════════════════════════════════════
elif "Aging" in page:
    section("⏳ Chip Aging Analysis",
            "NBTI + HCI degradation over operational lifetime using Black's equation")

    st.markdown("""
    **Aging mechanisms modeled:**
    - **NBTI** (Negative Bias Temperature Instability) — degrades PMOS, increases |Vtp|
    - **HCI** (Hot Carrier Injection) — degrades NMOS near drain, increases Vtn
    - **Electromigration** — metal wire failure from sustained high current density
    """)

    c1, c2 = st.columns(2)
    max_years      = c1.slider("Simulation Lifetime (years)", 1, 15, 10, 1)
    activity_factor = c2.slider("Activity Factor (switching %)", 0.022, 1.0, 0.5, 0.05)

    if st.button("▶ Run Aging Simulation"):
        from src.simulation.aging import (run_aging_sweep, compute_em_risk,
                                          compute_nbti_vth_shift, compute_hci_vth_shift)

        time_points = [0, 0.5, 1, 2, 3, 5, 7, max_years]
        time_points = sorted(set(time_points))

        with st.spinner("Simulating aging across lifetime..."):
            df_age = run_aging_sweep(time_points, CURRENT_PARAMS, activity_factor)
            em     = compute_em_risk(CURRENT_PARAMS)

        # EM risk banner
        if not em["em_safe"]:
            st.error(f"⚠️ Electromigration Risk Detected — "
                     f"J = {em['current_density_Apm2']:.2e} A/m² exceeds limit")
        else:
            st.success(f"✓ EM Safe — MTTF = {em['em_mttf_years']:.1f} years | "
                       f"J = {em['current_density_Apm2']:.2e} A/m²")

        n_pass_t0  = df_age.iloc[0]["is_valid"]
        n_pass_end = df_age.iloc[-1]["is_valid"]
        delay_t0   = df_age.iloc[0]["propagation_delay_ns"]
        delay_end  = df_age.iloc[-1]["propagation_delay_ns"]
        delay_deg  = round(100*(delay_end - delay_t0)/max(delay_t0, 1e-10), 2)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delay at t=0",      f"{delay_t0:.5f} ns")
        c2.metric(f"Delay at t={max_years}yr", f"{delay_end:.5f} ns",
                  delta=f"{delay_deg:+.1f}%")
        c3.metric("EM MTTF",           f"{em['em_mttf_years']:.1f} yr")
        c4.metric("Passes at End",      "✓ YES" if n_pass_end else "✗ NO")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_ad = px.line(df_age, x="years", y="propagation_delay_ns",
                             title="Delay Degradation Over Time",
                             labels={"years": "Operational Years",
                                     "propagation_delay_ns": "Delay (ns)"},
                             color_discrete_sequence=["#7F77DD"], markers=True)
            fig_ad.add_hline(y=1.0, line_dash="dash", line_color="red",
                             annotation_text="Spec limit")
            d_vals = df_age["propagation_delay_ns"]
            fig_ad.update_yaxes(range=[d_vals.min()*0.95, max(d_vals.max()*1.05, 0.022)])
            st.plotly_chart(fig_ad, use_container_width=True)

        with col_b:
            fig_vth = px.line(df_age, x="years",
                              y=["dvth_hci_mv", "dvth_nbti_mv"],
                              title="Threshold Voltage Shift (mV)",
                              labels={"years": "Operational Years",
                                      "value": "ΔVth (mV)",
                                      "variable": "Mechanism"},
                              markers=True,
                              color_discrete_sequence=["#EF553B", "#00CC96"])
            fig_vth.update_layout(legend_title="Mechanism")
            st.plotly_chart(fig_vth, use_container_width=True)

        st.subheader("Pass / Fail Status Over Lifetime")
        df_age["status"] = df_age["is_valid"].map({True: "✓ PASS", False: "✗ FAIL"})
        st.dataframe(
            df_age[["years","dvth_hci_mv","dvth_nbti_mv",
                    "vth_n_aged","vth_p_aged",
                    "propagation_delay_ns","is_valid","status"]].rename(
                columns={"years": "Year", "dvth_hci_mv": "ΔVtn HCI (mV)",
                         "dvth_nbti_mv": "ΔVtp NBTI (mV)",
                         "vth_n_aged": "Vtn (V)", "vth_p_aged": "Vtp (V)",
                         "propagation_delay_ns": "Delay (ns)",
                         "is_valid": "Valid", "status": "Status"}),
            hide_index=True, use_container_width=True)

        csv = df_age.to_csv(index=False).encode()
        st.download_button("⬇ Download Aging Data", csv, "aging_data.csv", "text/csv")
    else:
        st.info("Click **Run Aging Simulation** to model chip degradation over time.")


# ══════════════════════════════════════════════════════
# PAGE 11 — POWER BREAKDOWN
# ══════════════════════════════════════════════════════
elif "Power" in page:
    section("⚡ Power Breakdown Analysis",
            "Detailed split of dynamic, leakage, gate-oxide, and short-circuit power")

    from src.simulation.spice_engine import compute_power_breakdown, derive_cox, compute_parasitic_rc

    pb   = compute_power_breakdown(CURRENT_PARAMS)
    cox  = derive_cox(CURRENT_PARAMS)
    R_w, C_w = compute_parasitic_rc(CURRENT_PARAMS)
    tox  = CURRENT_PARAMS.get("tox", 4e-9)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dynamic Power",      f"{pb['p_dynamic_uw']:.4f} µW")
    c2.metric("Leakage Power",      f"{pb['p_leakage_uw']:.4f} µW")
    c3.metric("Gate Oxide Leak",    f"{pb['p_gate_oxide_uw']:.6f} µW")
    c4.metric("Short-Circuit",      f"{pb['p_short_circuit_uw']:.6f} µW")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["Dynamic", "Subthreshold Leakage",
                  "Gate Oxide Tunneling", "Short-Circuit"]
        values = [pb["p_dynamic_uw"], pb["p_leakage_uw"],
                  pb["p_gate_oxide_uw"], pb["p_short_circuit_uw"]]
        colors = ["#7F77DD", "#EF553B", "#00CC96", "#FFA15A"]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.4,
            marker_colors=colors,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title="Power Component Breakdown",
                              height=380, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Physical Parameters")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Oxide Thickness (tox) | {tox*1e9:.2f} nm |
        | Cox (derived) | {cox:.4e} F/m² |
        | Wire Resistance | {R_w:.2f} Ω |
        | Wire Capacitance | {C_w:.4e} F |
        | Temperature | {CURRENT_PARAMS.get('temperature', 27)} °C |
        | Supply Voltage | {CURRENT_PARAMS['vdd']:.2f} V |
        """)

        st.markdown("#### tox Effect on Power")
        st.caption("Thinner oxide → more Cox → more gate oxide tunneling leakage")

    # tox sweep on power
    st.markdown("---")
    st.subheader("Oxide Thickness vs Power Components")
    tox_vals = np.linspace(2e-9, 8e-9, 30)
    rows = []
    for t in tox_vals:
        p = CURRENT_PARAMS.copy(); p["tox"] = t
        pb2 = compute_power_breakdown(p)
        rows.append({"tox_nm": round(t*1e9, 2), **pb2})
    tox_df = pd.DataFrame(rows)

    fig_tox = go.Figure()
    fig_tox.add_trace(go.Scatter(x=tox_df["tox_nm"], y=tox_df["p_dynamic_uw"],
                                 name="Dynamic", line=dict(color="#7F77DD")))
    fig_tox.add_trace(go.Scatter(x=tox_df["tox_nm"], y=tox_df["p_leakage_uw"],
                                 name="Leakage", line=dict(color="#EF553B")))
    fig_tox.add_trace(go.Scatter(x=tox_df["tox_nm"], y=tox_df["p_gate_oxide_uw"],
                                 name="Gate Oxide", line=dict(color="#00CC96")))
    fig_tox.update_layout(
        title="Power Components vs Oxide Thickness",
        xaxis_title="tox (nm)",
        yaxis_title="Power (µW)",
        height=380,
    )
    st.plotly_chart(fig_tox, use_container_width=True)
    st.caption("Gate oxide tunneling leakage increases exponentially as tox decreases — "
               "critical trade-off in sub-5nm technology nodes.")


# ══════════════════════════════════════════════════════
# PAGE 12 — SENSITIVITY TORNADO
# ══════════════════════════════════════════════════════
elif "Tornado" in page:
    section("🌪️ Sensitivity Tornado Chart",
            "Ranks each process parameter by its impact on propagation delay")

    st.markdown("""
    The tornado chart sweeps each parameter from its **10th to 90th percentile**
    while holding all others at nominal. The width of each bar shows how much
    that parameter alone can shift the delay — wider = more critical to control.
    """)

    if st.button("▶ Generate Tornado Chart"):
        csv_path = "data/processed/monte_carlo_results.csv"
        if not os.path.exists(csv_path):
            st.warning("Run Monte Carlo first to generate the data file.")
        else:
            with st.spinner("Running sensitivity sweep (12 SPICE sims)..."):
                from src.simulation.monte_carlo import compute_sensitivity_tornado
                df_mc  = pd.read_csv(csv_path)
                tornado_df = compute_sensitivity_tornado(df_mc)

            fig_t = go.Figure()
            colors = ["#7F77DD","#00CC96","#EF553B","#FFA15A","#AB63FA","#19D3F3"]

            for i, row in tornado_df.iterrows():
                color = colors[i % len(colors)]
                lo    = min(row["delay_at_p10"], row["delay_at_p90"])
                hi    = max(row["delay_at_p10"], row["delay_at_p90"])
                fig_t.add_trace(go.Bar(
                    y=[row["parameter"]],
                    x=[hi - lo],
                    base=[lo],
                    orientation="h",
                    name=row["parameter"],
                    marker_color=color,
                    text=f"Δ={row['delta_delay_ns']:.5f} ns",
                    textposition="inside",
                ))

            nominal_delay = CURRENT_PARAMS.copy()
            from src.simulation.spice_engine import run_single_simulation
            nd = run_single_simulation(nominal_delay)["propagation_delay_ns"]

            fig_t.add_vline(x=nd, line_dash="dash", line_color="white",
                            annotation_text=f"Nominal: {nd:.4f} ns")
            fig_t.update_layout(
                title="Tornado Chart — Parameter Sensitivity to Delay",
                xaxis_title="Propagation Delay (ns)",
                yaxis_title="Parameter",
                height=420,
                showlegend=False,
                barmode="overlay",
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.subheader("Sensitivity Ranking")
            st.dataframe(
                tornado_df[["parameter","delay_at_p10","delay_at_p90",
                             "delta_delay_ns","impact_pct"]].rename(
                    columns={"parameter": "Parameter",
                             "delay_at_p10": "Delay @ P10 (ns)",
                             "delay_at_p90": "Delay @ P90 (ns)",
                             "delta_delay_ns": "Δ Delay (ns)",
                             "impact_pct": "Impact (%)"}),
                hide_index=True, use_container_width=True)

            st.caption("**Read this:** Vth_n and Length dominate delay — "
                       "foundry must control these tightest. "
                       "Width and Capacitance have less impact.")
    else:
        st.info("Click **Generate Tornado Chart** to run the sensitivity analysis.")