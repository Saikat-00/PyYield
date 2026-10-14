"""
setup_cloud.py — Runs on first launch to train ML models
if they don't already exist (for cloud deployment).
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def ensure_setup():
    models_ready = (
        os.path.exists("models/delay_model.pkl") and
        os.path.exists("models/power_model.pkl")
    )
    data_ready = os.path.exists("data/processed/monte_carlo_results.csv")

    if models_ready and data_ready:
        return   # Nothing to do

    print("[Setup] First launch detected — running initial setup...")
    os.makedirs("models",           exist_ok=True)
    os.makedirs("data/processed",   exist_ok=True)
    os.makedirs("data/raw",         exist_ok=True)
    os.makedirs("reports",          exist_ok=True)

    if not data_ready:
        print("[Setup] Running Monte Carlo simulation (300 samples)...")
        from src.simulation.monte_carlo import run_monte_carlo, save_results
        df = run_monte_carlo(n_samples=300, method="lhs", seed=42)
        save_results(df)
        print("[Setup] Monte Carlo complete.")

    if not models_ready:
        print("[Setup] Training ML surrogate models...")
        from src.ml.surrogate_model import (
            load_training_data, train_model, save_model)
        X, y_d, y_p = load_training_data()
        dm, ds, dm_m = train_model(X, y_d, "delay")
        pm, ps, pm_m = train_model(X, y_p, "power")
        save_model(dm, ds, "delay")
        save_model(pm, ps, "power")
        print(f"[Setup] Delay model R2: {dm_m['r2_score']}")
        print(f"[Setup] Power model R2: {pm_m['r2_score']}")
        print("[Setup] Models saved.")

    print("[Setup] Setup complete.")


if __name__ == "__main__":
    ensure_setup()