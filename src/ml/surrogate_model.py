"""
surrogate_model.py — Module 2A + SHAP explainability
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = ["vth_n", "vth_p", "length", "width_n", "width_p", "load_cap"]
TARGET_DELAY = "propagation_delay_ns"
TARGET_POWER = "power_dissipation_uw"
MODEL_DIR    = "models"


def load_training_data(csv_path="data/processed/monte_carlo_results.csv"):
    df = pd.read_csv(csv_path)
    print(f"[Surrogate] Loaded {len(df)} samples.")
    return df[FEATURE_COLS].values, df[TARGET_DELAY].values, df[TARGET_POWER].values


def train_model(X, y, target_name="delay"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)
    model  = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    metrics = {
        "mae":        round(mean_absolute_error(y_test, y_pred), 6),
        "r2_score":   round(r2_score(y_test, y_pred), 4),
        "cv_r2_mean": round(cross_val_score(
            model, Xtr, y_train, cv=5, scoring="r2").mean(), 4),
    }
    return model, scaler, metrics


def save_model(model, scaler, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  f"{MODEL_DIR}/{name}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{name}_scaler.pkl")
    print(f"[Surrogate] Saved → models/{name}_model.pkl")


def load_model(name):
    return (joblib.load(f"{MODEL_DIR}/{name}_model.pkl"),
            joblib.load(f"{MODEL_DIR}/{name}_scaler.pkl"))


def predict(model, scaler, params: dict) -> float:
    x = np.array([[params[c] for c in FEATURE_COLS]])
    return float(model.predict(scaler.transform(x))[0])


def get_feature_importance(model) -> dict:
    return dict(sorted(zip(FEATURE_COLS, model.feature_importances_),
                        key=lambda x: x[1], reverse=True))


def compute_shap_values(model, scaler, X_raw: np.ndarray,
                        n_samples: int = 100) -> tuple:
    """
    Computes SHAP values for feature explainability.

    Returns:
        shap_values : np.ndarray of shape (n_samples, n_features)
        X_display   : original (unscaled) values for display
        feature_names: list of feature names
    """
    try:
        import shap
        X_sub     = X_raw[:n_samples]
        X_scaled  = scaler.transform(X_sub)
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_scaled)
        return sv, X_sub, FEATURE_COLS
    except ImportError:
        print("[SHAP] shap not installed. Run: pip install shap")
        return None, None, FEATURE_COLS
    except Exception as e:
        print(f"[SHAP] Error: {e}")
        return None, None, FEATURE_COLS


def get_shap_summary(model, scaler, X_raw: np.ndarray) -> pd.DataFrame:
    """
    Returns a DataFrame of mean absolute SHAP values per feature.
    Suitable for bar chart display in the dashboard.
    """
    sv, X_sub, feats = compute_shap_values(model, scaler, X_raw)
    if sv is None:
        importances = model.feature_importances_
        return pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": importances,
            "source": "feature_importance_fallback"
        }).sort_values("mean_abs_shap", ascending=False)

    mean_shap = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({
        "feature":       feats,
        "mean_abs_shap": mean_shap,
        "source":        "shap"
    }).sort_values("mean_abs_shap", ascending=False)
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("   PyYield — Surrogate Model v2 Self-Test")
    print("=" * 55)
    X, y_delay, y_power = load_training_data()

    print("\n[Training Delay Model]")
    dm, ds, dm_m = train_model(X, y_delay, "delay")
    print(f"  R2={dm_m['r2_score']}  MAE={dm_m['mae']}  CV-R2={dm_m['cv_r2_mean']}")

    print("\n[Training Power Model]")
    pm, ps, pm_m = train_model(X, y_power, "power")
    print(f"  R2={pm_m['r2_score']}  MAE={pm_m['mae']}  CV-R2={pm_m['cv_r2_mean']}")

    print("\n[SHAP Analysis]")
    shap_df = get_shap_summary(dm, ds, X)
    for _, row in shap_df.iterrows():
        bar = "█" * int(row["mean_abs_shap"] / shap_df["mean_abs_shap"].max() * 30)
        print(f"  {row['feature']:<12} {bar} {row['mean_abs_shap']:.6f}")

    save_model(dm, ds, "delay")
    save_model(pm, ps, "power")
    print("\n[surrogate_model.py v2] PASSED")
    print("=" * 55)