"""
Chennai Traffic Congestion Analysis
=====================================
Data Science Portfolio Project
Author: Saturday-sudo
Dataset: Synthetic 2024 traffic sensor data (14 corridors, hourly granularity)

Pipeline:
  1. Data generation & EDA
  2. Feature engineering
  3. ML model training & comparison
  4. SHAP explainability
  5. Policy-ready insights
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── 1. SYNTHETIC DATA GENERATION ─────────────────────────────────────────────

np.random.seed(42)

CORRIDORS = {
    "Kathipara Junction":      {"type": "arterial",   "base": 0.82},
    "Koyambedu–Poonamallee":   {"type": "arterial",   "base": 0.79},
    "OMR Perungudi":           {"type": "it_corridor","base": 0.77},
    "Anna Salai–Gemini":       {"type": "arterial",   "base": 0.71},
    "Velachery Main Rd":       {"type": "residential","base": 0.68},
    "GST Road–Chrompet":       {"type": "arterial",   "base": 0.66},
    "Adyar Signal":            {"type": "residential","base": 0.58},
    "Porur Junction":          {"type": "arterial",   "base": 0.62},
    "ECR Sholinganallur":      {"type": "it_corridor","base": 0.60},
    "Tambaram Bypass":         {"type": "arterial",   "base": 0.55},
    "Poonamallee High Rd":     {"type": "arterial",   "base": 0.57},
    "Saidapet–Guindy":         {"type": "arterial",   "base": 0.64},
    "Thoraipakkam OMR":        {"type": "it_corridor","base": 0.59},
    "Broadway–Park Town":      {"type": "residential","base": 0.52},
}

CHENNAI_SCHOOL_HOLIDAYS = [
    (datetime(2024, 4, 13), datetime(2024, 6, 10)),   # summer
    (datetime(2024, 10, 2), datetime(2024, 10, 7)),    # Dussehra
    (datetime(2024, 12, 25), datetime(2024, 12, 31)),  # Christmas break
]

RAINY_DAYS_2024 = set(pd.date_range("2024-09-01", "2024-11-30", freq="3D").date)


def is_school_holiday(dt):
    for start, end in CHENNAI_SCHOOL_HOLIDAYS:
        if start <= dt <= end:
            return 1
    return 0


def hour_factor(hour, corridor_type):
    """AM/PM peak profile per corridor type."""
    if corridor_type == "it_corridor":
        am = np.exp(-0.5 * ((hour - 9.5) / 1.2) ** 2)
        pm = np.exp(-0.5 * ((hour - 18.5) / 1.3) ** 2)
    else:
        am = np.exp(-0.5 * ((hour - 8.5) / 1.1) ** 2)
        pm = np.exp(-0.5 * ((hour - 18.0) / 1.2) ** 2)
    return 0.3 + 0.7 * (am + pm) / (am + pm).max() if hasattr(am, "max") else 0.3 + 0.7 * max(am, pm)


def generate_dataset():
    records = []
    date_range = pd.date_range("2024-01-01", "2024-12-31", freq="h")

    for ts in date_range:
        dt = ts.to_pydatetime()
        for corridor, meta in CORRIDORS.items():
            h = dt.hour
            dow = dt.weekday()  # 0=Mon, 6=Sun
            month = dt.month
            is_holiday = is_school_holiday(dt)
            is_rain = int(dt.date() in RAINY_DAYS_2024)
            is_weekend = int(dow >= 5)

            # Hour factor
            if meta["type"] == "it_corridor":
                am_peak = np.exp(-0.5 * ((h - 9.5) / 1.2) ** 2)
                pm_peak = np.exp(-0.5 * ((h - 18.5) / 1.3) ** 2)
            else:
                am_peak = np.exp(-0.5 * ((h - 8.5) / 1.1) ** 2)
                pm_peak = np.exp(-0.5 * ((h - 18.0) / 1.2) ** 2)

            peak = max(am_peak, pm_peak)

            # Weekend/holiday reduction
            weekend_factor = 0.55 if is_weekend else 1.0
            holiday_factor = 0.70 if is_holiday else 1.0

            # Rain increases congestion
            rain_factor = 1.18 if is_rain and peak > 0.3 else 1.0

            # Month seasonality (Northeast monsoon = Oct–Dec heavier)
            month_factor = 1.0 + 0.08 * np.sin((month - 3) * np.pi / 6)

            ci = (meta["base"] * peak * weekend_factor * holiday_factor
                  * rain_factor * month_factor + np.random.normal(0, 0.04))
            ci = float(np.clip(ci, 0.05, 1.0))

            volume = int(ci * 9500 + np.random.normal(0, 300))
            volume = max(200, volume)

            records.append({
                "timestamp": ts,
                "corridor": corridor,
                "corridor_type": meta["type"],
                "hour": h,
                "day_of_week": dow,
                "month": month,
                "is_weekend": is_weekend,
                "is_school_holiday": is_holiday,
                "is_rain": is_rain,
                "congestion_index": round(ci, 4),
                "volume_veh_hr": volume,
            })

    return pd.DataFrame(records)


# ─── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

def engineer_features(df):
    df = df.sort_values(["corridor", "timestamp"]).reset_index(drop=True)
    grp = df.groupby("corridor")

    # Lag features per corridor
    df["ci_lag1"]  = grp["congestion_index"].shift(1)
    df["ci_lag6"]  = grp["congestion_index"].shift(6)
    df["ci_lag24"] = grp["congestion_index"].shift(24)

    # Rolling mean
    df["ci_roll3"]  = grp["congestion_index"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["ci_roll24"] = grp["congestion_index"].transform(lambda x: x.shift(1).rolling(24).mean())

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Corridor type encoding
    df["type_arterial"]   = (df["corridor_type"] == "arterial").astype(int)
    df["type_it"]         = (df["corridor_type"] == "it_corridor").astype(int)
    df["type_residential"] = (df["corridor_type"] == "residential").astype(int)

    return df.dropna().reset_index(drop=True)


# ─── 3. MODEL TRAINING ─────────────────────────────────────────────────────────

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_school_holiday", "is_rain",
    "type_arterial", "type_it", "type_residential",
    "ci_lag1", "ci_lag6", "ci_lag24",
    "ci_roll3", "ci_roll24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

TARGET = "congestion_index"


def train_and_evaluate(df):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    X = df[FEATURES]
    y = df[TARGET]

    # Time-based split: train on Jan–Sep, test on Oct–Dec
    split_date = pd.Timestamp("2024-10-01")
    mask = df["timestamp"] >= split_date
    X_train, X_test = X[~mask], X[mask]
    y_train, y_test = y[~mask], y[mask]

    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2   = r2_score(y_test, preds)
        results[name] = {"RMSE": round(rmse, 4), "R²": round(r2, 4)}
        fitted[name] = model
        print(f"  {name:22s} | RMSE={rmse:.4f}  R²={r2:.4f}")

    return fitted, results, X_test, y_test


# ─── 4. SHAP EXPLAINABILITY ────────────────────────────────────────────────────

def shap_analysis(gbm_model, X_sample):
    """
    Compute SHAP values for GBM. Requires: pip install shap
    Returns mean absolute SHAP per feature (feature importance).
    """
    try:
        import shap
        explainer = shap.TreeExplainer(gbm_model)
        shap_values = explainer.shap_values(X_sample)
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=FEATURES
        ).sort_values(ascending=False)
        return importance
    except ImportError:
        print("  [shap not installed — skipping SHAP analysis]")
        # Fallback: sklearn feature_importances_
        imp = pd.Series(gbm_model.feature_importances_, index=FEATURES)
        return imp.sort_values(ascending=False)


# ─── 5. POLICY INSIGHTS ────────────────────────────────────────────────────────

def compute_peak_corridor_stats(df):
    stats = (
        df[df["hour"].between(8, 10) | df["hour"].between(17, 20)]
        .groupby("corridor")["congestion_index"]
        .agg(["mean", "max", "std"])
        .round(3)
        .rename(columns={"mean": "avg_peak_ci", "max": "max_ci", "std": "variability"})
        .sort_values("avg_peak_ci", ascending=False)
    )
    stats["severity"] = pd.cut(
        stats["avg_peak_ci"],
        bins=[0, 0.60, 0.72, 0.80, 1.0],
        labels=["Moderate", "High", "Critical", "Extreme"]
    )
    return stats


def rain_impact_analysis(df):
    agg = df.groupby("is_rain")["congestion_index"].mean()
    lift = (agg[1] - agg[0]) / agg[0] * 100
    return round(lift, 1)


def weekday_vs_weekend(df):
    agg = df.groupby("is_weekend")["congestion_index"].mean()
    return {
        "weekday_avg_ci": round(agg[0], 3),
        "weekend_avg_ci": round(agg[1], 3),
        "reduction_pct": round((agg[0] - agg[1]) / agg[0] * 100, 1)
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CHENNAI TRAFFIC CONGESTION ANALYSIS — 2024")
    print("=" * 60)

    print("\n[1/5] Generating synthetic dataset...")
    df = generate_dataset()
    print(f"      Records: {len(df):,}  |  Corridors: {df['corridor'].nunique()}")
    print(f"      Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    print(f"      Training features: {len(FEATURES)}  |  Final rows: {len(df):,}")

    print("\n[3/5] Training & evaluating models...")
    fitted_models, results, X_test, y_test = train_and_evaluate(df)

    print("\n[4/5] SHAP feature importance (GBM)...")
    sample = X_test.sample(2000, random_state=42)
    importance = shap_analysis(fitted_models["Gradient Boosting"], sample)
    print("      Top-5 features:")
    for feat, val in importance.head(5).items():
        print(f"        {feat:20s} {val:.4f}")

    print("\n[5/5] Policy insights...")
    peak_stats = compute_peak_corridor_stats(df)
    print("\n  Peak-hour corridor severity:")
    print(peak_stats[["avg_peak_ci", "severity"]].to_string())

    rain_lift = rain_impact_analysis(df)
    print(f"\n  Rain day congestion lift: +{rain_lift}%")

    ww = weekday_vs_weekend(df)
    print(f"  Weekday avg CI : {ww['weekday_avg_ci']}")
    print(f"  Weekend avg CI : {ww['weekend_avg_ci']}")
    print(f"  Weekend reduction: {ww['reduction_pct']}%")

    # Save processed dataset
    df.to_csv("data/chennai_traffic_2024_processed.csv", index=False)
    print("\n  Processed dataset saved → data/chennai_traffic_2024_processed.csv")
    print("\nDone.")
