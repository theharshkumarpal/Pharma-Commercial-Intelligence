"""
TASKS 29–34: Forecasting
  Task 29 — Choose forecasting target (major diabetes therapeutic classes)
  Task 30 — Create time series
  Task 31 — Baseline naive forecast
  Task 32 — Train forecasting models (Naive, Exponential Smoothing, ARIMA)
  Task 33 — Backtest with train/validation/test split
  Task 34 — Generate 2025/2026 scenarios (base, upside, downside)
"""

import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

warnings.filterwarnings("ignore")


# ===================================================================
# TASK 29 & 30 — Create time series for major therapeutic classes
# ===================================================================
def task29_30_create_time_series() -> dict:
    print(f"\n{'='*60}")
    print("TASK 29-30 — Creating time series for major therapeutic classes")
    print(f"{'='*60}")

    therapy_yearly = pd.read_csv("outputs/therapy_yearly.csv")

    # Focus on major classes only
    major_classes = [
        "GLP-1 receptor agonists",
        "SGLT2 inhibitors",
        "DPP-4 inhibitors",
        "Insulin",
        "Biguanides",
        "Sulfonylureas",
    ]

    time_series = {}
    for cls in major_classes:
        subset = therapy_yearly[therapy_yearly["diabetes_class"] == cls].sort_values("year")
        if len(subset) >= 4:
            ts = subset.set_index("year")["class_fills"]
            time_series[cls] = ts
            print(f"\n  {cls}:")
            for yr, val in ts.items():
                print(f"    {yr}: {val:,.0f} fills")

    print(f"\n  Forecasting target: Annual standardized 30-day fills")
    print(f"  Classes to forecast: {len(time_series)}")
    return time_series


# ===================================================================
# TASK 31 — Naive baseline
# ===================================================================
def naive_forecast(ts: pd.Series, horizon: int = 1) -> float:
    """Naive forecast: next year = latest year's value."""
    return float(ts.iloc[-1])


# ===================================================================
# TASK 32 — Train forecasting models
# ===================================================================
def train_models(ts: pd.Series, class_name: str) -> dict:
    """Train Naive, Exponential Smoothing, and ARIMA on the series."""
    results = {}
    values = ts.values.astype(float)
    n = len(values)

    # Model 1: Naive
    results["Naive"] = {
        "model": "Naive",
        "description": "Next year = latest year value",
    }

    # Model 2: Exponential Smoothing
    try:
        if n >= 4:
            es_model = ExponentialSmoothing(
                values, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            results["ExpSmoothing"] = {
                "model": "Exponential Smoothing (additive trend)",
                "fitted": es_model,
            }
    except Exception as e:
        print(f"    ExpSmoothing failed for {class_name}: {e}")

    # Model 3: ARIMA
    try:
        arima_model = auto_arima(
            values, seasonal=False, suppress_warnings=True,
            max_p=3, max_q=3, max_d=2,
            stepwise=True, error_action="ignore"
        )
        results["ARIMA"] = {
            "model": f"ARIMA{arima_model.order}",
            "fitted": arima_model,
        }
    except Exception as e:
        print(f"    ARIMA failed for {class_name}: {e}")

    return results


# ===================================================================
# TASK 33 — Backtest
# ===================================================================
def backtest(ts: pd.Series, class_name: str) -> dict:
    """
    Train: 2019-2022, Validation: 2023, Final evaluation: 2024
    """
    years = ts.index.tolist()
    values = ts.values.astype(float)

    # We need at least 6 data points (2019-2024)
    if len(values) < 6:
        print(f"    Skipping backtest for {class_name}: insufficient data ({len(values)} points)")
        return {}

    # Split
    train = values[:4]    # 2019-2022
    val = values[4]       # 2023
    test = values[5]      # 2024

    metrics = {}

    # --- Naive ---
    naive_pred_val = train[-1]
    naive_pred_test = val  # for test, naive uses validation year
    metrics["Naive"] = {
        "val_pred": float(naive_pred_val),
        "val_actual": float(val),
        "val_ae": abs(naive_pred_val - val),
        "test_pred": float(naive_pred_test),
        "test_actual": float(test),
        "test_ae": abs(naive_pred_test - test),
    }

    # --- Exponential Smoothing ---
    try:
        es = ExponentialSmoothing(
            train, trend="add", seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        es_val_pred = float(es.forecast(1)[0])
        # Refit including validation for test prediction
        es2 = ExponentialSmoothing(
            np.append(train, val), trend="add", seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)
        es_test_pred = float(es2.forecast(1)[0])
        metrics["ExpSmoothing"] = {
            "val_pred": es_val_pred,
            "val_actual": float(val),
            "val_ae": abs(es_val_pred - val),
            "test_pred": es_test_pred,
            "test_actual": float(test),
            "test_ae": abs(es_test_pred - test),
        }
    except Exception as e:
        print(f"    ExpSmoothing backtest failed: {e}")

    # --- ARIMA ---
    try:
        arima = auto_arima(train, seasonal=False, suppress_warnings=True,
                           max_p=3, max_q=3, max_d=2, stepwise=True,
                           error_action="ignore")
        arima_val_pred = float(arima.predict(1)[0])
        # Refit for test
        arima2 = auto_arima(np.append(train, val), seasonal=False,
                            suppress_warnings=True, max_p=3, max_q=3, max_d=2,
                            stepwise=True, error_action="ignore")
        arima_test_pred = float(arima2.predict(1)[0])
        metrics["ARIMA"] = {
            "val_pred": arima_val_pred,
            "val_actual": float(val),
            "val_ae": abs(arima_val_pred - val),
            "test_pred": arima_test_pred,
            "test_actual": float(test),
            "test_ae": abs(arima_test_pred - test),
        }
    except Exception as e:
        print(f"    ARIMA backtest failed: {e}")

    # Summary metrics
    for model_name, m in metrics.items():
        m["val_mape"] = abs(m["val_ae"] / m["val_actual"]) * 100 if m["val_actual"] != 0 else None
        m["test_mape"] = abs(m["test_ae"] / m["test_actual"]) * 100 if m["test_actual"] != 0 else None
        m["combined_mae"] = (m["val_ae"] + m["test_ae"]) / 2

    return metrics


# ===================================================================
# TASK 34 — Generate 2025/2026 scenarios
# ===================================================================
def task34_scenarios(ts: pd.Series, class_name: str, best_model: str,
                     all_models: dict) -> dict:
    """Generate base, upside, downside scenarios for 2025-2026."""
    values = ts.values.astype(float)
    last_val = values[-1]

    # Get best model forecast
    if best_model == "Naive":
        base_2025 = last_val
        base_2026 = last_val
    elif best_model == "ExpSmoothing" and "ExpSmoothing" in all_models:
        try:
            es = ExponentialSmoothing(
                values, trend="add", seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            fcast = es.forecast(2)
            base_2025 = float(fcast[0])
            base_2026 = float(fcast[1])
        except:
            base_2025 = last_val
            base_2026 = last_val
    elif best_model == "ARIMA" and "ARIMA" in all_models:
        try:
            arima = auto_arima(values, seasonal=False, suppress_warnings=True,
                               stepwise=True, error_action="ignore")
            fcast = arima.predict(2)
            base_2025 = float(fcast[0])
            base_2026 = float(fcast[1])
        except:
            base_2025 = last_val
            base_2026 = last_val
    else:
        base_2025 = last_val
        base_2026 = last_val

    # Calculate historical growth rate for scenario bounds
    if len(values) >= 2 and values[-2] > 0:
        recent_growth = (values[-1] - values[-2]) / values[-2]
    else:
        recent_growth = 0.05

    scenarios = {
        "base_case": {
            "2025": round(base_2025, 0),
            "2026": round(base_2026, 0),
            "method": f"Best model ({best_model}) forecast",
        },
        "upside_case": {
            "2025": round(base_2025 * (1 + abs(recent_growth) * 0.5), 0),
            "2026": round(base_2026 * (1 + abs(recent_growth) * 0.75), 0),
            "method": "Base case + accelerated growth assumption",
        },
        "downside_case": {
            "2025": round(base_2025 * (1 - abs(recent_growth) * 0.3), 0),
            "2026": round(base_2026 * (1 - abs(recent_growth) * 0.5), 0),
            "method": "Base case with growth headwinds",
        },
        "disclaimer": "These are model-derived scenarios, NOT official CMS forecasts."
    }
    return scenarios


def main():
    # TASK 29-30
    time_series = task29_30_create_time_series()

    all_forecasts = {}
    all_backtests = {}
    all_scenarios = {}

    for cls, ts in time_series.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING: {cls}")
        print(f"{'='*60}")

        # TASK 32 — Train models
        models = train_models(ts, cls)

        # TASK 33 — Backtest
        print(f"\n  TASK 33 — Backtesting {cls}")
        bt = backtest(ts, cls)
        all_backtests[cls] = bt

        if bt:
            print(f"\n  Backtest results for {cls}:")
            print(f"  {'Model':<20} {'Val MAE':>12} {'Val MAPE%':>12} {'Test MAE':>12} {'Test MAPE%':>12}")
            best_model = None
            best_mae = float("inf")
            for mname, m in bt.items():
                val_mape = f"{m['val_mape']:.1f}" if m['val_mape'] else "N/A"
                test_mape = f"{m['test_mape']:.1f}" if m['test_mape'] else "N/A"
                print(f"  {mname:<20} {m['val_ae']:>12,.0f} {val_mape:>12} {m['test_ae']:>12,.0f} {test_mape:>12}")
                if m["combined_mae"] < best_mae:
                    best_mae = m["combined_mae"]
                    best_model = mname
            print(f"\n  Best model: {best_model} (lowest combined MAE)")
        else:
            best_model = "Naive"

        # TASK 34 — Scenarios
        print(f"\n  TASK 34 — Generating scenarios for {cls}")
        scenarios = task34_scenarios(ts, cls, best_model, models)
        all_scenarios[cls] = scenarios
        for case_name, case_data in scenarios.items():
            if isinstance(case_data, dict) and "2025" in case_data:
                print(f"    {case_name}: 2025={case_data['2025']:,.0f}, 2026={case_data['2026']:,.0f}")

        all_forecasts[cls] = {
            "time_series": ts.to_dict(),
            "best_model": best_model,
            "backtest": bt,
            "scenarios": scenarios,
        }

    # Build forecast output table
    forecast_rows = []
    for cls, data in all_forecasts.items():
        for yr, val in data["time_series"].items():
            forecast_rows.append({
                "diabetes_class": cls,
                "year": int(yr),
                "fills": float(val),
                "type": "actual"
            })
        sc = data["scenarios"]
        for case in ["base_case", "upside_case", "downside_case"]:
            if case in sc and isinstance(sc[case], dict):
                for yr in ["2025", "2026"]:
                    if yr in sc[case]:
                        forecast_rows.append({
                            "diabetes_class": cls,
                            "year": int(yr),
                            "fills": float(sc[case][yr]),
                            "type": case
                        })

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv("outputs/forecast.csv", index=False)

    # Save detailed results
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("outputs/forecast_analysis.json", "w") as f:
        json.dump(all_forecasts, f, indent=2, default=convert_types)

    print(f"\n{'='*60}")
    print("TASKS 29-34 COMPLETE — Forecasting saved to outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
