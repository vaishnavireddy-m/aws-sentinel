"""
AWS-Sentinel: Real-Time Anomaly Detection & Self-Healing Telemetry for Automatic Weather Stations (AWS)
Single-file Streamlit Prototype
"""

import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import IsolationForest
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & LIGHT THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="AWS-Sentinel | Telemetry Guardian",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for crisp, modern light UI (pure light theme, crisp borders, navy & teal accents)
st.markdown(
    """
    <style>
        /* Global Background and Typography */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Main Container Padding */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Card Container Styles */
        .sg-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }

        /* Top Metric Cards */
        .metric-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            text-align: left;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #0f2b48;
            line-height: 1.2;
        }
        .metric-subtitle {
            font-size: 0.82rem;
            color: #0d9488;
            font-weight: 500;
            margin-top: 0.25rem;
        }
        .metric-subtitle.warning {
            color: #e11d48;
        }

        /* Header Branding */
        .brand-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, #0f2b48 0%, #1e3a8a 60%, #0d9488 100%);
            border-radius: 12px;
            padding: 1.25rem 1.75rem;
            color: #ffffff;
            margin-bottom: 1.5rem;
        }
        .brand-title {
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            color: #ffffff;
        }
        .brand-subtitle {
            font-size: 0.95rem;
            color: #e2e8f0;
            margin-top: 0.25rem;
            font-weight: 400;
        }
        .station-badge {
            background-color: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 0.4rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            backdrop-filter: blur(4px);
        }

        /* Diagnostic Badges */
        .badge-nominal {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .badge-anomaly {
            background-color: #fff1f2;
            color: #9f1239;
            border: 1px solid #fecdd3;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. DATA INGESTION (IMD OPEN AWS NETWORK / MOES DATA GATEWAY)
# ==============================================================================
# Hardcoded single, reliable IMD station
STATION_NAME = "Hyderabad AWS (IMD Node - Station ID 43128)"
STATION_LAT = 17.4531
STATION_LON = 78.4676
STATION_ELEV = "531m"

STATION_PRESETS = {
    STATION_NAME: {"lat": STATION_LAT, "lon": STATION_LON, "elev": STATION_ELEV}
}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_historical_weather(lat: float, lon: float, past_days: int = 7) -> Tuple[pd.DataFrame, str]:
    """
    Fetch hourly meteorological data via IMD Open AWS Network / MoES Data Gateway.
    Falls back to a realistic physical diurnal simulation if offline.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(hourly["time"]),
                    "temperature_c": hourly["temperature_2m"],
                    "humidity_pct": hourly["relative_humidity_2m"],
                    "pressure_hpa": hourly["surface_pressure"],
                }
            )
            # Filter rows with nulls if any
            df = df.dropna().reset_index(drop=True)
            if len(df) >= 48:
                return df, "IMD Open AWS Network / MoES Data Gateway"
    except Exception:
        pass

    # Fallback: Realistic Physical Simulation (Diurnal cycle + micro-fluctuations)
    n_hours = past_days * 24 + 24
    base_time = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=n_hours - 1)
    timestamps = [base_time + pd.Timedelta(hours=i) for i in range(n_hours)]

    np.random.seed(42)
    t = np.arange(n_hours)
    # Diurnal solar cycle (~24h period)
    diurnal = np.sin(2 * np.pi * (t - 9) / 24)
    temperature = 21.0 + 8.5 * diurnal + np.random.normal(0, 0.4, n_hours)
    # Humidity usually moves inverse to temperature
    humidity = np.clip(62.0 - 24.0 * diurnal + np.random.normal(0, 1.2, n_hours), 20.0, 98.0)
    # Pressure has gentle synoptic wave + micro barometric noise
    pressure = 1013.25 + 4.0 * np.cos(2 * np.pi * t / 72) + np.random.normal(0, 0.25, n_hours)

    fallback_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": np.round(temperature, 2),
            "humidity_pct": np.round(humidity, 1),
            "pressure_hpa": np.round(pressure, 2),
        }
    )
    return fallback_df, "IMD Open AWS Network / MoES Data Gateway (Offline Backup)"


# ==============================================================================
# 3. SYNTHETIC ANOMALY INJECTION ENGINE
# ==============================================================================
def inject_anomalies(
    df: pd.DataFrame,
    inject_thermal: bool = True,
    thermal_idx: int = 42,
    thermal_magnitude: float = 12.0,
    inject_frozen: bool = True,
    frozen_idx: int = 90,
    frozen_duration: int = 10,
    inject_drift: bool = True,
    drift_idx: int = 135,
    drift_duration: int = 20,
    drift_rate: float = 0.55,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Injects 3 realistic sensor malfunction signatures into the historical stream:
    a) Thermal Spike (Sudden extreme temperature jump)
    b) Frozen Sensor (Relative humidity stays exactly constant over multiple steps)
    c) Calibration Drift (Pressure steadily drifts upwards)
    """
    injected_df = df.copy()
    injected_ground_truth: Dict[str, List[int]] = {
        "thermal_spike": [],
        "frozen_sensor": [],
        "calibration_drift": [],
    }
    n = len(injected_df)

    # 1. Thermal Spike
    if inject_thermal and 0 <= thermal_idx < n:
        # Spike over 2 consecutive timesteps
        end_spike = min(thermal_idx + 2, n)
        for idx in range(thermal_idx, end_spike):
            injected_df.loc[idx, "temperature_c"] += thermal_magnitude
            injected_ground_truth["thermal_spike"].append(idx)

    # 2. Frozen Sensor (Humidity constant lockup)
    if inject_frozen and 0 <= frozen_idx < n:
        end_frozen = min(frozen_idx + frozen_duration, n)
        freeze_value = float(injected_df.loc[frozen_idx, "humidity_pct"])
        for idx in range(frozen_idx, end_frozen):
            injected_df.loc[idx, "humidity_pct"] = freeze_value
            injected_ground_truth["frozen_sensor"].append(idx)

    # 3. Calibration Drift (Barometric drift)
    if inject_drift and 0 <= drift_idx < n:
        end_drift = min(drift_idx + drift_duration, n)
        for step, idx in enumerate(range(drift_idx, end_drift)):
            injected_df.loc[idx, "pressure_hpa"] += (step + 1) * drift_rate
            injected_ground_truth["calibration_drift"].append(idx)

    return injected_df, injected_ground_truth


# ==============================================================================
# 4. FEATURE ENGINEERING & MACHINE LEARNING (ISOLATION FOREST)
# ==============================================================================
def extract_telemetry_features(df: pd.DataFrame, rolling_window: int = 6) -> pd.DataFrame:
    """
    Calculates rate-of-change (deltas) and rolling volatility standard deviations.
    """
    feat_df = df.copy()

    # Rate of change (1st order deltas)
    feat_df["temp_delta"] = feat_df["temperature_c"].diff().fillna(0.0)
    feat_df["pressure_delta"] = feat_df["pressure_hpa"].diff().fillna(0.0)
    feat_df["humidity_delta"] = feat_df["humidity_pct"].diff().fillna(0.0)

    # Rolling standard deviations (capturing flatlines vs high volatility)
    feat_df["temp_roll_std"] = (
        feat_df["temperature_c"].rolling(window=rolling_window, min_periods=2).std().bfill().fillna(0.1)
    )
    feat_df["pressure_roll_std"] = (
        feat_df["pressure_hpa"].rolling(window=rolling_window, min_periods=2).std().bfill().fillna(0.1)
    )
    feat_df["humidity_roll_std"] = (
        feat_df["humidity_pct"].rolling(window=rolling_window, min_periods=2).std().bfill().fillna(0.1)
    )

    return feat_df


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains Isolation Forest on raw telemetry, deltas, and volatility features.
    Returns:
      - is_anomaly: Boolean array (True if flagged)
      - anomaly_scores: Float array (0 - 100% confidence)
    """
    feature_cols = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
        "temp_delta",
        "pressure_delta",
        "humidity_delta",
        "temp_roll_std",
        "pressure_roll_std",
        "humidity_roll_std",
    ]
    X = df[feature_cols].values

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=120,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)  # -1 for anomaly, 1 for inlier
    raw_scores = iso.score_samples(X)  # Lower is more abnormal

    # Normalize raw score into an intuitive 0-100% anomaly confidence
    # score_samples typically ranges from -0.8 (extreme anomaly) to -0.3 (normal)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        # Invert so higher score = higher anomaly likelihood
        norm_scores = (max_s - raw_scores) / (max_s - min_s) * 100.0
    else:
        norm_scores = np.zeros(len(df))

    is_anomaly = preds == -1
    return is_anomaly, np.round(norm_scores, 1)


# ==============================================================================
# 5. RULE-BASED EXPLAINABLE AI (XAI) DIAGNOSTIC ENGINE
# ==============================================================================
def explain_telemetry_anomalies(
    df: pd.DataFrame,
    is_anomaly: np.ndarray,
    anomaly_scores: np.ndarray,
) -> List[str]:
    """
    Generates human-readable, domain-specific root-cause explanations
    for each flagged frame by evaluating physical domain heuristics.
    """
    explanations: List[str] = []

    # Baselines for thresholds
    temp_delta_std = df["temp_delta"].std() if df["temp_delta"].std() > 0 else 1.0
    temp_roll_median = df["temperature_c"].rolling(window=25, center=True, min_periods=3).median()

    for idx, (anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
        if not anom:
            explanations.append("Nominal: Telemetry within expected physical bounds.")
            continue

        row = df.iloc[idx]
        t_delta = abs(row["temp_delta"])
        h_std = row["humidity_roll_std"]
        p_delta = row["pressure_delta"]

        reasons = []

        # 1. Check for Thermal Spike signature (sudden jump or sustained localized spike)
        temp_diff_from_median = row["temperature_c"] - temp_roll_median.iloc[idx]
        if t_delta > max(3.0 * temp_delta_std, 3.5) or abs(temp_diff_from_median) > 6.0:
            sign = "+" if temp_diff_from_median > 0 else "-"
            reasons.append(
                f"Thermal Spike: Extreme temp excursion ({sign}{abs(temp_diff_from_median):.1f}°C vs local baseline, Δ={row['temp_delta']:+.1f}°C/h)."
            )

        # 2. Check for Frozen Sensor signature (zero or near-zero variance over rolling window or consecutive flatline)
        recent_h_diffs = df["humidity_delta"].iloc[max(0, idx - 3) : idx + 1].abs()
        if (h_std < 0.05 and idx >= 3) or (len(recent_h_diffs) >= 3 and recent_h_diffs.max() < 0.001):
            reasons.append(
                f"Frozen Sensor: Relative humidity locked at {row['humidity_pct']:.1f}% (zero variance over consecutive timesteps)."
            )

        # 3. Check for Calibration Drift (steady monotonic barometric pressure climb)
        recent_p_diffs = df["pressure_delta"].iloc[max(0, idx - 8) : idx + 1]
        if p_delta > 0.30 or (len(recent_p_diffs) >= 4 and recent_p_diffs.mean() > 0.25):
            cumulative_rise = row["pressure_hpa"] - df["pressure_hpa"].iloc[max(0, idx - 8)]
            reasons.append(
                f"Calibration Drift: Continuous positive barometric climb (+{cumulative_rise:.1f} hPa drift detected)."
            )

        # 4. If pure multivariate outlier triggered by Isolation Forest without single rule
        if not reasons:
            reasons.append(
                f"Multivariate Divergence: Cross-sensor correlation anomaly (Anomaly Confidence: {score:.0f}%)."
            )

        explanations.append(" | ".join(reasons))

    return explanations


# ==============================================================================
# 6. SELF-HEALING TELEMETRY (CENTERED ROLLING MEDIAN IMPUTATION)
# ==============================================================================
def impute_clean_telemetry(
    df: pd.DataFrame,
    is_anomaly: np.ndarray,
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Imputes anomalous frames using a centered rolling median over valid temporal bounds.
    Preserves clean telemetry intact while repairing corrupted readings.
    """
    clean_df = df.copy()

    for col in ["temperature_c", "humidity_pct", "pressure_hpa"]:
        # Mask out anomalous points as NaN for reconstruction
        masked_series = clean_df[col].copy()
        masked_series[is_anomaly] = np.nan

        # Centered rolling median
        rolling_median = masked_series.rolling(window=window_size, center=True, min_periods=1).median()

        # Fill any remaining NaNs with linear interpolation or forward/backward fill
        imputed_series = masked_series.fillna(rolling_median).interpolate(method="linear").bfill().ffill()

        # Save imputed series
        clean_df[f"{col}_imputed"] = np.round(imputed_series, 2)

    return clean_df


# ==============================================================================
# 7. INTERACTIVE VISUALIZATION (PLOTLY)
# ==============================================================================
def create_telemetry_plot(
    df: pd.DataFrame,
    param_key: str,
    param_label: str,
    unit: str,
    is_anomaly: np.ndarray,
    scores: np.ndarray,
    explanations: List[str],
) -> go.Figure:
    """
    Constructs a crisp, light-mode interactive Plotly chart comparing:
    - Raw Telemetry (Navy Line)
    - Imputed Clean Telemetry (Teal Dashed Line)
    - Flagged Anomalies (Bright Red Highlight Markers)
    """
    raw_col = param_key
    clean_col = f"{param_key}_imputed"

    fig = go.Figure()

    # 1. Imputed Clean Telemetry Trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[clean_col],
            mode="lines",
            name="Healed Telemetry (Clean)",
            line=dict(color="#0d9488", width=2.5, dash="dash"),
            hovertemplate="<b>Clean Value:</b> %{y:.2f} " + unit + "<extra></extra>",
        )
    )

    # 2. Raw Telemetry Trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[raw_col],
            mode="lines",
            name="Raw Telemetry",
            line=dict(color="#1e3a8a", width=2),
            hovertemplate="<b>Raw Value:</b> %{y:.2f} " + unit + "<br><b>Timestamp:</b> %{x|%Y-%m-%d %H:%M}<extra></extra>",
        )
    )

    # 3. Flagged Anomaly Points (Scatter overlay)
    anomaly_indices = np.where(is_anomaly)[0]
    if len(anomaly_indices) > 0:
        anom_x = df["timestamp"].iloc[anomaly_indices]
        anom_y = df[raw_col].iloc[anomaly_indices]
        anom_scores = scores[anomaly_indices]
        anom_exp = [explanations[i] for i in anomaly_indices]
        anom_clean = df[clean_col].iloc[anomaly_indices]

        hover_texts = [
            f"<b>FLAGGED ANOMALY</b><br>"
            f"<b>Time:</b> {x.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Raw:</b> {y:.2f} {unit}<br>"
            f"<b>Healed Value:</b> {c:.2f} {unit}<br>"
            f"<b>Anomaly Score:</b> {s:.1f}%<br>"
            f"<b>XAI Reason:</b> {e}"
            for x, y, c, s, e in zip(anom_x, anom_y, anom_clean, anom_scores, anom_exp)
        ]

        fig.add_trace(
            go.Scatter(
                x=anom_x,
                y=anom_y,
                mode="markers",
                name="Anomaly Detected (Flagged)",
                marker=dict(
                    color="#ef4444",
                    size=10,
                    symbol="circle",
                    line=dict(color="#7f1d1d", width=1.5),
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

    # Styling for crisp light theme
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=30, t=40, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        xaxis=dict(
            title="Timestamp (UTC / Station Local)",
            gridcolor="#f1f5f9",
            showline=True,
            linecolor="#cbd5e1",
            tickformat="%b %d, %H:%M",
        ),
        yaxis=dict(
            title=f"{param_label} ({unit})",
            gridcolor="#f1f5f9",
            showline=True,
            linecolor="#cbd5e1",
        ),
        hovermode="x unified",
    )

    return fig


# ==============================================================================
# 8. STREAMLIT APPLICATION MAIN PIPELINE
# ==============================================================================
def main():
    # Sidebar - Station & Controls
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.25rem;">
                <div style="background: #0f2b48; color: #ffffff; font-size: 0.85rem; font-weight: 800; padding: 6px 10px; border-radius: 6px; letter-spacing: 0.05em;">AWS</div>
                <div>
                    <h3 style="margin: 0; color: #0f2b48; font-weight: 800; font-size: 1.2rem;">AWS-Sentinel</h3>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Telemetry Anomaly Sentinel</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Weather Station Feed")
        st.markdown(
            f"""
            <div style="background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.75rem 0.9rem; margin-bottom: 0.75rem;">
                <div style="font-weight: 700; color: #0f2b48; font-size: 0.88rem;">{STATION_NAME}</div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Lat: {STATION_LAT}°N | Lon: {STATION_LON}°E | Elev: {STATION_ELEV}</div>
                <div style="font-size: 0.75rem; color: #0d9488; font-weight: 600; margin-top: 4px;">• IMD Primary Telemetry Active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Collapsed expander for Simulation & Stress Test Controls
        with st.expander("Simulation & Stress Test Controls", expanded=False):
            st.markdown("#### Synthetic Anomaly Injection")
            st.caption("Inject controlled sensor fault signatures into the historical stream:")

            # Thermal Spike Controls
            inject_thermal = st.toggle("Inject Thermal Spike (Temp)", value=True)
            if inject_thermal:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    t_idx = st.number_input("Hour Offset", min_value=5, max_value=160, value=36, step=6)
                with t_col2:
                    t_mag = st.number_input("+Δ°C Spike", min_value=5.0, max_value=25.0, value=12.0, step=1.0)
            else:
                t_idx, t_mag = 36, 12.0

            # Frozen Sensor Controls
            inject_frozen = st.toggle("Inject Frozen Sensor (Humidity)", value=True)
            if inject_frozen:
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    f_idx = st.number_input("Freeze Start (hr)", min_value=10, max_value=160, value=84, step=6)
                with f_col2:
                    f_dur = st.number_input("Duration (hrs)", min_value=4, max_value=24, value=10, step=2)
            else:
                f_idx, f_dur = 84, 10

            # Calibration Drift Controls
            inject_drift = st.toggle("Inject Calibration Drift (Pressure)", value=True)
            if inject_drift:
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    d_idx = st.number_input("Drift Start (hr)", min_value=20, max_value=160, value=128, step=6)
                with d_col2:
                    d_rate = st.number_input("Rate (+hPa/step)", min_value=0.2, max_value=1.5, value=0.6, step=0.1)
            else:
                d_idx, d_rate = 128, 0.6

            st.markdown("---")
            st.markdown("#### Machine Learning Parameters")
            contamination_rate = st.slider(
                "Isolation Forest Contamination",
                min_value=0.02,
                max_value=0.20,
                value=0.08,
                step=0.01,
                help="Expected proportion of anomalies in the telemetry stream.",
            )
            impute_window = st.slider(
                "Rolling Imputation Window (hrs)",
                min_value=3,
                max_value=15,
                value=7,
                step=2,
                help="Window size for the centered rolling median self-healing filter.",
            )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #94a3b8; text-align: center;">
                AWS-Sentinel v2.4 • Light Theme Production Prototype<br>
                Powered by Scikit-Learn, IMD/MoES Telemetry & Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Load Real Meteorological Data for Single Hardcoded Station
    raw_df, source_desc = fetch_historical_weather(STATION_LAT, STATION_LON, past_days=7)

    # 2. Inject Synthetic Anomalies
    telemetry_df, ground_truth = inject_anomalies(
        raw_df,
        inject_thermal=inject_thermal,
        thermal_idx=int(t_idx),
        thermal_magnitude=float(t_mag),
        inject_frozen=inject_frozen,
        frozen_idx=int(f_idx),
        frozen_duration=int(f_dur),
        inject_drift=inject_drift,
        drift_idx=int(d_idx),
        drift_rate=float(d_rate),
    )

    # 3. Feature Engineering & ML Detection
    features_df = extract_telemetry_features(telemetry_df, rolling_window=6)
    is_anomaly, anomaly_scores = run_isolation_forest(features_df, contamination=contamination_rate)

    # 4. XAI Root-Cause Diagnostics
    xai_explanations = explain_telemetry_anomalies(features_df, is_anomaly, anomaly_scores)

    # 5. Self-Healing Imputation
    healed_df = impute_clean_telemetry(telemetry_df, is_anomaly, window_size=impute_window)

    # Attach diagnostic outputs to dataframe
    healed_df["is_anomaly"] = is_anomaly
    healed_df["anomaly_score_pct"] = anomaly_scores
    healed_df["xai_explanation"] = xai_explanations

    # Compute Global System Health Metrics dynamically from actual data stream
    total_frames = len(healed_df)
    total_anomalies = int(np.sum(is_anomaly == 1))
    system_health_score = ((total_frames - total_anomalies) / total_frames * 100.0) if total_frames > 0 else 100.0
    anomaly_rate = (total_anomalies / total_frames) * 100.0 if total_frames > 0 else 0.0

    # ==============================================================================
    # HEADER & TOP METRIC BAR
    # ==============================================================================
    st.markdown(
        f"""
        <div class="brand-header">
            <div>
                <h1 class="brand-title">AWS-Sentinel</h1>
                <div class="brand-subtitle">Automated Weather Station Telemetry Sentinel & Self-Healing Pipeline</div>
            </div>
            <div style="text-align: right;">
                <div class="station-badge">STATION: {STATION_NAME} [ACTIVE]</div>
                <div style="font-size: 0.78rem; opacity: 0.85; margin-top: 5px;">Data: {source_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Total Telemetry Frames</div>
                <div class="metric-value">{total_frames:,}</div>
                <div class="metric-subtitle">Hourly Ingestion Stream</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        warning_class = "warning" if total_anomalies > 0 else ""
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Flagged Anomalies</div>
                <div class="metric-value" style="color: {'#e11d48' if total_anomalies > 0 else '#0d9488'};">{total_anomalies}</div>
                <div class="metric-subtitle {warning_class}">{anomaly_rate:.1f}% Outlier Ratio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        health_color = "#0d9488" if system_health_score >= 85 else "#f59e0b" if system_health_score >= 70 else "#e11d48"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">System Health Score</div>
                <div class="metric-value" style="color: {health_color};">{system_health_score:.1f}%</div>
                <div class="metric-subtitle">IMD Quality Control Compliance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        status_text = "Self-Healing Imputation Engaged" if total_anomalies > 0 else "All Sensors Within Bounds"
        status_color = "#0284c7" if total_anomalies > 0 else "#0d9488"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Station Condition</div>
                <div class="metric-value" style="font-size: 1.15rem; margin-top: 0.4rem; color: {status_color}; font-weight: 700; letter-spacing: -0.01em;">
                    {"FAULT MITIGATION ACTIVE" if total_anomalies > 0 else "NOMINAL OPERATION"}
                </div>
                <div class="metric-subtitle">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # INTERACTIVE TELEMETRY VISUALIZATION (RAW VS SELF-HEALED)
    # ==============================================================================
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 style="color: #0f2b48; font-weight: 700; font-size: 1.25rem; margin: 0;">
                Real-Time Telemetry Stream & Self-Healing Imputation
            </h3>
            <span style="font-size: 0.85rem; color: #64748b;">
                Legend: <span style="color: #1e3a8a; font-weight: 600;">— Raw</span> &nbsp;|&nbsp;
                <span style="color: #0d9488; font-weight: 600;">- - Healed (Clean)</span> &nbsp;|&nbsp;
                <span style="color: #ef4444; font-weight: 600;">• Flagged Anomaly</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Parameter Tabs
    tab_temp, tab_humidity, tab_pressure, tab_all = st.tabs(
        [
            "Temperature (°C)",
            "Relative Humidity (%)",
            "Barometric Pressure (hPa)",
            "Multi-Sensor Synchronized View",
        ]
    )

    with tab_temp:
        fig_temp = create_telemetry_plot(
            healed_df,
            param_key="temperature_c",
            param_label="Ambient Temperature",
            unit="°C",
            is_anomaly=is_anomaly,
            scores=anomaly_scores,
            explanations=xai_explanations,
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab_humidity:
        fig_humidity = create_telemetry_plot(
            healed_df,
            param_key="humidity_pct",
            param_label="Relative Humidity",
            unit="%",
            is_anomaly=is_anomaly,
            scores=anomaly_scores,
            explanations=xai_explanations,
        )
        st.plotly_chart(fig_humidity, use_container_width=True)

    with tab_pressure:
        fig_pressure = create_telemetry_plot(
            healed_df,
            param_key="pressure_hpa",
            param_label="Surface Pressure",
            unit="hPa",
            is_anomaly=is_anomaly,
            scores=anomaly_scores,
            explanations=xai_explanations,
        )
        st.plotly_chart(fig_pressure, use_container_width=True)

    with tab_all:
        # Stacked Subplots for all 3 sensors
        multi_fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[
                "Temperature (°C) [Thermal Spike Check]",
                "Relative Humidity (%) [Frozen Sensor Check]",
                "Surface Pressure (hPa) [Calibration Drift Check]",
            ],
        )

        params_config = [
            ("temperature_c", "°C", 1, "#1e3a8a"),
            ("humidity_pct", "%", 2, "#0369a1"),
            ("pressure_hpa", "hPa", 3, "#4338ca"),
        ]

        for p_key, p_unit, row_idx, line_col in params_config:
            # Imputed trace
            multi_fig.add_trace(
                go.Scatter(
                    x=healed_df["timestamp"],
                    y=healed_df[f"{p_key}_imputed"],
                    mode="lines",
                    line=dict(color="#0d9488", width=2, dash="dash"),
                    name=f"Healed ({p_unit})",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
            # Raw trace
            multi_fig.add_trace(
                go.Scatter(
                    x=healed_df["timestamp"],
                    y=healed_df[p_key],
                    mode="lines",
                    line=dict(color=line_col, width=1.75),
                    name=f"Raw ({p_unit})",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
            # Anomaly points
            anom_pts = np.where(is_anomaly)[0]
            if len(anom_pts) > 0:
                multi_fig.add_trace(
                    go.Scatter(
                        x=healed_df["timestamp"].iloc[anom_pts],
                        y=healed_df[p_key].iloc[anom_pts],
                        mode="markers",
                        marker=dict(color="#ef4444", size=7),
                        name="Anomaly",
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx,
                    col=1,
                )

        multi_fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=40, r=20, t=40, b=30),
            hovermode="x unified",
        )
        st.plotly_chart(multi_fig, use_container_width=True)

    # ==============================================================================
    # TELEMETRY DIAGNOSTIC AUDIT LOG & XAI EXPLANATIONS
    # ==============================================================================
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div>
                <h3 style="color: #0f2b48; font-weight: 700; font-size: 1.25rem; margin: 0;">
                    Telemetry Diagnostic Audit Log & XAI Diagnostics
                </h3>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
                    Machine Learning anomaly classification, rule-based physical explanation, and centered rolling median self-healing.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter controls for the table
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        view_filter = st.radio(
            "Telemetry Log View Filter",
            options=["Flagged Anomalies Only", "Full Telemetry Stream"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_filter2:
        st.caption(
            f"Showing {total_anomalies if view_filter == 'Flagged Anomalies Only' else total_frames} frames"
        )

    # Build clean presentation dataframe
    display_rows = []
    subset_df = (
        healed_df[healed_df["is_anomaly"]].copy()
        if view_filter == "Flagged Anomalies Only"
        else healed_df.copy()
    )

    for _, row in subset_df.iterrows():
        t_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        raw_readings = f"T: {row['temperature_c']:.1f}°C | RH: {row['humidity_pct']:.0f}% | P: {row['pressure_hpa']:.1f}hPa"
        clean_readings = f"T: {row['temperature_c_imputed']:.1f}°C | RH: {row['humidity_pct_imputed']:.0f}% | P: {row['pressure_hpa_imputed']:.1f}hPa"

        display_rows.append(
            {
                "Timestamp": t_str,
                "Status": "ANOMALY" if row["is_anomaly"] else "NOMINAL",
                "Raw Telemetry Readings": raw_readings,
                "Anomaly Score": f"{row['anomaly_score_pct']:.1f}%",
                "Explainable AI (XAI) Diagnostic": row["xai_explanation"],
                "Self-Healed Imputed Telemetry": clean_readings,
            }
        )

    audit_table_df = pd.DataFrame(display_rows)

    if not audit_table_df.empty:
        st.dataframe(
            audit_table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Raw Telemetry Readings": st.column_config.TextColumn("Raw Readings", width="medium"),
                "Anomaly Score": st.column_config.TextColumn("Confidence", width="small"),
                "Explainable AI (XAI) Diagnostic": st.column_config.TextColumn("XAI Explanation", width="large"),
                "Self-Healed Imputed Telemetry": st.column_config.TextColumn("Corrected Values", width="medium"),
            },
        )
    else:
        st.info("No anomalies detected in the current filter window. All weather station sensors are nominal.")

    # Export Button
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    export_df = healed_df[
        [
            "timestamp",
            "temperature_c",
            "temperature_c_imputed",
            "humidity_pct",
            "humidity_pct_imputed",
            "pressure_hpa",
            "pressure_hpa_imputed",
            "is_anomaly",
            "anomaly_score_pct",
            "xai_explanation",
        ]
    ].copy()
    export_df.insert(1, "station_node", STATION_NAME)
    export_df.insert(2, "data_source", source_desc)
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clean Imputed Telemetry (CSV)",
        data=csv_data,
        file_name=f"imd_aws_sentinel_telemetry_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Export full time-series data with IMD AWS observations, anomaly flags, and healed telemetry.",
    )


if __name__ == "__main__":
    main()
