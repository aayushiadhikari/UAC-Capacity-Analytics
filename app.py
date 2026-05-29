import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="UAC Capacity Analytics", layout="wide")

st.title("System Capacity & Care Load Analytics for Unaccompanied Children")

df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")
df = df.dropna(how="all")

df.columns = [
    "Date", "Apprehended", "CBP_Custody",
    "Transferred", "HHS_Care", "Discharged"
]

df["Date"] = pd.to_datetime(df["Date"])

num_cols = ["Apprehended", "CBP_Custody", "Transferred", "HHS_Care", "Discharged"]

for col in num_cols:
    df[col] = df[col].astype(str).str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df.sort_values("Date").reset_index(drop=True)

df["Total_System_Load"] = df["CBP_Custody"] + df["HHS_Care"]
df["Net_Daily_Intake"] = df["Transferred"] - df["Discharged"]
df["Care_Load_Growth_Rate"] = df["Total_System_Load"].pct_change() * 100
df["Backlog"] = df["Net_Daily_Intake"].cumsum()
df["Discharge_Offset_Ratio"] = df["Discharged"] / df["Transferred"].replace(0, pd.NA)

df["Load_7Day_Avg"] = df["Total_System_Load"].rolling(7).mean()
df["Load_14Day_Avg"] = df["Total_System_Load"].rolling(14).mean()
df["Volatility_7Day"] = df["Total_System_Load"].rolling(7).std()

load_norm = df["Total_System_Load"] / df["Total_System_Load"].max()
backlog_norm = abs(df["Backlog"]) / abs(df["Backlog"]).max()
volatility_norm = df["Volatility_7Day"].fillna(0) / df["Volatility_7Day"].max()

df["Stress_Score"] = (
    (0.5 * load_norm) +
    (0.3 * backlog_norm) +
    (0.2 * volatility_norm)
) * 100

st.sidebar.header("Dashboard Filters")

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

current_stress = filtered_df["Stress_Score"].iloc[-1]

if current_stress < 30:
    st.sidebar.success(f"Low Stress ({current_stress:.1f})")
elif current_stress < 60:
    st.sidebar.warning(f"Moderate Stress ({current_stress:.1f})")
else:
    st.sidebar.error(f"High Stress ({current_stress:.1f})")

st.success("Dataset Loaded and Cleaned Successfully ✅")

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", len(filtered_df))

with col2:
    st.metric("Total Columns", len(filtered_df.columns))

with col3:
    st.metric(
        "Date Range",
        f"{filtered_df['Date'].min().date()} to {filtered_df['Date'].max().date()}"
    )

st.subheader("Data Quality Check")

missing_values = filtered_df.isnull().sum()

st.dataframe(
    missing_values.reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    )
)

st.header("Key Performance Indicators")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Current System Load", f"{int(filtered_df['Total_System_Load'].iloc[-1]):,}")

with c2:
    st.metric("Current HHS Care", f"{int(filtered_df['HHS_Care'].iloc[-1]):,}")

with c3:
    st.metric("Current CBP Load", f"{int(filtered_df['CBP_Custody'].iloc[-1]):,}")

with c4:
    st.metric("Current Backlog", f"{int(filtered_df['Backlog'].iloc[-1]):,}")

with c5:
    st.metric("Stress Score", f"{current_stress:.1f}")

st.header("Trend Analysis")

fig1 = px.line(filtered_df, x="Date", y="Total_System_Load", title="Total System Load Over Time")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(filtered_df, x="Date", y=["CBP_Custody", "HHS_Care"], title="CBP vs HHS Care Load")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(filtered_df, x="Date", y="Net_Daily_Intake", title="Net Daily Intake Pressure")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.line(filtered_df, x="Date", y="Backlog", title="Backlog Indicator Over Time")
st.plotly_chart(fig4, use_container_width=True)

st.header("Pressure & Stress Monitoring")

fig5 = px.line(
    filtered_df,
    x="Date",
    y=["Total_System_Load", "Load_7Day_Avg", "Load_14Day_Avg"],
    title="Total Load with 7-Day and 14-Day Rolling Averages"
)
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.line(filtered_df, x="Date", y="Volatility_7Day", title="7-Day Care Load Volatility")
st.plotly_chart(fig6, use_container_width=True)

fig7 = px.line(filtered_df, x="Date", y="Stress_Score", title="Healthcare System Stress Score")
st.plotly_chart(fig7, use_container_width=True)

st.header("30-Day Forecasting Analysis")

forecast_days = st.slider(
    "Select Forecast Horizon (Days)",
    min_value=7,
    max_value=60,
    value=30
)

forecast_data = df[["Date", "Total_System_Load"]].rename(
    columns={"Date": "ds", "Total_System_Load": "y"}
)

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(forecast_data)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)

fig8 = px.line(
    forecast_result,
    x="ds",
    y=["yhat", "yhat_lower", "yhat_upper"],
    title=f"Next {forecast_days} Days Forecast: Total System Load"
)
st.plotly_chart(fig8, use_container_width=True)

latest_forecast = forecast_result["yhat"].iloc[-1]

st.metric(
    f"Predicted System Load After {forecast_days} Days",
    f"{int(latest_forecast):,}"
)

st.header("Capacity Strain Detection")

strain_days = 0
max_strain = 0

for value in filtered_df["Net_Daily_Intake"]:
    if value > 0:
        strain_days += 1
        max_strain = max(max_strain, strain_days)
    else:
        strain_days = 0

if max_strain >= 7:
    st.error(
        f"⚠️ Capacity Strain Detected: Positive intake pressure persisted for {max_strain} consecutive days."
    )
else:
    st.success("✅ No major capacity strain window detected.")

st.metric("Longest Strain Window (Days)", max_strain)

st.header("Executive Summary")

current_load = int(filtered_df["Total_System_Load"].iloc[-1])
current_hhs = int(filtered_df["HHS_Care"].iloc[-1])
current_cbp = int(filtered_df["CBP_Custody"].iloc[-1])

if current_stress < 30:
    stress_label = "Low"
elif current_stress < 60:
    stress_label = "Moderate"
else:
    stress_label = "High"

summary = f"""
Current Total System Load is {current_load:,} children.

HHS Care Load currently stands at {current_hhs:,} children, while CBP custody load is {current_cbp:,} children.

The calculated system stress score is {current_stress:.1f}, which indicates a {stress_label} stress environment.

Forecasting analysis suggests an estimated system load of {int(latest_forecast):,} children after {forecast_days} days.

The longest detected capacity strain window is {max_strain} consecutive days.

Decision makers should continue monitoring intake, transfer, discharge, backlog, and stress trends to maintain operational stability.
"""

st.info(summary)

st.subheader("Derived Metrics Preview")
st.dataframe(filtered_df.tail())