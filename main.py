import streamlit as st
from prophet.plot import plot_plotly
from component.data_loader import load_data
from component.metrics_summary import show_summary
from component.forecast_model import train_model
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()  

st.set_page_config(layout="wide")
st.title("🌍 COVID Dashboard")

df = load_data()

columns_needed = [
    "location", "date", "new_cases", "total_cases", "total_deaths",
    "total_vaccinations", "total_tests", "new_deaths","year",
    "gdp_per_capita", "human_development_index", "iso_code", "population"
]
df = df[columns_needed]

# Country selection
countries = sorted(df["location"].dropna().unique())
country = st.sidebar.selectbox("Select a Country", countries, index=countries.index("India"))
years = sorted(df[df["location"] == country]["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select a Year", years, index=len(years) - 1)
df_country_all = df[(df["location"] == country) & (df["year"] == selected_year)].sort_values("date")
top_n = st.sidebar.slider("Top N Countries by Active Cases", 5, 50, 10)

st.markdown("### 📊 Latest COVID Stats (with Daily Change)")

if df_country_all.shape[0] < 2:
    st.warning("Not enough data to compute summary metrics.")
else:
    show_summary(df_country_all)

# Filter country data for forecasting
df_country = df[df["location"] == country].copy()
df_country = df_country[["date", "new_cases", "total_vaccinations", "gdp_per_capita", "human_development_index"]]

# Preprocessing
df_country["new_cases"] = df_country["new_cases"].rolling(7).mean()
df_country["total_vaccinations"] = df_country["total_vaccinations"].ffill().bfill()
df_country["gdp_per_capita"] = df_country["gdp_per_capita"].ffill().bfill()
df_country["human_development_index"] = df_country["human_development_index"].ffill().bfill()
df_country = df_country.dropna(subset=["new_cases", "total_vaccinations", "gdp_per_capita", "human_development_index"])

# Rename columns for Prophet
df_prophet = df_country.rename(columns={
    "date": "ds",
    "new_cases": "y",
    "total_vaccinations": "vaccinations",
    "gdp_per_capita": "gdp",
    "human_development_index": "health_index"
})
df_prophet["y"] = df_prophet["y"].apply(lambda x: max(x, 0))

# Forecast window and tuning
periods = st.sidebar.slider("Forecast Days", 7, 90, 30)
changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale", 0.001, 1.0, 0.05)

if df_prophet.shape[0] < 2:
    st.error("❌ Not enough data to train the model. Try another country.")
    st.stop()

if df_prophet.empty:
    st.warning("Insufficient data for forecasting.")
    st.stop()
# Prophet model
with st.spinner("Training model..."):
    model = train_model(df_prophet, changepoint_prior_scale)
    future = model.make_future_dataframe(periods=periods)
    future = future.merge(df_prophet[["ds", "vaccinations", "gdp", "health_index"]], on="ds", how="left")
    for col in ["vaccinations", "gdp", "health_index"]:
        future[col] = future[col].ffill().bfill()
    forecast = model.predict(future)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    fig1.update_traces(line=dict(color="orange"))  # Forecast line color
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    actual = df_prophet.set_index("ds")["y"]
    predicted = forecast.set_index("ds")["yhat"]
    comparison = pd.concat([actual, predicted], axis=1).dropna()
    comparison.columns = ["actual", "predicted"]
    if comparison.empty:
        st.warning("No overlapping data to compare actual and predicted values.")
    else:
        mae = mean_absolute_error(comparison["actual"], comparison["predicted"])
        st.metric("Mean Absolute Error", f"{mae:,.2f}")

        csv = forecast[["ds", "yhat"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# Actual vs Predicted
st.subheader("📊 Actual vs Predicted Cases")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=comparison.index, y=comparison["actual"], name="Actual", line=dict(color="green")))
fig2.add_trace(go.Scatter(x=comparison.index, y=comparison["predicted"], name="Predicted", line=dict(color="red")))
fig2.update_layout(xaxis_title="Date", yaxis_title="Cases", height=500)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("## 📊 Trends & Comparisons")

df_trend = df[(df["location"] == country) & (df["year"] == selected_year)].sort_values("date")
df_trend["new_cases"] = df_trend["new_cases"].fillna(0)
df_trend["new_deaths"] = df_trend["new_deaths"].fillna(0)
df_trend["new_recovered"] = df_trend["new_recovered"] if "new_recovered" in df_trend else 0
df_trend["total_cases"] = df_trend["total_cases"].fillna(method="ffill").fillna(0)
df_trend["total_deaths"] = df_trend["total_deaths"].fillna(method="ffill").fillna(0)
df_trend["total_recovered"] = df_trend["total_recovered"] if "total_recovered" in df_trend else 0

df_trend["active_cases"] = (
    df_trend["total_cases"] - df_trend["total_deaths"]
)

# Optional Rt calculation
df_trend["Rt"] = df_trend["new_cases"] / df_trend["new_cases"].shift(7).replace(0, pd.NA)
df_trend["Rt"] = df_trend["Rt"].replace([float("inf"), -float("inf")], pd.NA).fillna(method='bfill')

# --- Plot 1: Daily New Cases ---
fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(x=df_trend["date"], y=df_trend["new_cases"], name="New Cases", marker_color="orange"))
fig_daily.update_layout(title="📊 Daily New Cases", barmode='group',
                        xaxis_title="Date", yaxis_title="Count", hovermode="x unified")
fig_daily.update_xaxes(rangeslider_visible=True)

# --- Plot 2: Cumulative Cases (Total/Deaths) ---
fig_cumulative = go.Figure()
fig_cumulative.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["total_cases"], mode="lines", name="Total Cases", line=dict(color="orange")))
fig_cumulative.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["total_deaths"], mode="lines", name="Deaths", line=dict(color="red")))
fig_cumulative.update_layout(title="📈 Cumulative Cases Over Time", xaxis_title="Date", yaxis_title="Count", hovermode="x unified")
fig_cumulative.update_xaxes(rangeslider_visible=True)

# --- Plot 3: Rt (Effective Reproduction Number) ---
fig_rt = go.Figure()
fig_rt.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["Rt"], mode="lines+markers", name="Rt", line=dict(color="blue")))
fig_rt.update_layout(title="📉 Effective Reproduction Number (Rt)", xaxis_title="Date", yaxis_title="Rt", hovermode="x unified")
fig_rt.update_xaxes(rangeslider_visible=True)

# Display Plots
st.plotly_chart(fig_daily, use_container_width=True)
st.plotly_chart(fig_cumulative, use_container_width=True)
st.plotly_chart(fig_rt, use_container_width=True)

st.markdown("## 🌐 Geographic Overview")

# Get latest data for all countries
df_latest = df.sort_values("date").groupby("location").last().reset_index()

# Clean data for mapping
world_df = df_latest[["location", "iso_code", "total_cases", "total_deaths", "total_vaccinations", "population"]].copy()
world_df["active_cases"] = world_df["total_cases"] - world_df["total_deaths"]
world_df["cases_per_100k"] = (world_df["active_cases"] / world_df["population"]) * 100000
world_df["vaccination_pct"] = (world_df["total_vaccinations"] / world_df["population"]) * 100

# Replace infinite or NaN values
world_df = world_df.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["cases_per_100k", "vaccination_pct", "iso_code"])

top_countries = world_df.nlargest(top_n, "cases_per_100k")
fig_bar = px.bar(top_countries, x="location", y="cases_per_100k", color="cases_per_100k",
                 title="Top Countries by Active Cases per 100k", labels={"cases_per_100k": "Cases per 100k"})
st.plotly_chart(fig_bar, use_container_width=True)

# Select map type
map_choice = st.radio("🗺️ Choose Map Type:", ["Active Cases per 100k", "Vaccination Coverage (%)"], horizontal=True)

if map_choice == "Active Cases per 100k":
    fig_map = go.Figure(data=go.Choropleth(
        locations=world_df["iso_code"],
        z=world_df["cases_per_100k"],
        text=world_df["location"],
        colorscale="OrRd",
        colorbar_title="Cases/100k",
        marker_line_color="white",
        marker_line_width=0.5,
    ))
    fig_map.update_layout(title_text="🌍 Active COVID Cases per 100,000 People", geo=dict(showframe=False, projection_type='natural earth'))

elif map_choice == "Vaccination Coverage (%)":
    fig_map = go.Figure(data=go.Choropleth(
        locations=world_df["iso_code"],
        z=world_df["vaccination_pct"],
        text=world_df["location"],
        colorscale="Greens",
        colorbar_title="Vaccination %",
        marker_line_color="white",
        marker_line_width=0.5,
    ))
    fig_map.update_layout(title_text="🌍 COVID Vaccination Coverage by Country", geo=dict(showframe=False, projection_type='natural earth'))

# Show map
st.plotly_chart(fig_map, use_container_width=True)







st.caption("🔍 Powered by Streamlit, Prophet & Our World in Data")