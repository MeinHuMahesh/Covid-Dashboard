import pandas as pd
import streamlit as st

def show_summary(df_country_all):
    latest = df_country_all.iloc[-1]
    previous = df_country_all.iloc[-2]

    def safe_int(val):
        return int(val) if pd.notna(val) else 0

    total_cases = safe_int(latest.get("total_cases"))
    total_deaths = safe_int(latest.get("total_deaths"))
    total_recovered = safe_int(latest.get("total_recovered")) if "total_recovered" in latest else 0
    active_cases = total_cases - total_deaths - total_recovered
    vaccinations = safe_int(latest.get("total_vaccinations"))
    tests = safe_int(latest.get("total_tests")) if "total_tests" in latest else 0

    delta_cases = total_cases - safe_int(previous.get("total_cases"))
    delta_deaths = total_deaths - safe_int(previous.get("total_deaths"))
    delta_recovered = (
        total_recovered - safe_int(previous.get("total_recovered"))
        if "total_recovered" in latest else 0
    )
    prev_active = (
        safe_int(previous.get("total_cases")) - safe_int(previous.get("total_deaths")) -
        (safe_int(previous.get("total_recovered")) if "total_recovered" in previous else 0)
    )
    delta_active = active_cases - prev_active
    delta_vaccinations = vaccinations - safe_int(previous.get("total_vaccinations"))
    delta_tests = tests - safe_int(previous.get("total_tests")) if "total_tests" in latest else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🨠 Total Cases", f"{total_cases:,}", f"{delta_cases:+,}")
    c2.metric("🔴 Active Cases", f"{active_cases:,}", f"{delta_active:+,}")
    c3.metric("💚 Recovered", f"{total_recovered:,}", f"{delta_recovered:+,}")
    c4.metric("⚰️ Total Deaths", f"{total_deaths:,}", f"{delta_deaths:+,}")
    c5.metric("💉 Vaccinations", f"{vaccinations:,}", f"{delta_vaccinations:+,}")
    c6.metric("🧪 Tests Conducted", f"{tests:,}", f"{delta_tests:+,}")