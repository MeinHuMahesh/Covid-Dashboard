import pandas as pd
import streamlit as st
@st.cache_data
def load_data():
    # url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv("owid-covid-data.csv")
    df["date"]=pd.to_datetime(df["date"],dayfirst=True)
    df["year"] = df["date"].dt.year
    return df