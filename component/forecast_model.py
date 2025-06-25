import pandas as pd
from prophet import Prophet
import streamlit as st

@st.cache_resource
def train_model(df, changepoint_prior_scale):
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    model.add_regressor("vaccinations")
    model.add_regressor("gdp")
    model.add_regressor("health_index")
    model.fit(df)
    return model