import streamlit as st
import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ----------------------------
# Reading Files
# ----------------------------
df = pd.read_csv('E:\Analysis\Pyhton Scripts\Forecasting Model\Data\Combined Data.csv')


# ----------------------------
# Changing Data Type
# ----------------------------
df['Date'] = pd.to_datetime(df['Date'])

df['Volume'].loc[df['Volume']==' -   '] = 0
df['Volume'].loc[df['Volume']==' - '] = 0
df['Volume'] = df['Volume'].astype(float)

df['Value'].loc[df['Value']==' $-   '] = 0
df['Value'].loc[df['Value']==' - '] = 0
df['Value'] = df['Value'].astype(float)


# ----------------------------
# Aggregating Dataframe
# ----------------------------

def prepare_data(df):
    df_grouped = df.groupby('Date').agg({'Volume': 'sum'}).reset_index()
    df_grouped.columns = ['ds', 'y']
    return df_grouped.sort_values('ds')


# ----------------------------
# Forecasting Function
# ----------------------------
def forecast_order(df, days):
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days, freq='D')
    forecast = model.predict(future)
    return model, forecast


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="FutureLens", layout="wide")
st.title("📊 A forecasting application developed using fitted data models.")

df = df

# Sidebar filters
st.sidebar.header("🔎 Filters")
factories = st.sidebar.multiselect("Select Factory", sorted(df['Factory'].dropna().unique()))
rbos = st.sidebar.multiselect("Select RBO", sorted(df['RBO'].dropna().unique()))
products = st.sidebar.multiselect("Select Product", sorted(df['Product'].dropna().unique()))

# Filter data
filtered_df = df.copy()
if factories:
    filtered_df = filtered_df[filtered_df['Factory'].isin(factories)]
if rbos:
    filtered_df = filtered_df[filtered_df['RBO'].isin(rbos)]
if products:
    filtered_df = filtered_df[filtered_df['Product'].isin(products)]

# Forecast period input
forecast_days = st.number_input("📅 Forecast Period (in days)", min_value=7, max_value=365, value=90, step=10)

# Forecasting
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    df_prepared = prepare_data(filtered_df)
    model, forecast = forecast_order(df_prepared, forecast_days)

    # Plot forecast
    st.subheader("📈 Forecast Chart")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Growth/Decline Table
    st.subheader("📊 Forecast Growth/Decline Summary")
    forecast_filtered = forecast[['ds', 'yhat']].tail(forecast_days).copy()
    forecast_filtered['Month'] = forecast_filtered['ds'].dt.to_period('M')
    forecast_monthly = forecast_filtered.groupby('Month').agg({'yhat': 'sum'}).reset_index()
    forecast_monthly['Growth %'] = forecast_monthly['yhat'].pct_change().fillna(0) * 100
    forecast_monthly['Growth %'] = forecast_monthly['Growth %'].round(2)

    st.dataframe(forecast_monthly.rename(columns={
        'Month': 'Forecast Month',
        'yhat': 'Predicted Value',
        'Growth %': 'Growth/Decline (%)'
    }))

    st.markdown("""
    - **Growth/Decline (%)** compares the forecasted value to the previous month.
    - Positive values = 📈 Growth, Negative values = 📉 Decline.
    """)