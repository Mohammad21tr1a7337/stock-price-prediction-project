import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# Title
st.title("📈 Multi-Company Stock Forecasting")
st.markdown("Upload a dataset with multiple companies and forecast future closing prices.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Basic validation
    required_cols = {'date', 'close', 'Name'}
    if not required_cols.issubset(df.columns):
        st.error("Your dataset must contain 'date', 'close', and 'Name' columns.")
    else:
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        # Company selection
        companies = df['Name'].unique()
        selected_company = st.selectbox("Select a company", companies)

        # Forecast horizon
        forecast_days = st.slider("Select forecast horizon (days)", min_value=1, max_value=90, value=30)

        # Filter data
        company_df = df[df['Name'] == selected_company].copy()
        company_df.set_index('date', inplace=True)
        ts = company_df['close'].sort_index()

        # Use last 2 years for training
        last_date = ts.index[-1]
        two_years_ago = last_date - pd.DateOffset(years=2)
        ts_recent = ts[ts.index >= two_years_ago]

        # Fit Holt’s model
        model = ExponentialSmoothing(ts_recent, trend='add', seasonal=None)
        model_fit = model.fit(optimized=True)

        # Forecast
        forecast_values = model_fit.forecast(forecast_days)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'Forecast': forecast_values.values}, index=future_dates)

        # Display forecasted values
        st.subheader(f"📊 Forecasted Closing Prices for {selected_company} (Next {forecast_days} Days)")
        st.dataframe(forecast_df.style.format("{:.2f}"))

        # Plot
        st.subheader("📉 Forecast Visualization")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(ts_recent, label='Last 2 Years Close Price')
        ax.plot(forecast_df, label=f'{forecast_days}-Day Forecast', linestyle='--', color='red')
        ax.set_title(f'{selected_company} Closing Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
