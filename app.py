
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm

def prepare_features(df, n_lags=5):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Volatility'].shift(lag)
    df = df.dropna()
    return df

st.set_page_config(page_title="AI-Driven Derivatives Forecasting", layout="wide")
st.title("📈 AI-Driven Derivatives Forecasting & Risk Management")

# Sidebar
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
n_lags = st.sidebar.slider("Lag Features", 1, 10, 5)

@st.cache_data
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
    return df.dropna()


# ⬇️ Fetch data first
df = fetch_data(ticker, start_date, end_date)

# ⬇️ Then prepare features
df = prepare_features(df, n_lags)

X = df[[f'lag_{i}' for i in range(1, n_lags+1)]]
y = df['Volatility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Training
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
st.metric("📊 RMSE (Volatility Forecast)", f"{rmse:.4f}")

# Value at Risk
st.subheader("📉 Value at Risk (VaR)")
weights = np.ones(len(df)) / len(df)
portfolio_returns = df['Returns'].dropna()
mean = portfolio_returns.mean()
std = portfolio_returns.std()
VaR_95 = norm.ppf(0.05, mean, std)
VaR_99 = norm.ppf(0.01, mean, std)
st.write(f"**95% VaR (1-day): {-VaR_95:.2%}**")
st.write(f"**99% VaR (1-day): {-VaR_99:.2%}**")

# Option Pricing
st.subheader("💰 Black-Scholes Option Pricing")
S = df['Close'].iloc[-1]
sigma = df['Volatility'].iloc[-1]
K = S * 1.05
T = 30 / 252
r = 0.05

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

call_price = float(black_scholes_call(S, K, T, r, sigma))
st.write(f"**Call Option Price (30D @ +5% Strike): ${call_price:.2f}**")


# SHAP Interpretability
st.subheader("🧠 SHAP Feature Importance")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
fig, ax = plt.subplots(figsize=(10, 4))
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

