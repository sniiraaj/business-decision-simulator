import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Set Streamlit page layout
st.set_page_config(
    page_title="Business Decision Simulator",
    layout="wide",
)

# 1. Synthetic Data Generation
@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", periods=1095, freq="D")
    df = pd.DataFrame({"date": dates})

    df["price"] = np.random.uniform(40, 60, size=len(df)).round(2)
    df["ad_spend"] = np.random.uniform(100, 1000, size=len(df)).round(2)
    df["discount_pct"] = np.random.choice(
        [0.0, 0.05, 0.10, 0.15, 0.20], size=len(df), p=[0.4, 0.2, 0.2, 0.1, 0.1]
    )
    df["is_festival"] = df["date"].dt.month.isin([11, 12]).astype(int)

    day_of_year = df["date"].dt.dayofyear
    seasonality = np.sin(2 * np.pi * day_of_year / 365)

    base_demand = 500
    price_effect = -4.5 * df["price"]
    ad_effect = 15.0 * np.sqrt(df["ad_spend"])
    discount_effect = 300 * df["discount_pct"]
    festival_effect = 150 * df["is_festival"]
    seasonal_effect = 50 * seasonality
    noise = np.random.normal(0, 25, size=len(df))

    df["units_sold"] = (
        base_demand
        + price_effect
        + ad_effect
        + discount_effect
        + festival_effect
        + seasonal_effect
        + noise
    ).astype(int)

    df["units_sold"] = df["units_sold"].clip(lower=0)
    return df

df = generate_data()

# 2. ML Pipeline
@st.cache_resource
def train_model(data):
    X = data[["price", "ad_spend", "discount_pct", "is_festival"]]
    y = data["units_sold"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, r2, mae

model, r2, mae = train_model(df)

# 3. User Interface & Controls
st.title("📊 Business Decision Simulator & Analysis System")
st.markdown("Forecast sales and evaluate financial metrics across business decision scenarios.")

st.sidebar.header("🛠️ Scenario Parameters")
input_price = st.sidebar.slider("Selling Price ($)", 30.0, 80.0, 50.0, 0.5)
input_ad_spend = st.sidebar.slider("Daily Ad Spend ($)", 0.0, 2000.0, 500.0, 25.0)
input_discount = st.sidebar.select_slider("Discount (%)", options=[0.0, 0.05, 0.10, 0.15, 0.20], value=0.05)
input_festival = st.sidebar.selectbox("Festival Season?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
unit_cost = st.sidebar.number_input("Unit Production Cost ($)", 1.0, 50.0, 20.0, 1.0)

# 4. Simulation Engine
input_df = pd.DataFrame([{
    "price": input_price,
    "ad_spend": input_ad_spend,
    "discount_pct": input_discount,
    "is_festival": input_festival
}])

predicted_units = int(model.predict(input_df)[0])
effective_price = input_price * (1 - input_discount)
total_revenue = predicted_units * effective_price
production_cost = predicted_units * unit_cost
total_cost = production_cost + input_ad_spend
net_profit = total_revenue - total_cost
profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0.0
roi = (net_profit / total_cost * 100) if total_cost > 0 else 0.0

# 5. Display Outputs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Predicted Demand", f"{predicted_units:,} units")
col2.metric("Revenue", f"${total_revenue:,.2f}")
col3.metric("Total Cost", f"${total_cost:,.2f}")
col4.metric("Net Profit", f"${net_profit:,.2f}")
col5.metric("ROI", f"{roi:.1f}%")

st.markdown("---")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(["Revenue", "Production Cost", "Ad Spend", "Net Profit"], 
       [total_revenue, production_cost, input_ad_spend, net_profit],
       color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c" if net_profit >= 0 else "#d62728"])
ax.set_ylabel("USD ($)")
ax.set_title(f"Scenario Financial Structure (Model R²: {r2:.3f})")
st.pyplot(fig)