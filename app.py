import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(
    page_title="Executive Business Suite", layout="wide", page_icon="💼"
)

# Initialize navigation page state
if "page" not in st.session_state:
    st.session_state.page = "Home"


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


# ==========================================
# SIDEBAR QUICK NAVIGATION
# ==========================================
st.sidebar.title("💼 Navigation Hub")
if st.sidebar.button("🏠 Home Dashboard", use_container_width=True):
    go_to("Home")

st.sidebar.markdown("---")
st.sidebar.caption("Quick Jump To:")
if st.sidebar.button("📊 1. Demand & Decision Simulator (ML)", use_container_width=True):
    go_to("Module 1")
if st.sidebar.button("⚖️ 2. Break-Even Units Calculator", use_container_width=True):
    go_to("Module 2")
if st.sidebar.button("🏦 3. Loan & Interest Calculator", use_container_width=True):
    go_to("Module 3")
if st.sidebar.button("🏷️ 4. Product Pricing Planner", use_container_width=True):
    go_to("Module 4")
if st.sidebar.button("📑 5. Simple Business P&L Modeler", use_container_width=True):
    go_to("Module 5")


# ==========================================
# PAGE 0: HOME DASHBOARD (5-BUTTON HUB)
# ==========================================
if st.session_state.page == "Home":
    st.title("💼 Executive Business & Financial Suite")
    st.markdown("Select a business module to open its dedicated workbench:")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("📊 Module 1: Demand & Decision Simulator (ML)")
            st.write(
                "Predict unit demand using a regression matrix engine and run live P&L scenarios."
            )
            if st.button("Launch ML Simulator 🚀", key="btn1", use_container_width=True, type="primary"):
                go_to("Module 1")

        with st.container(border=True):
            st.subheader("🏦 Module 3: Loan & Compound Interest Calculator")
            st.write(
                "Calculate monthly debt service payments and visualize loan payoff schedules."
            )
            if st.button("Launch Loan Calculator 🏦", key="btn3", use_container_width=True):
                go_to("Module 3")

        with st.container(border=True):
            st.subheader("📑 Module 5: Simple Business P&L Modeler")
            st.write(
                "Model sequential income statement math from gross sales down to net profit."
            )
            if st.button("Launch P&L Modeler 📑", key="btn5", use_container_width=True):
                go_to("Module 5")

    with col2:
        with st.container(border=True):
            st.subheader("⚖️ Module 2: Break-Even Units Calculator")
            st.write(
                "Determine the minimum sales volume required to cover fixed overhead expenses."
            )
            if st.button("Launch Break-Even Calculator ⚖️", key="btn2", use_container_width=True):
                go_to("Module 2")

        with st.container(border=True):
            st.subheader("🏷️ Module 4: Product Pricing & Margin Planner")
            st.write(
                "Calculate optimal selling prices and effective markups based on cost price and target margin %."
            )
            if st.button("Launch Pricing Planner 🏷️", key="btn4", use_container_width=True):
                go_to("Module 4")


# ==========================================
# SUB-MODULE PAGES (WITH TOP BACK BUTTON)
# ==========================================
else:
    # Universal Back Button Header
    top_col1, _ = st.columns([2, 8])
    with top_col1:
        if st.button("⬅️ Back to Main Hub", type="secondary", use_container_width=True):
            go_to("Home")

    st.markdown("---")

    # --------------------------------------
    # MODULE 1: ML DEMAND SIMULATOR
    # --------------------------------------
    if st.session_state.page == "Module 1":
        st.title("📊 Demand & Business Decision Simulator (ML)")
        st.markdown(
            "Forecast unit demand using an OLS matrix model and calculate real-time financial ROI."
        )

        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", periods=1095, freq="D")
        df = pd.DataFrame({"date": dates})
        df["price"] = np.random.uniform(40, 60, size=len(df)).round(2)
        df["ad_spend"] = np.random.uniform(100, 1000, size=len(df)).round(2)
        df["discount_pct"] = np.random.choice(
            [0.0, 0.05, 0.10, 0.15, 0.20], size=len(df), p=[0.4, 0.2, 0.2, 0.1, 0.1]
        )
        df["is_festival"] = df["date"].dt.month.isin([11, 12]).astype(int)

        base_demand = 500
        df["units_sold"] = (
            base_demand
            - (4.5 * df["price"])
            + (15.0 * np.sqrt(df["ad_spend"]))
            + (300 * df["discount_pct"])
            + (150 * df["is_festival"])
            + np.random.normal(0, 25, size=len(df))
        ).astype(int)
        df["units_sold"] = df["units_sold"].clip(lower=0)

        X = np.column_stack(
            [
                df["price"].values,
                np.sqrt(df["ad_spend"].values),
                df["discount_pct"].values,
                df["is_festival"].values,
            ]
        )
        y = df["units_sold"].values
        X_b = np.c_[np.ones((len(X), 1)), X]
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

        col1, col2 = st.columns(2)
        with col1:
            price = st.slider("Selling Price ($)", 30.0, 80.0, 50.0, 0.5)
            ad_spend = st.slider("Daily Ad Spend ($)", 0.0, 2000.0, 500.0, 25.0)
        with col2:
            discount = st.select_slider(
                "Discount Rate (%)",
                options=[0.0, 0.05, 0.10, 0.15, 0.20],
                value=0.05,
            )
            is_fest = st.selectbox(
                "Festival Season?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        unit_cost = st.number_input("Production Unit Cost ($)", 5.0, 50.0, 20.0, 1.0)

        user_x = np.array([[1, price, np.sqrt(ad_spend), discount, is_fest]])
        pred_units = int(max(0, np.dot(user_x, beta)[0]))

        effective_price = price * (1 - discount)
        total_revenue = pred_units * effective_price
        production_cost = pred_units * unit_cost
        total_cost = production_cost + ad_spend
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predicted Demand", f"{pred_units:,} units")
        c2.metric("Revenue", f"${total_revenue:,.2f}")
        c3.metric("Total Cost", f"${total_cost:,.2f}")
        c4.metric("Net Profit", f"${net_profit:,.2f}")
        c5.metric("ROI", f"{roi:.1f}%")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(
            ["Revenue", "Production Cost", "Ad Spend", "Net Profit"],
            [total_revenue, production_cost, ad_spend, net_profit],
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#d62728",
                "#2ca02c" if net_profit >= 0 else "#d62728",
            ],
        )
        ax.set_ylabel("USD ($)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig)

    # --------------------------------------
    # MODULE 2: BREAK-EVEN CALCULATOR
    # --------------------------------------
    elif st.session_state.page == "Module 2":
        st.title("⚖️ Break-Even Units Calculator")
        st.markdown(
            "Calculate the exact sales volume needed to cover fixed overhead expenses."
        )

        col1, col2, col3 = st.columns(3)
        fixed_costs = col1.number_input(
            "Fixed Overhead Costs ($)", value=10000.0, step=500.0
        )
        unit_price = col2.number_input(
            "Selling Price per Unit ($)", value=50.0, step=1.0
        )
        variable_cost = col3.number_input(
            "Variable Cost per Unit ($)", value=20.0, step=1.0
        )

        contrib_margin = unit_price - variable_cost

        if contrib_margin <= 0:
            st.error(
                "⚠️ Selling price must be greater than variable cost per unit to generate a positive contribution margin."
            )
        else:
            be_units = int(np.ceil(fixed_costs / contrib_margin))
            be_revenue = be_units * unit_price

            m1, m2, m3 = st.columns(3)
            m1.metric("Contribution Margin / Unit", f"${contrib_margin:,.2f}")
            m2.metric("Break-Even Volume", f"{be_units:,} units")
            m3.metric("Break-Even Revenue", f"${be_revenue:,.2f}")

            units = np.linspace(0, max(be_units * 2, 50), 100)
            rev = units * unit_price
            costs = fixed_costs + (units * variable_cost)

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(units, rev, label="Total Revenue", color="green", linewidth=2)
            ax.plot(units, costs, label="Total Cost", color="red", linewidth=2)
            ax.axvline(
                be_units,
                color="gray",
                linestyle="--",
                label=f"Break-Even ({be_units} units)",
            )
            ax.set_xlabel("Units Sold")
            ax.set_ylabel("USD ($)")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    # --------------------------------------
    # MODULE 3: LOAN CALCULATOR
    # --------------------------------------
    elif st.session_state.page == "Module 3":
        st.title("🏦 Loan & Compound Interest Calculator")
        st.markdown(
            "Calculate monthly debt service payments and analyze the principal payoff schedule."
        )

        col1, col2, col3 = st.columns(3)
        principal = col1.number_input(
            "Loan Principal ($)", value=50000.0, step=1000.0
        )
        annual_rate = col2.number_input(
            "Annual Interest Rate (%)", value=6.0, step=0.25
        )
        years = col3.slider("Loan Term (Years)", 1, 30, 5)

        n_payments = years * 12
        r_monthly = (annual_rate / 100) / 12

        if r_monthly > 0:
            monthly_pmt = (
                principal
                * (r_monthly * (1 + r_monthly) ** n_payments)
                / ((1 + r_monthly) ** n_payments - 1)
            )
        else:
            monthly_pmt = principal / n_payments

        total_paid = monthly_pmt * n_payments
        total_interest = total_paid - principal

        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly Payment", f"${monthly_pmt:,.2f}")
        m2.metric("Total Principal Paid", f"${principal:,.2f}")
        m3.metric("Total Interest Expense", f"${total_interest:,.2f}")

        months = np.arange(1, n_payments + 1)
        if r_monthly > 0:
            balance = [
                principal
                * (
                    ((1 + r_monthly) ** n_payments - (1 + r_monthly) ** m)
                    / ((1 + r_monthly) ** n_payments - 1)
                )
                for m in months
            ]
        else:
            balance = [principal - (monthly_pmt * m) for m in months]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(months, balance, color="purple", linewidth=2)
        ax.set_title("Remaining Principal Payoff Schedule")
        ax.set_xlabel("Months")
        ax.set_ylabel("Remaining Balance ($)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # --------------------------------------
    # MODULE 4: PRICING PLANNER
    # --------------------------------------
    elif st.session_state.page == "Module 4":
        st.title("🏷️ Product Pricing, Markup & Margin Planner")
        st.markdown(
            "Calculate recommended selling prices and markups based on unit cost price and target profit margin."
        )

        col1, col2 = st.columns(2)
        cost_price = col1.number_input(
            "Cost Price per Unit ($)", value=20.0, step=1.0
        )
        target_margin = col2.slider("Target Profit Margin (%)", 5.0, 80.0, 30.0, 1.0)

        selling_price = cost_price / (1 - (target_margin / 100))
        gross_profit = selling_price - cost_price
        markup_pct = (gross_profit / cost_price) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended Selling Price", f"${selling_price:,.2f}")
        m2.metric("Gross Profit per Unit", f"${gross_profit:,.2f}")
        m3.metric("Markup Percentage", f"{markup_pct:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.barh(["Selling Price"], [cost_price], label="Cost Price", color="#ff7f0e")
        ax.barh(
            ["Selling Price"],
            [gross_profit],
            left=[cost_price],
            label="Gross Profit",
            color="#2ca02c",
        )
        ax.set_xlabel("USD ($)")
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)

    # --------------------------------------
    # MODULE 5: P&L MODELER
    # --------------------------------------
    elif st.session_state.page == "Module 5":
        st.title("📑 Simple Business P&L Modeler")
        st.markdown(
            "Run top-to-bottom income statement math from gross revenue down to net profit."
        )

        col1, col2 = st.columns(2)
        with col1:
            revenue = st.number_input(
                "Total Sales Revenue ($)", value=100000.0, step=5000.0
            )
            cogs = st.number_input(
                "Cost of Goods Sold / COGS ($)", value=40000.0, step=2000.0
            )
        with col2:
            opex = st.number_input(
                "Operating Expenses / OpEx ($)", value=30000.0, step=2000.0
            )
            tax_rate = st.number_input("Tax Rate (%)", value=20.0, step=1.0)

        gross_profit = revenue - cogs
        gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0
        ebit = gross_profit - opex
        taxes = max(0.0, ebit * (tax_rate / 100))
        net_income = ebit - taxes
        net_margin = (net_income / revenue * 100) if revenue > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Gross Profit", f"${gross_profit:,.2f}", f"{gross_margin:.1f}% Margin"
        )
        m2.metric("Operating Profit (EBIT)", f"${ebit:,.2f}")
        m3.metric("Taxes", f"${taxes:,.2f}")
        m4.metric(
            "Net Income", f"${net_income:,.2f}", f"{net_margin:.1f}% Net Margin"
        )

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(
            ["Revenue", "COGS", "OpEx", "Taxes", "Net Income"],
            [revenue, cogs, opex, taxes, net_income],
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#d62728",
                "#9467bd",
                "#2ca02c" if net_income >= 0 else "#d62728",
            ],
        )
        ax.set_ylabel("USD ($)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig)
