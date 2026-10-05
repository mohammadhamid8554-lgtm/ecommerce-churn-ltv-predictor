import os

import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Churn & LTV Intelligence",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce Customer Churn & LTV Prediction Dashboard")
st.markdown("Enterprise decision engine for customer retention and RFM risk scoring.")

# Sidebar Navigation & Endpoint Config
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "FastAPI Endpoint URL",
    value=os.getenv(
        "API_URL",
        "https://ecommerce-churn-ltv-predictor.onrender.com/predict",
    ),
)

st.sidebar.markdown("---")
st.sidebar.info("Adjust the customer metrics below to evaluate churn risk in real-time.")

# Main Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer RFM Metrics")
    recency = st.number_input("Recency (Days since last order)", min_value=1, max_value=365, value=120)
    frequency = st.number_input("Frequency (Total orders count)", min_value=1, max_value=100, value=2)

with col2:
    st.subheader("💰 Customer Value Metrics")
    monetary = st.number_input("Monetary (Total spend $)", min_value=1.0, max_value=100000.0, value=150.0)
    avg_order_value = st.number_input("Average Order Value ($)", min_value=1.0, max_value=10000.0, value=75.0)

st.markdown("---")

# Trigger Button
if st.button("🚀 Analyze Customer Churn Risk", use_container_width=True):
    payload = {
        "recency": float(recency),
        "frequency": float(frequency),
        "monetary": float(monetary),
        "avg_order_value": float(avg_order_value)
    }

    try:
        # Request inference from FastAPI service
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")
            prob = result.get("churn_probability_percentage")

            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Status", prediction)
            m2.metric("Churn Risk Probability", f"{prob}%")
            m3.metric("Retention Priority", "HIGH" if prob > 70 else "NORMAL")

            # Risk Alert Banner
            if result.get("churn_class") == 1:
                st.error(f"⚠️ High Risk Alert: Customer has a {prob}% probability of churning. Recommend targeted retention campaigns.")
            else:
                st.success(f"✅ Active Customer: Customer has a {prob}% probability of churning. Account is healthy.")

        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"Failed to connect to FastAPI service at {api_url}. Error: {e}")