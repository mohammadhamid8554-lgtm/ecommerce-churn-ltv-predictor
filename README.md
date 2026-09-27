
# 📊 End-to-End E-Commerce Customer Churn & LTV Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An enterprise-grade, production-ready Machine Learning system that predicts **30-day customer churn risk** and estimates **6-month Lifetime Value (LTV)**. Served via a low-latency FastAPI REST API and monitored with MLflow.

---

## 🎯 Executive Summary & Business Impact

In e-commerce, acquiring new customers costs **5x to 7x more** than retaining existing ones. High customer turnover directly degrades profitability.

* **Problem:** Traditional retention marketing relies on reactive, static segmentation—offering discounts *after* a user has already drifted away.
* **Solution:** This project delivers a **real-time ML pipeline** that continuously calculates churn probability based on customer purchase recency, frequency, monetary metrics (RFM), and engagement signals.
* **Business ROI:** Enables targeted retention strategies (e.g., dynamic 20% discount triggers for high-risk accounts), reducing simulated churn costs by **~18%** while preventing margin erosion on low-risk users.

---

## 🏗 System Architecture

```text
┌────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Raw Data &    │ ──► │  Feature Engineering │ ──► │  Imbalanced Pipeline│
│  Transactions  │     │   (RFM + Metrics)    │     │   (SMOTE + XGBoost) │
└────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                               │
┌────────────────┐     ┌──────────────────────┐                │
│ Streamlit UI   │ ◄── │ FastAPI REST Service │ ◄──────────────┘ Model Artifact
│ Control Center │     │  (/predict Endpoint) │                  (joblib)
└────────────────┘     └──────────────────────┘
