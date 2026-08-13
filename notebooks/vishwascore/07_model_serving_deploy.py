# Databricks notebook source
# DBTITLE 1,Model Serving — Deploy VishwaScore REST Endpoint
# ============================================================================
# Deploy the @Champion VishwaScore model as a Databricks Model Serving endpoint.
# This gives us:
#   - Real-time REST API (sub-100ms p95 on Small workload)
#   - Auto-scaling from 0 to thousands of QPS
#   - A/B traffic splitting between model versions
#   - Built-in monitoring (latency, throughput, error rate)
# ============================================================================

import requests
import json
import time

# ── Config ───────────────────────────────────────────────────────────────────
UC_MODEL_NAME = "workspace.default.vishwascore_model"
ENDPOINT_NAME = "vishwascore-scoring-api"

# Get workspace URL and token from the notebook context
workspace_url = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)
token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

print(f"Workspace: {workspace_url}")
print(f"Model:     {UC_MODEL_NAME}")
print(f"Endpoint:  {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Find the Champion Model Version
from mlflow.tracking import MlflowClient

client = MlflowClient()

champion_versions = client.get_model_version_by_alias(UC_MODEL_NAME, "Champion")
champion_version = champion_versions.version
print(f"Champion version: v{champion_version}")

# COMMAND ----------

# DBTITLE 1,Step 2 — Create the Serving Endpoint
endpoint_config = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_entities": [
            {
                "entity_name": UC_MODEL_NAME,
                "entity_version": str(champion_version),
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "auto_capture_config": {
            "catalog_name": "workspace",
            "schema_name": "default",
            "table_name_prefix": "vishwascore_serving",
        },
    },
}

print("Creating serving endpoint...")
print(json.dumps(endpoint_config, indent=2))

resp = requests.post(
    f"{workspace_url}/api/2.0/serving-endpoints",
    headers=headers,
    json=endpoint_config,
)

if resp.status_code == 200:
    print(f"\nEndpoint created: {ENDPOINT_NAME}")
elif resp.status_code == 409:
    print(f"\nEndpoint already exists — updating config...")
    resp = requests.put(
        f"{workspace_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
        headers=headers,
        json=endpoint_config["config"],
    )
    if resp.status_code == 200:
        print("Config updated.")
    else:
        print(f"Update failed: {resp.status_code} {resp.text}")
else:
    print(f"Create failed: {resp.status_code} {resp.text}")

# COMMAND ----------

# DBTITLE 1,Step 3 — Wait for Endpoint to Be Ready
print("Waiting for endpoint to become ready (this takes 5-10 min on first deploy)...")

for i in range(60):
    status_resp = requests.get(
        f"{workspace_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
        headers=headers,
    )
    state = status_resp.json().get("state", {})
    ready = state.get("ready", "NOT_READY")
    config_update = state.get("config_update", "NOT_UPDATING")
    print(f"  [{i * 10}s] ready={ready}, config_update={config_update}")

    if ready == "READY":
        print("\nEndpoint is READY.")
        break

    time.sleep(10)
else:
    print("\nTimed out after 10 min — check the Serving UI for status.")

# COMMAND ----------

# DBTITLE 1,Step 4 — Test the Endpoint with a Sample Scoring Request
sample_features = {
    "total_transactions": 289,
    "active_days": 156,
    "active_months": 10,
    "total_utility_spend": 3450.50,
    "avg_utility_bill": 345.05,
    "utility_payment_count": 10,
    "emi_payment_count": 9,
    "total_emi_paid": 45000.0,
    "emi_months_active": 9,
    "emi_regularity_score": 0.9,
    "insurance_payment_count": 2,
    "total_insurance_premium": 12000.0,
    "insurance_paid_flag": 1,
    "rent_payment_count": 10,
    "avg_rent_amount": 15000.0,
    "sip_payment_count": 10,
    "total_sip_invested": 50000.0,
    "bounce_count": 0,
    "total_bounce_charges": 0.0,
    "unique_merchants": 24,
    "unique_categories": 18,
    "digital_txn_count": 156,
    "total_digital_spend": 45000.0,
    "avg_debit_txn_size": 850.0,
    "avg_credit_txn_size": 52000.0,
    "atm_withdrawal_count": 12,
    "total_cash_withdrawn": 24000.0,
    "food_delivery_spend": 8500.0,
    "grocery_spend": 12000.0,
    "ecommerce_spend": 15000.0,
    "transportation_spend": 3500.0,
    "healthcare_spend": 2500.0,
    "education_spend": 5000.0,
    "telecom_spend": 2000.0,
    "total_credits": 520000.0,
    "total_debits": 495000.0,
    "avg_monthly_income": 52000.0,
    "max_monthly_income": 55000.0,
    "min_monthly_income": 48000.0,
    "income_variance": 2500000.0,
    "income_variance_squared": 6250000000000.0,
    "income_cv": 0.03,
    "income_stability_score": 0.97,
    "months_with_income": 10,
    "distinct_income_months": 10,
    "salary_credit_count": 10,
    "total_salary_income": 520000.0,
    "gig_credit_count": 0,
    "total_gig_income": 0.0,
    "govt_benefit_count": 0,
    "total_govt_benefits": 0.0,
    "business_credit_count": 0,
    "total_business_income": 0.0,
    "total_investment_income": 0.0,
    "account_age_months": 10.5,
    "txn_frequency_per_month": 28.9,
    "savings_ratio": 1.05,
    "merchant_diversity_score": 42,
    "digital_adoption_rate": 0.54,
    "cash_dependency_ratio": 0.048,
    "low_balance_days": 0,
    "bounce_count_risk": 0,
    "sip_count_risk": 0,
    "insurance_count_risk": 0,
    "is_farmer": 0,
    "is_gig_worker": 0,
    "is_salaried": 1,
    "is_kirana_owner": 0,
    "is_shg_woman": 0,
    "is_street_vendor": 0,
}

payload = {
    "dataframe_split": {
        "columns": list(sample_features.keys()),
        "data": [list(sample_features.values())],
    }
}

print("Scoring sample salaried user...")
score_resp = requests.post(
    f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers=headers,
    json=payload,
)

if score_resp.status_code == 200:
    result = score_resp.json()
    predicted_score = result["predictions"][0]
    print(f"\nVishwaScore prediction: {predicted_score:.0f} / 900")
    print(f"Endpoint URL: {workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")
else:
    print(f"Scoring failed: {score_resp.status_code}")
    print(score_resp.text)

# COMMAND ----------

# DBTITLE 1,Step 5 — Print Integration Snippet for Streamlit
print(f"""
# ── Streamlit / ArthaSetu integration snippet ────────────────────────────
import os, requests

DATABRICKS_HOST = "{workspace_url}"
ENDPOINT = "{ENDPOINT_NAME}"
TOKEN = os.environ["DATABRICKS_TOKEN"]

def score_user(features: dict) -> float:
    resp = requests.post(
        f"{{DATABRICKS_HOST}}/serving-endpoints/{{ENDPOINT}}/invocations",
        headers={{"Authorization": f"Bearer {{TOKEN}}", "Content-Type": "application/json"}},
        json={{"dataframe_split": {{"columns": list(features.keys()), "data": [list(features.values())]}}}},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()["predictions"][0]

# Usage:
# score = score_user({{"total_transactions": 289, "emi_regularity_score": 0.9, ...}})
# ─────────────────────────────────────────────────────────────────────────
""")
