# Databricks notebook source
# DBTITLE 1,Feature Store Registration — Unity Catalog
# ============================================================================
# Register the Gold features table as a Unity Catalog Feature Store table.
# This gives us:
#   - Feature lineage (which models consume which features)
#   - Point-in-time lookups for training sets
#   - Online-serving compatibility (Feature Serving)
#   - Governance via Unity Catalog ACLs
# ============================================================================

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
GOLD_TABLE = "xscore.gold.credit_feature_store"
FEATURE_TABLE = "xscore.gold.vishwascore_credit_features"
UC_MODEL_NAME = "xscore.gold.credit_scorer"
EXPERIMENT = "/Users/aryamanbhati8@gmail.com/vishwascore_experiments"

fe = FeatureEngineeringClient()
mlflow.set_experiment(EXPERIMENT)

print("Feature Engineering client initialised")

# COMMAND ----------

# DBTITLE 1,Step 1 — Create the Feature Table from Gold
df_gold = spark.read.table(GOLD_TABLE)
row_count = df_gold.count()
col_count = len(df_gold.columns)
print(f"Gold table: {row_count:,} rows, {col_count} columns")

# Drop date columns that aren't features
feature_df = df_gold.drop("first_transaction_date", "last_transaction_date")

fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["user_id"],
    df=feature_df,
    description=(
        f"VishwaScore credit features — {col_count - 3} engineered features across "
        "4 categories (payment behaviour, digital flow, income stability, persona) "
        f"for {row_count:,} synthetic Indian borrowers."
    ),
)

print(f"Feature table created: {FEATURE_TABLE}")

# COMMAND ----------

# DBTITLE 1,Step 2 — Build a Training Set via Feature Lookups
# The label (vishwascore) lives in a separate labels table.
# In production you'd store labels separately; here we split from the same Gold
# table and demonstrate the FeatureLookup workflow.

labels_df = spark.read.table(GOLD_TABLE).select("user_id", "vishwascore_raw")

# If vishwascore_raw doesn't exist yet, fall back to computing it inline.
# (The 05_model_train notebook already creates it.)
if "vishwascore_raw" not in [c.name for c in spark.read.table(GOLD_TABLE).schema]:
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType
    _gold = spark.read.table(GOLD_TABLE)
    _gold = _gold.withColumn(
        "vishwascore",
        F.greatest(
            F.least(
                (F.coalesce(F.col("emi_regularity_score"), F.lit(0)) * 75)
                + (F.coalesce(F.col("insurance_paid_flag"), F.lit(0)) * 50)
                + (F.when(F.col("utility_payment_count") >= 3, 50).otherwise(0))
                + (F.when(F.col("bounce_count") == 0, 50).otherwise(-50))
                + (F.least(F.coalesce(F.col("digital_adoption_rate"), F.lit(0)) * 100, F.lit(100)))
                + (F.least(F.coalesce(F.col("merchant_diversity_score"), F.lit(0)) * 2, F.lit(80)))
                + (F.least(F.coalesce(F.col("savings_ratio"), F.lit(0)) * 100, F.lit(100)))
                + (F.least(F.coalesce(F.col("txn_frequency_per_month"), F.lit(0)), F.lit(80)))
                + (F.coalesce(F.col("income_stability_score"), F.lit(0)) * 90)
                + (F.col("is_salaried") * 80)
                + (F.col("is_farmer") * 60)
                + (F.col("is_gig_worker") * 70)
                + (F.col("is_kirana_owner") * 75)
                + (F.col("is_shg_woman") * 65)
                + (F.col("is_street_vendor") * 55)
                + (F.rand() * 50 - 25),
                F.lit(900),
            ),
            F.lit(300),
        ).cast(IntegerType()),
    )
    labels_df = _gold.select("user_id", F.col("vishwascore").alias("label"))
else:
    labels_df = spark.read.table(GOLD_TABLE).select(
        "user_id",
        F.col("vishwascore_raw").alias("label"),
    )

# Exclude non-feature columns from lookups
exclude = {"user_id", "persona", "vishwascore", "vishwascore_raw",
           "first_transaction_date", "last_transaction_date"}
feature_cols = [c for c in spark.read.table(FEATURE_TABLE).columns if c not in exclude]

feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        feature_names=feature_cols,
        lookup_key="user_id",
    )
]

training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="label",
    exclude_columns=[],
)

training_df = training_set.load_df()
print(f"Training set: {training_df.count():,} rows, {len(training_df.columns)} columns (including label)")

# COMMAND ----------

# DBTITLE 1,Step 3 — Train + Log Model with Feature Store Lineage
pdf = training_df.toPandas()
X = pdf[feature_cols].fillna(0).values
y = pdf["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = GradientBoostingRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    subsample=0.8, min_samples_split=10, min_samples_leaf=5,
    random_state=42,
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)

print(f"Test R²:   {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test MAE:  {test_mae:.2f}")

# COMMAND ----------

# DBTITLE 1,Step 4 — Log to Feature Store + Register in UC
with mlflow.start_run(run_name="vishwascore_feature_store_v1") as run:
    mlflow.log_params({
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
        "subsample": 0.8, "feature_count": len(feature_cols),
    })
    mlflow.log_metrics({
        "test_r2": test_r2, "test_rmse": test_rmse, "test_mae": test_mae,
    })

    # fe.log_model links the model to the Feature Table in UC lineage
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=UC_MODEL_NAME,
    )

    # Also log the scaler so serving can use it
    mlflow.sklearn.log_model(scaler, artifact_path="scaler")

    print(f"Model logged with Feature Store lineage → {UC_MODEL_NAME}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Set Champion Alias
from mlflow.tracking import MlflowClient

client = MlflowClient()
latest = client.get_registered_model(UC_MODEL_NAME)
latest_version = max(int(v.version) for v in latest.latest_versions)

client.set_registered_model_alias(UC_MODEL_NAME, "Champion", str(latest_version))
print(f"Set @Champion alias → v{latest_version}")

# COMMAND ----------

# DBTITLE 1,Step 6 — Verify Lineage
print("Feature table details:")
ft = fe.get_table(FEATURE_TABLE)
print(f"  Name:         {ft.name}")
print(f"  Primary keys: {ft.primary_keys}")
print(f"  Description:  {ft.description[:120]}...")

print(f"\nModel {UC_MODEL_NAME} now has Feature Store lineage.")
print("View in UC: Catalog → xscore → gold → vishwascore_credit_features → Lineage tab")
