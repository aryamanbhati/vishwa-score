# Databricks notebook source
# DBTITLE 1,Configuration and Imports
# ============================================================================
# GOLD LAYER: Feature Engineering for VishwaScore ML Model
# ============================================================================
# Aggregates Silver transaction data into user-level features (35+ features)
# Output: One row per user with all features ready for ML training
# ============================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, DoubleType
from datetime import datetime, timedelta

# Configuration
SILVER_TABLE_NAME = "workspace.default.silver_transactions_enriched"
GOLD_TABLE_NAME = "workspace.default.gold_vishwascore_features"

print("="*70)
print("  GOLD LAYER: Feature Engineering for VishwaScore")
print("="*70)
print(f"Source: {SILVER_TABLE_NAME}")
print(f"Target: {GOLD_TABLE_NAME}")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Read Silver Layer and Calculate Monthly Aggregates
# Step 1: Read Silver Layer
print("Reading Silver layer data...")
df_silver = spark.read.table(SILVER_TABLE_NAME)

print(f"Silver records loaded: {df_silver.count():,}")
print(f"Unique users: {df_silver.select('user_id').distinct().count():,}")

# Add month column for monthly aggregations
df_silver = df_silver.withColumn("month", F.date_trunc("month", F.col("date")))

print("\nSample Silver data:")
display(df_silver.select("user_id", "persona", "date", "month", "category", "clean_withdrawal", "clean_deposit").limit(10))

# COMMAND ----------

# DBTITLE 1,Payment Behaviour Features (25% weight)
# ============================================================================
# PAYMENT BEHAVIOUR FEATURES (25% Weight in VishwaScore)
# ============================================================================
print("Extracting Payment Behaviour features...")

df_payment_features = df_silver.groupBy("user_id").agg(
    # Utility Bill Payments
    F.sum(F.when(F.col("category") == "Utility_Electricity", F.col("clean_withdrawal")).otherwise(0)).alias("total_utility_spend"),
    F.avg(F.when(F.col("category") == "Utility_Electricity", F.col("clean_withdrawal"))).alias("avg_utility_bill"),
    F.count(F.when(F.col("category") == "Utility_Electricity", 1)).alias("utility_payment_count"),
    
    # EMI Regularity Score
    F.count(F.when(F.col("category") == "EMI_Loan_Payment", 1)).alias("emi_payment_count"),
    F.sum(F.when(F.col("category") == "EMI_Loan_Payment", F.col("clean_withdrawal")).otherwise(0)).alias("total_emi_paid"),
    F.countDistinct(F.when(F.col("category") == "EMI_Loan_Payment", F.col("month"))).alias("emi_months_active"),
    
    # Insurance Premium Payments
    F.count(F.when(F.col("category") == "Insurance_Premium", 1)).alias("insurance_payment_count"),
    F.sum(F.when(F.col("category") == "Insurance_Premium", F.col("clean_withdrawal")).otherwise(0)).alias("total_insurance_premium"),
    
    # Rent Payments
    F.count(F.when(F.col("category") == "Rent_Expense", 1)).alias("rent_payment_count"),
    F.avg(F.when(F.col("category") == "Rent_Expense", F.col("clean_withdrawal"))).alias("avg_rent_amount"),
    
    # SIP/Investment Discipline
    F.count(F.when(F.col("category") == "SIP_Investment", 1)).alias("sip_payment_count"),
    F.sum(F.when(F.col("category") == "SIP_Investment", F.col("clean_withdrawal")).otherwise(0)).alias("total_sip_invested"),
    
    # Negative Signals
    F.count(F.when(F.col("category") == "Payment_Bounce", 1)).alias("bounce_count"),
    F.sum(F.when(F.col("category") == "Payment_Bounce", F.col("clean_withdrawal")).otherwise(0)).alias("total_bounce_charges")
)

# Calculate EMI Regularity Score (0-1 scale)
df_payment_features = df_payment_features.withColumn(
    "emi_regularity_score",
    F.when(F.col("emi_payment_count") > 0,
           F.col("emi_months_active") / F.greatest(F.lit(1), F.col("emi_payment_count"))).otherwise(0)
)

# Calculate Insurance Flag (Binary: 1 if paid, 0 otherwise)
df_payment_features = df_payment_features.withColumn(
    "insurance_paid_flag",
    F.when(F.col("insurance_payment_count") > 0, 1).otherwise(0)
)

print("Payment Behaviour features extracted:")
print(f"  - Utility payment features: 3")
print(f"  - EMI regularity features: 4")
print(f"  - Insurance features: 2")
print(f"  - Rent features: 2")
print(f"  - Investment features: 2")
print(f"  - Risk signals: 2")

display(df_payment_features.limit(10))

# COMMAND ----------

# DBTITLE 1,UPI and Digital Flow Features (40% weight)
# ============================================================================
# UPI & DIGITAL FLOW FEATURES (40% Weight in VishwaScore)
# ============================================================================
print("Extracting UPI & Digital Flow features...")

df_digital_features = df_silver.groupBy("user_id").agg(
    # Transaction Volume & Frequency
    F.count("*").alias("total_transactions"),
    F.countDistinct("date").alias("active_days"),
    F.countDistinct("month").alias("active_months"),
    
    # Average Transaction Size
    F.avg(F.when(F.col("transaction_type") == "Debit", F.col("clean_withdrawal"))).alias("avg_debit_txn_size"),
    F.avg(F.when(F.col("transaction_type") == "Credit", F.col("clean_deposit"))).alias("avg_credit_txn_size"),
    
    # Transaction Frequency per Month
    F.count("*").alias("total_txn_for_freq_calc"),  # Will divide by active_months later
    
    # Merchant/Vendor Diversity
    F.countDistinct(F.when(F.col("merchant").isNotNull(), F.col("merchant"))).alias("unique_merchants"),
    F.countDistinct("category").alias("unique_categories"),
    
    # Digital Payment Adoption
    F.count(F.when(F.col("category").isin("UPI_Payment", "Food_Delivery", "Quick_Commerce", "Ecommerce_Shopping"), 1)).alias("digital_txn_count"),
    F.sum(F.when(F.col("category").isin("UPI_Payment", "Food_Delivery", "Quick_Commerce"), F.col("clean_withdrawal")).otherwise(0)).alias("total_digital_spend"),
    
    # Cash Usage (ATM withdrawals)
    F.count(F.when(F.col("category") == "ATM_Cash_Withdrawal", 1)).alias("atm_withdrawal_count"),
    F.sum(F.when(F.col("category") == "ATM_Cash_Withdrawal", F.col("clean_withdrawal")).otherwise(0)).alias("total_cash_withdrawn"),
    
    # Savings Ratio (Credits / Debits)
    F.sum(F.when(F.col("transaction_type") == "Credit", F.col("clean_deposit")).otherwise(0)).alias("total_credits"),
    F.sum(F.when(F.col("transaction_type") == "Debit", F.col("clean_withdrawal")).otherwise(0)).alias("total_debits"),
    
    # Category-wise spending
    F.sum(F.when(F.col("category") == "Food_Delivery", F.col("clean_withdrawal")).otherwise(0)).alias("food_delivery_spend"),
    F.sum(F.when(F.col("category") == "Grocery_Store", F.col("clean_withdrawal")).otherwise(0)).alias("grocery_spend"),
    F.sum(F.when(F.col("category") == "Ecommerce_Shopping", F.col("clean_withdrawal")).otherwise(0)).alias("ecommerce_spend"),
    F.sum(F.when(F.col("category") == "Transportation", F.col("clean_withdrawal")).otherwise(0)).alias("transportation_spend"),
    F.sum(F.when(F.col("category") == "Healthcare_Medical", F.col("clean_withdrawal")).otherwise(0)).alias("healthcare_spend"),
    F.sum(F.when(F.col("category") == "Education", F.col("clean_withdrawal")).otherwise(0)).alias("education_spend")
)

# Calculate derived metrics
df_digital_features = df_digital_features \
    .withColumn("txn_frequency_per_month", 
                F.col("total_txn_for_freq_calc") / F.greatest(F.lit(1), F.col("active_months"))) \
    .withColumn("merchant_diversity_score", 
                F.col("unique_merchants") + F.col("unique_categories")) \
    .withColumn("savings_ratio",
                F.when(F.col("total_debits") > 0, 
                       F.col("total_credits") / F.col("total_debits")).otherwise(0)) \
    .withColumn("digital_adoption_rate",
                F.when(F.col("total_transactions") > 0,
                       F.col("digital_txn_count") / F.col("total_transactions")).otherwise(0)) \
    .withColumn("cash_dependency_ratio",
                F.when(F.col("total_debits") > 0,
                       F.col("total_cash_withdrawn") / F.col("total_debits")).otherwise(0)) \
    .drop("total_txn_for_freq_calc")

print("UPI & Digital Flow features extracted:")
print(f"  - Transaction volume features: 3")
print(f"  - Transaction size features: 2")
print(f"  - Merchant diversity features: 3")
print(f"  - Digital adoption features: 3")
print(f"  - Savings features: 3")
print(f"  - Category spending features: 6")

display(df_digital_features.limit(10))

# COMMAND ----------

# DBTITLE 1,Income Stability Features (10% weight)
# ============================================================================
# INCOME STABILITY FEATURES (10% Weight in VishwaScore)
# ============================================================================
print("Extracting Income Stability features...")

# Monthly income aggregation
df_monthly_income = df_silver.filter(F.col("transaction_type") == "Credit") \
    .groupBy("user_id", "month") \
    .agg(
        F.sum("clean_deposit").alias("monthly_income"),
        F.count("*").alias("monthly_credit_count")
    )

# Calculate income stability metrics
df_income_features = df_monthly_income.groupBy("user_id").agg(
    # Average Monthly Income
    F.avg("monthly_income").alias("avg_monthly_income"),
    F.max("monthly_income").alias("max_monthly_income"),
    F.min("monthly_income").alias("min_monthly_income"),
    
    # Income Variance (Lower is better)
    F.stddev("monthly_income").alias("income_variance"),
    F.variance("monthly_income").alias("income_variance_squared"),
    
    # Months Active
    F.count("*").alias("months_with_income"),
    
    # Income Regularity
    F.countDistinct("month").alias("distinct_income_months")
)

# Calculate Coefficient of Variation (CV) for income stability
df_income_features = df_income_features.withColumn(
    "income_cv",
    F.when(F.col("avg_monthly_income") > 0,
           F.col("income_variance") / F.col("avg_monthly_income")).otherwise(0)
)

# Income stability score (0-1 scale, higher is better)
df_income_features = df_income_features.withColumn(
    "income_stability_score",
    F.when(F.col("income_cv") > 0, 1 / (1 + F.col("income_cv"))).otherwise(0)
)

# Specific income type features
df_income_type_features = df_silver.groupBy("user_id").agg(
    # Salary Income
    F.count(F.when(F.col("category") == "Salary_Income", 1)).alias("salary_credit_count"),
    F.sum(F.when(F.col("category") == "Salary_Income", F.col("clean_deposit")).otherwise(0)).alias("total_salary_income"),
    
    # Gig Income
    F.count(F.when(F.col("category") == "Gig_Income", 1)).alias("gig_credit_count"),
    F.sum(F.when(F.col("category") == "Gig_Income", F.col("clean_deposit")).otherwise(0)).alias("total_gig_income"),
    
    # Government Benefits
    F.count(F.when(F.col("category").isin("Govt_DBT_PMKisan", "MGNREGA_Income"), 1)).alias("govt_benefit_count"),
    F.sum(F.when(F.col("category").isin("Govt_DBT_PMKisan", "MGNREGA_Income"), F.col("clean_deposit")).otherwise(0)).alias("total_govt_benefits"),
    
    # Business/P2P Income
    F.count(F.when(F.col("category") == "Business_P2P_Income", 1)).alias("business_credit_count"),
    F.sum(F.when(F.col("category") == "Business_P2P_Income", F.col("clean_deposit")).otherwise(0)).alias("total_business_income"),
    
    # Investment Income
    F.sum(F.when(F.col("category") == "Interest_Dividend_Income", F.col("clean_deposit")).otherwise(0)).alias("total_investment_income")
)

# Join income features
df_income_all = df_income_features.join(df_income_type_features, on="user_id", how="left")

print("Income Stability features extracted:")
print(f"  - Income statistics: 3")
print(f"  - Income variance metrics: 3")
print(f"  - Income stability scores: 2")
print(f"  - Income type features: 5")

display(df_income_all.limit(10))

# COMMAND ----------

# DBTITLE 1,Persona and Risk Features
# ============================================================================
# PERSONA & RISK FEATURES
# ============================================================================
print("Extracting Persona and Risk features...")

df_persona_risk = df_silver.groupBy("user_id").agg(
    # Get persona (take the first one since all rows for a user have same persona)
    F.first("persona").alias("persona"),
    
    # Risk Signals
    F.count(F.when(F.col("category") == "Payment_Bounce", 1)).alias("bounce_count_risk"),
    F.sum(F.when(F.col("clean_balance") < 500, 1).otherwise(0)).alias("low_balance_days"),
    
    # Financial Discipline Signals
    F.count(F.when(F.col("category") == "SIP_Investment", 1)).alias("sip_count_risk"),
    F.count(F.when(F.col("category") == "Insurance_Premium", 1)).alias("insurance_count_risk"),
    
    # Date Range
    F.min("date").alias("first_transaction_date"),
    F.max("date").alias("last_transaction_date")
)

# Calculate account age in months
df_persona_risk = df_persona_risk.withColumn(
    "account_age_months",
    F.months_between(F.col("last_transaction_date"), F.col("first_transaction_date"))
)

# One-hot encode persona
df_persona_risk = df_persona_risk \
    .withColumn("is_farmer", F.when(F.col("persona") == "Farmer", 1).otherwise(0)) \
    .withColumn("is_gig_worker", F.when(F.col("persona") == "Gig_Worker", 1).otherwise(0)) \
    .withColumn("is_salaried", F.when(F.col("persona") == "Salaried_Employee", 1).otherwise(0)) \
    .withColumn("is_kirana_owner", F.when(F.col("persona") == "Kirana_Owner", 1).otherwise(0)) \
    .withColumn("is_shg_woman", F.when(F.col("persona") == "SHG_Woman", 1).otherwise(0)) \
    .withColumn("is_street_vendor", F.when(F.col("persona") == "Street_Vendor", 1).otherwise(0))

print("Persona & Risk features extracted:")
print(f"  - Persona one-hot encoding: 6")
print(f"  - Risk signals: 2")
print(f"  - Financial discipline: 2")
print(f"  - Account metrics: 1")

display(df_persona_risk.limit(10))

# COMMAND ----------

# DBTITLE 1,Join All Features and Write to Gold Layer
# ============================================================================
# JOIN ALL FEATURES & WRITE TO GOLD LAYER
# ============================================================================
print("Joining all feature sets...")

# Join all feature DataFrames
df_gold = df_payment_features \
    .join(df_digital_features, on="user_id", how="left") \
    .join(df_income_all, on="user_id", how="left") \
    .join(df_persona_risk, on="user_id", how="left")

# Fill nulls with 0 for numeric columns
numeric_cols = [field.name for field in df_gold.schema.fields 
                if field.name != "user_id" and field.name != "persona" 
                and field.name != "first_transaction_date" and field.name != "last_transaction_date"
                and (isinstance(field.dataType, (FloatType, DoubleType)) or str(field.dataType).startswith("long") or str(field.dataType).startswith("int"))]

df_gold = df_gold.fillna(0, subset=numeric_cols)

print(f"\nGold layer prepared with {df_gold.count():,} users")
print(f"Total features: {len(df_gold.columns) - 1}")  # Exclude user_id

# Show feature summary
print("\n=== GOLD LAYER FEATURE SUMMARY ===")
print(f"Total Features: {len(df_gold.columns)}")
print("\nSample features:")
display(df_gold.limit(10))

# Write to Gold Layer
print(f"\nWriting to Gold layer: {GOLD_TABLE_NAME}")
df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(GOLD_TABLE_NAME)

print(f"\n✓ Gold layer successfully created!")
print(f"  - Table: {GOLD_TABLE_NAME}")
print(f"  - Users: {df_gold.count():,}")
print(f"  - Features: {len(df_gold.columns)}")

# Optimize the table
print("\nOptimizing Gold table...")
spark.sql(f"OPTIMIZE {GOLD_TABLE_NAME} ZORDER BY (user_id)")
print("✓ Table optimized")

# COMMAND ----------

# DBTITLE 1,Feature Statistics and Validation
# ============================================================================
# FEATURE STATISTICS & VALIDATION
# ============================================================================
print("Generating feature statistics...")

# Read the Gold table
df_gold_check = spark.read.table(GOLD_TABLE_NAME)

# Feature statistics
print("\n=== FEATURE STATISTICS ===")
display(df_gold_check.select(
    "avg_monthly_income",
    "emi_regularity_score",
    "savings_ratio",
    "digital_adoption_rate",
    "merchant_diversity_score",
    "income_stability_score",
    "bounce_count",
    "persona"
).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max"))

# Persona distribution in Gold layer
print("\n=== PERSONA DISTRIBUTION ===")
display(df_gold_check.groupBy("persona").count().orderBy(F.desc("count")))

# Check for null values
print("\n=== NULL VALUE CHECK ===")
null_counts = df_gold_check.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_gold_check.columns])
display(null_counts)

print("\n" + "="*70)
print("  ✓ Gold Layer Feature Engineering Complete!")
print("  Ready for ML Model Training")
print("="*70)