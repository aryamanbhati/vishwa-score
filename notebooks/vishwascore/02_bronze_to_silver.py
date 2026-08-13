# Databricks notebook source
# DBTITLE 1,Imports and Configuration
# Bronze to Silver Layer Transformation - VishwaScore Pipeline
# This script cleans, categorizes, and extracts persona from raw bank statement data

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DateType, StringType, IntegerType
from pyspark.sql.window import Window
from datetime import datetime

# Configuration
BRONZE_TABLE_PATH = "/Volumes/workspace/default/bank_statement_01/*.csv"
SILVER_TABLE_NAME = "workspace.default.silver_transactions_enriched"

print("✓ Imports loaded successfully")
print(f"Bronze Source: {BRONZE_TABLE_PATH}")
print(f"Silver Target: {SILVER_TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Bronze to Silver - Data Cleaning and Categorization
# Step 1: Read Bronze Layer (Raw CSV Data)
print("Reading Bronze layer data...")
df_bronze = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(BRONZE_TABLE_PATH)

print(f"Bronze records loaded: {df_bronze.count():,}")
print(f"Bronze schema: {df_bronze.columns}")

# Step 2: Data Cleaning and Type Casting
print("\nCleaning and casting data types...")
df_clean = df_bronze \
    .withColumn("date", F.to_date(F.col("date"), "dd-MM-yyyy")) \
    .withColumn("clean_withdrawal", 
                F.regexp_replace(F.col("withdrawal"), ",", "").cast(FloatType())) \
    .withColumn("clean_deposit", 
                F.regexp_replace(F.col("deposit"), ",", "").cast(FloatType())) \
    .withColumn("clean_balance", 
                F.regexp_replace(F.col("balance"), ",", "").cast(FloatType())) \
    .withColumn("particulars_lower", F.lower(F.col("particulars"))) \
    .withColumn("ingestion_timestamp", F.current_timestamp())

# Step 3: Enhanced Category Extraction - Matching Synthetic Data Generation Patterns
print("Applying enhanced categorization engine...")
df_categorized = df_clean.withColumn(
    "category",
    # Food Delivery
    F.when(F.col("particulars_lower").rlike(".*(zomato|swiggy|dominos|domino|mcdonald|kfc|pizza hut|burger king).*"), "Food_Delivery")
    # Quick Commerce / Grocery Delivery
    .when(F.col("particulars_lower").rlike(".*(blinkit|zepto|dunzo|instamart|bigbasket|big basket|grofers).*"), "Quick_Commerce")
    # Physical Grocery Stores
    .when(F.col("particulars_lower").rlike(".*(dmart|d mart|big bazaar|bigbazaar|more store|reliance fresh|kirana|grocery).*"), "Grocery_Store")
    # Telecom
    .when(F.col("particulars_lower").rlike(".*(recharge|jio|airtel|vi|vodafone|bsnl|mobile|prepaid|postpaid).*"), "Telecom_Expense")
    # Electricity
    .when(F.col("particulars_lower").rlike(".*(bescom|msedcl|tpddl|uppcl|electricity|power|light bill|eb payment|energy bill).*"), "Utility_Electricity")
    # Rent
    .when(F.col("particulars_lower").rlike(".*(rent|house rent|monthly rent|landlord).*"), "Rent_Expense")
    # Fuel
    .when(F.col("particulars_lower").rlike(".*(hpcl|indian oil|iocl|bpcl|bharat petro|petrol|diesel|fuel).*"), "Fuel_Expense")
    # Ride Sharing
    .when(F.col("particulars_lower").rlike(".*(uber|ola|rapido|cab|taxi|auto|ride).*"), "Transportation")
    # E-commerce
    .when(F.col("particulars_lower").rlike(".*(amazon|flipkart|myntra|meesho|ajio|snapdeal|ecommerce).*"), "Ecommerce_Shopping")
    # EMI / Loan
    .when(F.col("particulars_lower").rlike(".*(emi|loan|hdfc loan|sbi loan|bajaj finance|home credit|lendenclub|lending club|nach|repayment).*"), "EMI_Loan_Payment")
    # SHG (Self Help Group)
    .when(F.col("particulars_lower").rlike(".*(shg|self help|women group|micro credit).*"), "SHG_Contribution")
    # SIP / Mutual Fund
    .when(F.col("particulars_lower").rlike(".*(sip|mutual fund|starmf|groww|zerodha|coin|investment).*"), "SIP_Investment")
    # Insurance
    .when(F.col("particulars_lower").rlike(".*(lic premium|lic|hdfc life|sbi life|max life|icici pru|insurance premium|policy).*"), "Insurance_Premium")
    # Agriculture
    .when(F.col("particulars_lower").rlike(".*(fertilizer|agri store|krishi|seed|pesticide|farm input).*"), "Agriculture_Input")
    # Tractor EMI
    .when(F.col("particulars_lower").rlike(".*(tractor|mahindra finance tractor|farm equipment).*"), "Tractor_EMI")
    # ATM Withdrawal
    .when(F.col("particulars_lower").rlike(".*(atm|cash withdrawal|atm wd|cash wd).*"), "ATM_Cash_Withdrawal")
    # Local Market
    .when(F.col("particulars_lower").rlike(".*(local market|market|vendor|sabzi|vegetable).*"), "Local_Market")
    # LPG Gas
    .when(F.col("particulars_lower").rlike(".*(lpg|gas|hp gas|indane|bharat gas).*"), "LPG_Gas")
    # Healthcare
    .when(F.col("particulars_lower").rlike(".*(hospital|doctor|clinic|pharmacy|medical|apollo|fortis|medicine).*"), "Healthcare_Medical")
    # Education
    .when(F.col("particulars_lower").rlike(".*(school|college|university|education|course|fees|tuition).*"), "Education")
    # SMS Charges
    .when(F.col("particulars_lower").rlike(".*(sms charges|sms|quarterly charges).*"), "Bank_Charges")
    # Bounce Charges (Negative Signal)
    .when(F.col("particulars_lower").rlike(".*(ach return|bounce|insuff|insufficient funds|cheque return).*"), "Payment_Bounce")
    # UPI Generic
    .when(F.col("particulars_lower").rlike(".*(upi|paytm|phonepe|google pay|gpay|amazonpay|bhim).*"), "UPI_Payment")
    # INCOME CATEGORIES
    .when(F.col("particulars_lower").rlike(".*(salary|payroll|income credit|employer|wages).*"), "Salary_Income")
    .when(F.col("particulars_lower").rlike(".*(bundl|swiggy.*payout|zomato.*payout|urban company|porter|rapido.*payout|delivery partner|gig).*"), "Gig_Income")
    .when(F.col("particulars_lower").rlike(".*(dbt.*pm.*kisan|pm-kisan|kisan samman|pmkisan).*"), "Govt_DBT_PMKisan")
    .when(F.col("particulars_lower").rlike(".*(enam|mandi|crop sale|agricultural sale).*"), "Mandi_Sale_Income")
    .when(F.col("particulars_lower").rlike(".*(mgnrega|nrega|rural employment).*"), "MGNREGA_Income")
    .when(F.col("particulars_lower").rlike(".*(shg dividend|shg credit|group dividend).*"), "SHG_Dividend_Income")
    .when(F.col("particulars_lower").rlike(".*(int\\.pd|interest|fd interest|savings interest|dividend).*"), "Interest_Dividend_Income")
    .when(F.col("particulars_lower").rlike(".*(cust payment|customer|business deposit|p2p|peer to peer|personal transfer).*"), "Business_P2P_Income")
    .otherwise("Other")
)

# Step 4: Transaction Type Classification
df_categorized = df_categorized.withColumn(
    "transaction_type",
    F.when(F.col("clean_withdrawal").isNotNull(), "Debit")
    .when(F.col("clean_deposit").isNotNull(), "Credit")
    .otherwise("Unknown")
)

# Step 5: Extract Merchant/Vendor Name
df_categorized = df_categorized.withColumn(
    "merchant",
    F.regexp_extract(F.col("particulars"), "(UPI[A-Z]*-[A-Z0-9]+|[A-Z]{3,})", 0)
)

print(f"\nCategorization complete. Category distribution:")
df_categorized.groupBy("category").count().orderBy(F.desc("count")).show(20, truncate=False)

# Display sample cleaned data
print("\nSample cleaned and categorized transactions:")
display(df_categorized.select(
    "user_id", "date", "particulars", "category", 
    "clean_withdrawal", "clean_deposit", "transaction_type"
).limit(20))

# COMMAND ----------

# DBTITLE 1,Persona Extraction Logic
# Step 6: Persona Extraction Based on Transaction Patterns
# Matching the 6 personas from synthetic data generation script
print("Extracting user personas...")

# Create aggregated persona features per user
df_persona_features = df_categorized.groupBy("user_id").agg(
    # === SHG_Woman Indicators ===
    F.sum(F.when(F.col("category") == "SHG_Contribution", 1).otherwise(0)).alias("shg_txn_count"),
    F.sum(F.when(F.col("category") == "SHG_Dividend_Income", 1).otherwise(0)).alias("shg_income_count"),
    F.sum(F.when(F.col("category") == "Local_Market", 1).otherwise(0)).alias("market_txn_count"),
    F.sum(F.when(F.col("category") == "LPG_Gas", 1).otherwise(0)).alias("gas_txn_count"),
    
    # === Gig_Worker Indicators ===
    F.sum(F.when(F.col("category") == "Gig_Income", 1).otherwise(0)).alias("gig_income_count"),
    F.sum(F.when(F.col("category") == "Gig_Income", F.col("clean_deposit")).otherwise(0)).alias("total_gig_income"),
    F.sum(F.when(F.col("category") == "Food_Delivery", 1).otherwise(0)).alias("food_delivery_count"),
    F.sum(F.when(F.col("category") == "Quick_Commerce", 1).otherwise(0)).alias("quickcom_count"),
    F.sum(F.when(F.col("category") == "Transportation", 1).otherwise(0)).alias("ride_txn_count"),
    F.sum(F.when(F.col("category") == "Fuel_Expense", 1).otherwise(0)).alias("fuel_txn_count"),
    F.sum(F.when(F.col("category") == "Rent_Expense", 1).otherwise(0)).alias("rent_payment_count"),
    
    # === Salaried Indicators ===
    F.sum(F.when(F.col("category") == "Salary_Income", 1).otherwise(0)).alias("salary_credit_count"),
    F.sum(F.when(F.col("category") == "Salary_Income", F.col("clean_deposit")).otherwise(0)).alias("total_salary"),
    F.sum(F.when(F.col("category") == "SIP_Investment", 1).otherwise(0)).alias("sip_count"),
    F.sum(F.when(F.col("category") == "Ecommerce_Shopping", 1).otherwise(0)).alias("ecommerce_count"),
    F.sum(F.when(F.col("category") == "Interest_Dividend_Income", 1).otherwise(0)).alias("fd_interest_count"),
    
    # === Farmer Indicators ===
    F.sum(F.when(F.col("category") == "Govt_DBT_PMKisan", 1).otherwise(0)).alias("pmkisan_count"),
    F.sum(F.when(F.col("category") == "Mandi_Sale_Income", 1).otherwise(0)).alias("mandi_sale_count"),
    F.sum(F.when(F.col("category") == "MGNREGA_Income", 1).otherwise(0)).alias("mgnrega_count"),
    F.sum(F.when(F.col("category") == "Agriculture_Input", 1).otherwise(0)).alias("fertilizer_count"),
    F.sum(F.when(F.col("category") == "Tractor_EMI", 1).otherwise(0)).alias("tractor_emi_count"),
    
    # === Kirana_Owner Indicators ===
    F.sum(F.when(F.col("category") == "Business_P2P_Income", 1).otherwise(0)).alias("business_deposit_count"),
    F.sum(F.when(F.col("category") == "Business_P2P_Income", F.col("clean_deposit")).otherwise(0)).alias("total_business_income"),
    
    # === Street_Vendor Indicators ===
    F.sum(F.when(F.col("category") == "ATM_Cash_Withdrawal", 1).otherwise(0)).alias("atm_txn_count"),
    
    # === General Financial Discipline Indicators ===
    F.sum(F.when(F.col("category") == "Insurance_Premium", 1).otherwise(0)).alias("insurance_payment_count"),
    F.sum(F.when(F.col("category") == "EMI_Loan_Payment", 1).otherwise(0)).alias("emi_payment_count"),
    F.sum(F.when(F.col("category") == "Payment_Bounce", 1).otherwise(0)).alias("bounce_count"),
    
    # === General Activity ===
    F.count("*").alias("total_transactions"),
    F.countDistinct("category").alias("category_diversity"),
    F.sum(F.when(F.col("transaction_type") == "Credit", F.col("clean_deposit")).otherwise(0)).alias("total_credits"),
    F.sum(F.when(F.col("transaction_type") == "Debit", F.col("clean_withdrawal")).otherwise(0)).alias("total_debits")
)

# Apply Persona Tagging Logic - Hierarchical matching based on strongest signals
df_persona = df_persona_features.withColumn(
    "persona",
    # 1. Farmer (strongest signal: PM-Kisan DBT or Mandi sales)
    F.when((F.col("pmkisan_count") >= 1) | (F.col("mandi_sale_count") >= 1) | 
           ((F.col("mgnrega_count") >= 2) & (F.col("tractor_emi_count") >= 1)), "Farmer")
    
    # 2. Gig_Worker (gig income payouts)
    .when((F.col("gig_income_count") >= 3) & (F.col("total_gig_income") > 5000), "Gig_Worker")
    
    # 3. Salaried (regular salary credits)
    .when((F.col("salary_credit_count") >= 3) & (F.col("total_salary") > 10000), "Salaried_Employee")
    
    # 4. Kirana_Owner (high business deposits with market transactions)
    .when((F.col("business_deposit_count") >= 10) & (F.col("total_business_income") > 15000), "Kirana_Owner")
    
    # 5. SHG_Woman (SHG transactions or dividend income)
    .when((F.col("shg_txn_count") >= 1) | (F.col("shg_income_count") >= 1), "SHG_Woman")
    
    # 6. Street_Vendor (high ATM usage, low digital payments, market transactions)
    .when((F.col("atm_txn_count") >= 8) & (F.col("market_txn_count") >= 3) & 
          (F.col("total_credits") < 20000), "Street_Vendor")
    
    # 7. Financially Disciplined (has SIP + Insurance + EMI with no bounces)
    .when((F.col("sip_count") >= 2) & (F.col("insurance_payment_count") >= 2) & 
          (F.col("emi_payment_count") >= 3) & (F.col("bounce_count") == 0), "Financially_Disciplined")
    
    # 8. High Risk (payment bounces)
    .when(F.col("bounce_count") >= 1, "High_Risk_Bounces")
    
    # 9. Diverse Spender (high category diversity)
    .when(F.col("category_diversity") >= 10, "Diverse_Spender")
    
    # 10. Default
    .otherwise("Casual_User")
)

print("\nPersona Distribution:")
df_persona.groupBy("persona").count().orderBy(F.desc("count")).show(truncate=False)

# Join persona back to main categorized data
df_silver = df_categorized.join(df_persona.select("user_id", "persona"), on="user_id", how="left")

print(f"\nSilver layer prepared with {df_silver.count():,} records")
print("\nSample enriched data with persona:")
display(df_silver.select(
    "user_id", "persona", "date", "category", 
    "clean_withdrawal", "clean_deposit", "particulars"
).limit(20))

# COMMAND ----------

# DBTITLE 1,Data Quality Checks and Write to Silver Layer
# Step 7: Data Quality Checks
print("Running data quality checks...")

# Check for null critical fields
null_checks = df_silver.select(
    F.sum(F.when(F.col("user_id").isNull(), 1).otherwise(0)).alias("null_user_id"),
    F.sum(F.when(F.col("date").isNull(), 1).otherwise(0)).alias("null_date"),
    F.sum(F.when((F.col("clean_withdrawal").isNull()) & (F.col("clean_deposit").isNull()), 1).otherwise(0)).alias("null_both_amounts")
).collect()[0]

print(f"\nData Quality Report:")
print(f"  - Null user_id: {null_checks['null_user_id']}")
print(f"  - Null dates: {null_checks['null_date']}")
print(f"  - Missing amounts: {null_checks['null_both_amounts']}")

# Remove duplicates based on user_id, date, and particulars
df_silver_dedup = df_silver.dropDuplicates(["user_id", "date", "particulars"])
duplicates_removed = df_silver.count() - df_silver_dedup.count()
print(f"\nDuplicates removed: {duplicates_removed:,}")

# Step 8: Write to Silver Layer (Delta Table)
print(f"\nWriting to Silver layer: {SILVER_TABLE_NAME}")
df_silver_dedup.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE_NAME)

print(f"\u2713 Silver layer successfully created!")
print(f"  - Total records: {df_silver_dedup.count():,}")
print(f"  - Unique users: {df_silver_dedup.select('user_id').distinct().count():,}")
print(f"  - Date range: {df_silver_dedup.agg(F.min('date'), F.max('date')).collect()[0]}")

# Step 9: Create optimized indexes for faster queries
print("\nOptimizing Delta table...")
spark.sql(f"OPTIMIZE {SILVER_TABLE_NAME} ZORDER BY (user_id, date)")
print("✓ Table optimized with Z-ORDER")

# Final verification
print("\n=== SILVER LAYER SUMMARY ===")
spark.sql(f"DESCRIBE EXTENDED {SILVER_TABLE_NAME}").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,BONUS: Delta Live Tables (DLT) Version for Continuous Pipeline
# ============================================================================
# BONUS: Delta Live Tables (DLT) Declarative Pipeline Version
# ============================================================================
# Uncomment and use this code if you want to convert this notebook to a 
# Lakeflow Declarative Pipeline (DLT) for continuous, automated processing
# ============================================================================

"""
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

# Bronze Layer: Auto Loader for Continuous Ingestion
@dlt.table(
    name="bronze_bank_statements",
    comment="Raw bank statement data ingested via Auto Loader"
)
def bronze_bank_statements():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .load("/Volumes/workspace/default/bank_statement_01/")
    )

# Silver Layer: Cleaned and Categorized Transactions
@dlt.table(
    name="silver_transactions_enriched",
    comment="Cleaned, categorized transactions with persona extraction"
)
@dlt.expect_or_drop("valid_user_id", "user_id IS NOT NULL")
@dlt.expect_or_drop("valid_date", "date IS NOT NULL")
def silver_transactions_enriched():
    df = dlt.read_stream("bronze_bank_statements")
    
    # Data cleaning and categorization (same logic as above)
    df_clean = df \
        .withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd")) \
        .withColumn("clean_withdrawal", F.regexp_replace(F.col("withdrawal"), ",", "").cast(FloatType())) \
        .withColumn("clean_deposit", F.regexp_replace(F.col("deposit"), ",", "").cast(FloatType())) \
        .withColumn("particulars_lower", F.lower(F.col("particulars")))
    
    # Apply categorization
    df_categorized = df_clean.withColumn(
        "category",
        F.when(F.col("particulars_lower").rlike(".*(recharge|jio|airtel).*"), "Telecom_Expense")
        .when(F.col("particulars_lower").rlike(".*(bescom|electricity).*"), "Utility_Electricity")
        .when(F.col("particulars_lower").rlike(".*(zomato|swiggy).*"), "Digital_Food_Grocery")
        .when(F.col("particulars_lower").rlike(".*(lic|insurance).*"), "Insurance_Premium")
        .when(F.col("particulars_lower").rlike(".*(pm.*kisan).*"), "Govt_Subsidy_Income")
        .otherwise("Other")
    )
    
    return df_categorized

# To deploy this as a DLT pipeline:
# 1. Create a new DLT pipeline in Databricks UI
# 2. Point it to this notebook
# 3. Configure source path and target catalog/schema
# 4. Enable continuous mode for real-time processing
"""

print("✓ DLT pipeline template available (commented out)")
print("  To use: Uncomment the code above and create a DLT pipeline")
print("  Benefits: Automatic orchestration, data quality checks, lineage tracking")