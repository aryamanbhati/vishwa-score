# Install required packages
%pip install datasets huggingface_hub --quiet

from pyspark.sql import SparkSession
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
import requests
import io

spark = SparkSession.builder.getOrCreate()

# ── Setup ────────────────────────────────────────
import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

spark.sql("CREATE CATALOG IF NOT EXISTS arthasetu")
spark.sql("CREATE SCHEMA IF NOT EXISTS arthasetu.bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS arthasetu.silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS arthasetu.gold")
spark.sql("CREATE VOLUME IF NOT EXISTS arthasetu.bronze.uploads")
print("✅ Catalog ready")

login(token=HF_TOKEN)
print("✅ HuggingFace logged in")

# ── Helper ───────────────────────────────────────
def save_bronze(df, table):
    if isinstance(df, pd.DataFrame):
        sdf = spark.createDataFrame(df.astype(str))
    else:
        sdf = df
    sdf.write.format("delta").mode("overwrite") \
        .option("delta.enableChangeDataFeed", "true") \
        .saveAsTable(f"arthasetu.bronze.{table}")
    cnt = spark.sql(f"SELECT count(*) as c FROM arthasetu.bronze.{table}").collect()[0]['c']
    print(f"  ✅ arthasetu.bronze.{table} → {cnt:,} rows")

# ── Dataset 1: Government Schemes ────────────────
print("\n[1/4] Government Schemes...")
schemes = [
    {"scheme_id":"SVA001","scheme_name":"PM SVANidhi Tier 1","ministry":"Ministry of Housing","max_loan_inr":10000,"interest_pct":7.0,"collateral":"No","eligibility":"Street vendor before March 2020, vending certificate or ULB letter, Aadhaar linked","target":"street vendor hawker thela wala rehri wala","description":"Working capital loan for urban street vendors without collateral. 7 percent interest subsidy on timely repayment.","source":"pmsvanidhi.mohua.gov.in"},
    {"scheme_id":"SVA002","scheme_name":"PM SVANidhi Tier 2","ministry":"Ministry of Housing","max_loan_inr":20000,"interest_pct":7.0,"collateral":"No","eligibility":"Fully repaid Tier 1 loan, active UPI transactions, Aadhaar linked","target":"street vendor who repaid first loan","description":"Enhanced loan for vendors with good repayment. Continued interest subsidy.","source":"pmsvanidhi.mohua.gov.in"},
    {"scheme_id":"SVA003","scheme_name":"PM SVANidhi Tier 3","ministry":"Ministry of Housing","max_loan_inr":50000,"interest_pct":7.0,"collateral":"No","eligibility":"Fully repaid Tier 1 and Tier 2 loans, consistent digital payment history","target":"experienced street vendor proven repayment","description":"Highest tier for vendors with excellent track record. Up to Rs 50000.","source":"pmsvanidhi.mohua.gov.in"},
    {"scheme_id":"MUD001","scheme_name":"PMMY Mudra Shishu","ministry":"Ministry of Finance","max_loan_inr":50000,"interest_pct":10.0,"collateral":"No","eligibility":"No prior default, basic KYC Aadhaar PAN, new or existing business","target":"micro enterprise vegetable vendor small shopkeeper artisan","description":"Micro business loan up to Rs 50000. No collateral needed. For new and small businesses.","source":"mudra.org.in"},
    {"scheme_id":"MUD002","scheme_name":"PMMY Mudra Kishor","ministry":"Ministry of Finance","max_loan_inr":500000,"interest_pct":12.0,"collateral":"Partial","eligibility":"1 year business vintage, bank statement, Aadhaar linked account","target":"growing small business trader manufacturer service provider","description":"Loan Rs 50000 to Rs 5 lakh for growing businesses.","source":"mudra.org.in"},
    {"scheme_id":"MUD003","scheme_name":"PMMY Mudra Tarun","ministry":"Ministry of Finance","max_loan_inr":1000000,"interest_pct":14.0,"collateral":"Yes","eligibility":"2 years business vintage, audited financials, good repayment history","target":"established medium enterprise exporter manufacturer","description":"Loan Rs 5 lakh to Rs 10 lakh for established businesses.","source":"mudra.org.in"},
    {"scheme_id":"KCC001","scheme_name":"Kisan Credit Card Crop Loan","ministry":"Ministry of Agriculture","max_loan_inr":300000,"interest_pct":7.0,"collateral":"Land records","eligibility":"Own or lease agricultural land, Aadhaar linked, land records verified","target":"farmer crop production seeds fertilizer pesticide","description":"Flexible credit for farmers for crop inputs. Repayment after harvest. 3 percent subvention.","source":"pmkisan.gov.in"},
    {"scheme_id":"KCC002","scheme_name":"Kisan Credit Card Allied Activities","ministry":"Ministry of Agriculture","max_loan_inr":200000,"interest_pct":7.0,"collateral":"Land records","eligibility":"Farmer with dairy poultry fishery activity, land or lease records, Aadhaar","target":"farmer dairy poultry fishery animal husbandry","description":"Credit for dairy, poultry, fishery along with farming. Same low interest rate.","source":"pmkisan.gov.in"},
    {"scheme_id":"NAB001","scheme_name":"NABARD Dairy Development Scheme","ministry":"NABARD","max_loan_inr":700000,"interest_pct":8.5,"collateral":"Yes","eligibility":"Business plan for dairy unit, land for cattle shed, minimum 2 animals","target":"rural entrepreneur dairy cattle milk production","description":"Capital subsidy and credit for setting up dairy farms. 25 percent subsidy general, 33 percent SC/ST.","source":"nabard.org"},
    {"scheme_id":"SHG001","scheme_name":"Self Help Group Bank Linkage","ministry":"NABARD","max_loan_inr":1000000,"interest_pct":7.0,"collateral":"No","eligibility":"Active SHG 6 months, regular savings, meeting minutes, 10 to 20 women members","target":"women self help group rural microfinance","description":"Women SHGs access collective loans without individual collateral. Low interest.","source":"nabard.org"},
    {"scheme_id":"VIS001","scheme_name":"PM Vishwakarma Tier 1","ministry":"Ministry of MSME","max_loan_inr":100000,"interest_pct":5.0,"collateral":"No","eligibility":"Registered on Vishwakarma portal, practicing traditional trade, Aadhaar, age 18 to 60","target":"artisan blacksmith carpenter potter tailor weaver cobbler barber","description":"Collateral-free loan at 5 percent for traditional artisans. Includes skill training.","source":"pmvishwakarma.gov.in"},
    {"scheme_id":"VIS002","scheme_name":"PM Vishwakarma Tier 2","ministry":"Ministry of MSME","max_loan_inr":200000,"interest_pct":5.0,"collateral":"No","eligibility":"Repaid Tier 1, completed skill training, Aadhaar verified","target":"experienced artisan business expansion craft","description":"Up to Rs 2 lakh for artisans who repaid first loan. Business expansion support.","source":"pmvishwakarma.gov.in"},
]
save_bronze(pd.DataFrame(schemes), "gov_schemes_raw")

# ── Dataset 2: BhashaBench Finance ───────────────
print("\n[2/4] BhashaBench Finance Q&A...")
try:
    ds_hi = load_dataset("bharatgenai/BhashaBench-Finance", data_dir="Hindi", split="test", token=True)
    df_hi = ds_hi.to_pandas()
    df_hi["lang"] = "hi"

    ds_en = load_dataset("bharatgenai/BhashaBench-Finance", data_dir="English", split="test", token=True)
    df_en = ds_en.to_pandas()
    df_en["lang"] = "en"

    df_bbf = pd.concat([df_hi, df_en], ignore_index=True)
    print(f"  Downloaded: {len(df_bbf):,} rows | Columns: {df_bbf.columns.tolist()}")
    save_bronze(df_bbf, "bhashbench_finance_raw")
except Exception as e:
    print(f"  ❌ BhashaBench failed: {e}")
    print("  Check HuggingFace token and dataset access")

# ── Dataset 3: Rural Loan from Volume ────────────
print("\n[3/4] Rural India Loan Dataset...")
try:
    sdf = spark.read.csv(
        "/Volumes/arthasetu/bronze/uploads/rural_india_loan.csv",
        header=True, inferSchema=True
    )
    print(f"  Rows: {sdf.count():,} | Columns: {sdf.columns}")
    save_bronze(sdf, "rural_loan_raw")
except Exception as e:
    print(f"  ❌ Rural loan failed: {e}")
    print("  Make sure CSV is uploaded to Volume")

# ── Dataset 4: PMMY State Data ───────────────────
print("\n[4/4] PMMY State Data...")
df_pmmy = pd.DataFrame([
    {"state":"Uttar Pradesh","shishu_accounts":4521000,"shishu_cr":12543,"kishor_accounts":892000,"kishor_cr":18234,"tarun_accounts":124000,"tarun_cr":9876,"credit_penetration":"Low"},
    {"state":"Bihar","shishu_accounts":3214000,"shishu_cr":8932,"kishor_accounts":654000,"kishor_cr":13421,"tarun_accounts":89000,"tarun_cr":7234,"credit_penetration":"Low"},
    {"state":"Maharashtra","shishu_accounts":2987000,"shishu_cr":9876,"kishor_accounts":743000,"kishor_cr":15678,"tarun_accounts":213000,"tarun_cr":18934,"credit_penetration":"High"},
    {"state":"Tamil Nadu","shishu_accounts":2543000,"shishu_cr":7654,"kishor_accounts":612000,"kishor_cr":13456,"tarun_accounts":187000,"tarun_cr":15678,"credit_penetration":"High"},
    {"state":"Madhya Pradesh","shishu_accounts":2134000,"shishu_cr":5678,"kishor_accounts":432000,"kishor_cr":8932,"tarun_accounts":76000,"tarun_cr":5432,"credit_penetration":"Medium"},
    {"state":"Rajasthan","shishu_accounts":1876000,"shishu_cr":4932,"kishor_accounts":387000,"kishor_cr":7654,"tarun_accounts":67000,"tarun_cr":4876,"credit_penetration":"Low"},
    {"state":"Karnataka","shishu_accounts":1987000,"shishu_cr":6543,"kishor_accounts":498000,"kishor_cr":11234,"tarun_accounts":156000,"tarun_cr":13456,"credit_penetration":"Medium"},
    {"state":"West Bengal","shishu_accounts":2876000,"shishu_cr":7891,"kishor_accounts":567000,"kishor_cr":12345,"tarun_accounts":98000,"tarun_cr":8765,"credit_penetration":"Medium"},
    {"state":"Gujarat","shishu_accounts":1654000,"shishu_cr":5432,"kishor_accounts":423000,"kishor_cr":9876,"tarun_accounts":134000,"tarun_cr":11234,"credit_penetration":"Medium"},
    {"state":"Odisha","shishu_accounts":1234000,"shishu_cr":3456,"kishor_accounts":287000,"kishor_cr":6543,"tarun_accounts":45000,"tarun_cr":3456,"credit_penetration":"Low"},
    {"state":"Andhra Pradesh","shishu_accounts":1543000,"shishu_cr":4321,"kishor_accounts":321000,"kishor_cr":7654,"tarun_accounts":87000,"tarun_cr":6543,"credit_penetration":"Medium"},
    {"state":"Telangana","shishu_accounts":1234000,"shishu_cr":3456,"kishor_accounts":298000,"kishor_cr":6789,"tarun_accounts":76000,"tarun_cr":5432,"credit_penetration":"Medium"},
])
save_bronze(df_pmmy, "pmmy_state_raw")

# ── Final check ──────────────────────────────────
print("\n" + "="*50)
print("BRONZE VERIFICATION")
print("="*50)
for t in ["gov_schemes_raw","bhashbench_finance_raw","rural_loan_raw","pmmy_state_raw"]:
    try:
        cnt = spark.sql(f"SELECT count(*) as c FROM arthasetu.bronze.{t}").collect()[0]['c']
        print(f"  ✅ {t:35s} {cnt:>8,} rows")
    except:
        print(f"  ❌ {t:35s} MISSING")
print("="*50)
