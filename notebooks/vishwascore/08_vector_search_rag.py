# Databricks notebook source
# DBTITLE 1,Vector Search — Managed RAG for ArthaSetu
# ============================================================================
# Replace the local FAISS index with Databricks Vector Search.
# This gives us:
#   - Managed delta-sync: the index auto-refreshes when the source table changes
#   - Foundation Model embeddings (databricks-gte-large-en) — no self-hosted model
#   - Unity Catalog governance on the index
#   - REST API for retrieval (usable from the voice pipeline or Streamlit)
# ============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
import json

# ── Config ───────────────────────────────────────────────────────────────────
CATALOG = "arthasetu"
SCHEMA = "gold"
CORPUS_TABLE = f"{CATALOG}.{SCHEMA}.scheme_corpus"
VS_ENDPOINT = "arthasetu_vs"
VS_INDEX = f"{CATALOG}.{SCHEMA}.scheme_index"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Corpus table: {CORPUS_TABLE}")
print(f"VS endpoint:  {VS_ENDPOINT}")
print(f"VS index:     {VS_INDEX}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Create the Scheme Corpus Delta Table
# The 11 government loan schemes that ArthaSetu recommends.
# Previously these lived as an inline string in voice_advisor.py;
# now they're a proper Delta table that Vector Search can sync from.

schemes = [
    {
        "scheme_id": 1,
        "scheme_name": "PM SVANidhi Tier 1",
        "content_type": "scheme",
        "target_group": "street_vendor",
        "state": "All India",
        "content": (
            "PM SVANidhi Tier 1: Working capital loan for street vendors up to Rs 10,000. "
            "No collateral or guarantor required. 7% annual interest subsidy on timely repayment. "
            "Eligibility: Must have a vending certificate or letter of recommendation from "
            "the Town Vending Committee. Repayment: 12 monthly instalments. "
            "Apply at any bank, NBFC-MFI, or through the PM SVANidhi portal."
        ),
    },
    {
        "scheme_id": 2,
        "scheme_name": "PM SVANidhi Tier 2",
        "content_type": "scheme",
        "target_group": "street_vendor",
        "state": "All India",
        "content": (
            "PM SVANidhi Tier 2: Enhanced loan of Rs 20,000 for street vendors who fully "
            "repaid Tier 1. Same 7% interest subsidy. Digital transaction cashback of up to "
            "Rs 1,200/year for vendors who use digital payments. "
            "Repayment: 18 monthly instalments."
        ),
    },
    {
        "scheme_id": 3,
        "scheme_name": "PM SVANidhi Tier 3",
        "content_type": "scheme",
        "target_group": "street_vendor",
        "state": "All India",
        "content": (
            "PM SVANidhi Tier 3: Highest tier loan of Rs 50,000 for vendors with excellent "
            "repayment track record (completed Tier 1 + Tier 2). 7% interest subsidy continues. "
            "Designed for vendors who want to scale up their business. "
            "Repayment: 18-24 monthly instalments."
        ),
    },
    {
        "scheme_id": 4,
        "scheme_name": "PMMY Mudra Shishu",
        "content_type": "scheme",
        "target_group": "micro_business",
        "state": "All India",
        "content": (
            "Pradhan Mantri Mudra Yojana — Shishu category: Micro business loan up to Rs 50,000 "
            "for non-farm small businesses. No collateral needed. Covers small shops, street vendors, "
            "artisans, food stall operators, repair services. Interest rate: 10-12% (varies by bank). "
            "Apply at any scheduled commercial bank, RRB, or MFI."
        ),
    },
    {
        "scheme_id": 5,
        "scheme_name": "PMMY Mudra Kishor",
        "content_type": "scheme",
        "target_group": "small_business",
        "state": "All India",
        "content": (
            "Pradhan Mantri Mudra Yojana — Kishor category: Business loan from Rs 50,000 to "
            "Rs 5,00,000 for growing enterprises. Requires at least 1 year of business history. "
            "Suitable for expanding inventory, buying equipment, or adding employees. "
            "Collateral may be required for amounts above Rs 2 lakh."
        ),
    },
    {
        "scheme_id": 6,
        "scheme_name": "PMMY Mudra Tarun",
        "content_type": "scheme",
        "target_group": "established_business",
        "state": "All India",
        "content": (
            "Pradhan Mantri Mudra Yojana — Tarun category: Business loan from Rs 5 lakh to "
            "Rs 10 lakh for established businesses looking to scale up significantly. "
            "Requires strong business track record and financial documentation. "
            "Collateral and income proof typically required."
        ),
    },
    {
        "scheme_id": 7,
        "scheme_name": "Kisan Credit Card",
        "content_type": "scheme",
        "target_group": "farmer",
        "state": "All India",
        "content": (
            "Kisan Credit Card (KCC): Agricultural credit up to Rs 3,00,000 at 7% interest "
            "(effective 4% with timely repayment subsidy under Interest Subvention Scheme). "
            "Covers crop cultivation, post-harvest expenses, dairy, fishery, and animal husbandry. "
            "Eligibility: All farmers — owner cultivators, tenant farmers, sharecroppers, "
            "oral lessees, SHGs, and joint liability groups. Apply at any bank branch."
        ),
    },
    {
        "scheme_id": 8,
        "scheme_name": "NABARD Dairy Entrepreneurship Scheme",
        "content_type": "scheme",
        "target_group": "farmer",
        "state": "All India",
        "content": (
            "NABARD Dairy Entrepreneurship Development Scheme: Loans up to Rs 7,00,000 for "
            "setting up small dairy farms (2-10 animals), milk processing, and dairy product units. "
            "Capital subsidy: 25% of project cost (33.33% for SC/ST and North-Eastern states). "
            "Maximum subsidy: Rs 1.75 lakh for general, Rs 2.33 lakh for SC/ST. "
            "Apply through NABARD-affiliated commercial banks or RRBs."
        ),
    },
    {
        "scheme_id": 9,
        "scheme_name": "SHG Bank Linkage Programme",
        "content_type": "scheme",
        "target_group": "shg_woman",
        "state": "All India",
        "content": (
            "Self-Help Group Bank Linkage Programme: Women's SHGs (10-20 members) can access "
            "collective loans up to Rs 10,00,000 without individual collateral. Interest rates "
            "typically 10-12%. The group's internal savings record (minimum 6 months) serves as "
            "collateral substitute. SHGs can lend to members at group-decided rates. "
            "Facilitated by NABARD, NRLM, and state livelihoods missions."
        ),
    },
    {
        "scheme_id": 10,
        "scheme_name": "PM Vishwakarma Tier 1",
        "content_type": "scheme",
        "target_group": "artisan",
        "state": "All India",
        "content": (
            "PM Vishwakarma Tier 1: Collateral-free loan of Rs 1,00,000 at 5% interest for "
            "traditional artisans and craftspeople — blacksmiths, goldsmiths, potters, carpenters, "
            "sculptors, cobblers, tailors, basket/mat/broom makers, doll/toy makers, barbers, "
            "garland makers, washermen, boat makers. Includes 5-7 days of skill training with "
            "Rs 500/day stipend + Rs 15,000 toolkit grant. Apply at PM Vishwakarma portal."
        ),
    },
    {
        "scheme_id": 11,
        "scheme_name": "PM Vishwakarma Tier 2",
        "content_type": "scheme",
        "target_group": "artisan",
        "state": "All India",
        "content": (
            "PM Vishwakarma Tier 2: Enhanced loan of Rs 2,00,000 at 5% interest for artisans "
            "who have repaid Tier 1. Includes 15 days of advanced skill training with stipend. "
            "Digital incentive: Rs 1 per digital transaction (max Rs 100/month) for artisans "
            "who adopt digital payments. Same 18 traditional craft categories as Tier 1."
        ),
    },
]

df_schemes = spark.createDataFrame(schemes)
df_schemes = df_schemes.select(
    "scheme_id", "scheme_name", "content_type", "target_group", "state", "content"
)

# Enable Change Data Feed so Vector Search can delta-sync
df_schemes.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(CORPUS_TABLE)

spark.sql(f"ALTER TABLE {CORPUS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Scheme corpus written: {CORPUS_TABLE}")
display(spark.read.table(CORPUS_TABLE))

# COMMAND ----------

# DBTITLE 1,Step 2 — Create a Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create the compute endpoint (takes ~5 min first time)
try:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Creating VS endpoint: {VS_ENDPOINT} (takes ~5 min)...")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"VS endpoint already exists: {VS_ENDPOINT}")
    else:
        raise

# Wait for endpoint to be ready
import time
for i in range(30):
    ep = vsc.get_endpoint(VS_ENDPOINT)
    status = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  [{i * 10}s] Endpoint status: {status}")
    if status == "ONLINE":
        break
    time.sleep(10)
else:
    print("Timed out — check Vector Search UI. Proceeding anyway...")

# COMMAND ----------

# DBTITLE 1,Step 3 — Create Delta-Sync Vector Search Index
# Uses Databricks Foundation Model `databricks-gte-large-en` for embeddings.
# No self-hosted embedding model needed — Databricks manages it.

try:
    idx = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=CORPUS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="scheme_id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print(f"Index created: {VS_INDEX}")
    print(f"Pipeline type: TRIGGERED (manually sync or schedule)")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Index already exists: {VS_INDEX}")
        # Trigger a sync to pick up any table changes
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
        idx.sync()
        print("Sync triggered.")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Step 4 — Wait for Index Sync and Test Retrieval
import time

print("Waiting for index to sync...")
for i in range(30):
    idx = vsc.get_index(VS_ENDPOINT, VS_INDEX)
    status = idx.describe().get("status", {})
    ready = status.get("ready", False)
    msg = status.get("message", "")
    print(f"  [{i * 10}s] ready={ready}  {msg}")
    if ready:
        break
    time.sleep(10)

# Test a retrieval query
print("\nTest query: 'I need a loan for my vegetable cart'")
results = idx.similarity_search(
    columns=["scheme_id", "scheme_name", "target_group", "content"],
    query_text="I need a loan for my vegetable cart",
    num_results=3,
)

print(f"\nTop {len(results.get('result', {}).get('data_array', []))} results:")
for row in results.get("result", {}).get("data_array", []):
    scheme_name = row[1]
    score = row[-1] if len(row) > 4 else "n/a"
    print(f"  {scheme_name} (score: {score})")

# COMMAND ----------

# DBTITLE 1,Step 5 — Test More Queries Across Personas
test_queries = [
    ("I am a farmer, I need money for my crops", "farmer"),
    ("I need a loan for my dairy farm", "farmer"),
    ("I want to start a small shop", "micro_business"),
    ("Our women's group needs a loan", "shg_woman"),
    ("I am a tailor, I need money for a sewing machine", "artisan"),
    ("I sell vegetables on the street, I need working capital", "street_vendor"),
]

print("Retrieval test across personas:\n")
for query, expected_group in test_queries:
    results = idx.similarity_search(
        columns=["scheme_name", "target_group"],
        query_text=query,
        num_results=2,
    )
    top = results.get("result", {}).get("data_array", [])
    top_name = top[0][0] if top else "—"
    top_group = top[0][1] if top else "—"
    match = "Y" if top_group == expected_group else " "
    print(f"  [{match}] '{query[:50]}' → {top_name} ({top_group})")

# COMMAND ----------

# DBTITLE 1,Step 6 — Print Voice Pipeline Integration Snippet
workspace_url = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)

print(f"""
# ── Voice pipeline integration — replace FAISS with Vector Search ─────────
# In voice/4_voice_pipeline.py, replace the FAISS retrieve() function with:

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vs_index = vsc.get_index("{VS_ENDPOINT}", "{VS_INDEX}")

def retrieve(query: str, state: str | None = None, top_k: int = 5) -> list[dict]:
    results = vs_index.similarity_search(
        columns=["scheme_id", "scheme_name", "content_type", "target_group", "state", "content"],
        query_text=query,
        num_results=top_k,
        filters={{"state": ["All India", state]}} if state else None,
    )
    chunks = []
    for row in results.get("result", {{}}).get("data_array", []):
        chunks.append({{
            "scheme_id": row[0],
            "scheme_name": row[1],
            "content_type": row[2],
            "target_group": row[3],
            "state": row[4],
            "content": row[5],
            "score": row[6] if len(row) > 6 else 0.0,
        }})
    return chunks

# For local dev (outside Databricks), fall back to FAISS:
#   if not on_databricks:
#       from voice.local.faiss_retriever import retrieve
# ─────────────────────────────────────────────────────────────────────────
""")

print(f"Vector Search endpoint: {VS_ENDPOINT}")
print(f"Index: {VS_INDEX}")
print(f"REST URL: {workspace_url}/api/2.0/vector-search/indexes/{VS_INDEX}/query")
