# Databricks notebook source
# ─────────────────────────────────────────────────────
# ArthaSetu + Xscore Integration Layer
# ─────────────────────────────────────────────────────
# This notebook bridges Xscore credit scoring with
# ArthaSetu voice RAG pipeline.
#
# Run AFTER:
#   1. Xscore gold layer is built (credit_scores, score_explanations)
#   2. ArthaSetu gold layer is built (rag_corpus, FAISS index)
#
# Creates:
#   - arthasetu.gold.user_credit_context  (joined view)
#   - arthasetu.gold.scheme_eligibility   (per-user scheme matching)
#   - Updates FAISS index with Xscore context chunks
# ─────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, concat, round as spark_round, when, coalesce,
    monotonically_increasing_id, array, struct
)

spark = SparkSession.getActiveSession()
if spark is None:
    spark = SparkSession.builder.getOrCreate()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Create joined user credit context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("[1/4] Building user credit context from Xscore...")
print("=" * 60)

# Join Xscore tables: profiles + features + scores + explanations
user_context = spark.sql("""
    SELECT
        p.user_id,
        p.name,
        p.city,
        p.state,
        p.occupation,
        p.age,
        p.gender,

        -- Financial features
        f.annual_income,
        f.monthly_expenses,
        f.savings_ratio,
        f.total_upi_txns,
        f.total_upi_volume,
        f.avg_upi_amount,
        f.upi_days_active,
        f.bill_ontime_rate,
        f.total_bills_paid,

        -- Credit score
        s.xscore          AS credit_score,
        s.score_band,
        s.percentile,

        -- SHAP explanations
        e.top_positive_factors,
        e.top_negative_factors,
        e.shap_values,

        -- Derived: score tier for scheme matching
        CASE
            WHEN s.xscore >= 700 THEN 'excellent'
            WHEN s.xscore >= 550 THEN 'good'
            WHEN s.xscore >= 400 THEN 'fair'
            ELSE 'building'
        END AS score_tier,

        -- RAG text for embedding
        CONCAT(
            'User Profile: ', p.occupation, ' from ', p.city, ', ', p.state, '. ',
            'Annual income Rs ', ROUND(f.annual_income, 0), '. ',
            'Credit score ', s.xscore, '/900 (', s.score_band, '). ',
            'UPI transactions: ', f.total_upi_txns, ' totalling Rs ', ROUND(f.total_upi_volume, 0), '. ',
            'Bill payment rate: ', ROUND(f.bill_ontime_rate * 100, 0), '%. ',
            'Savings ratio: ', ROUND(f.savings_ratio * 100, 0), '%.'
        ) AS rag_text

    FROM xscore.silver.user_profiles p
    LEFT JOIN xscore.silver.user_features f ON p.user_id = f.user_id
    LEFT JOIN xscore.gold.credit_scores s   ON p.user_id = s.user_id
    LEFT JOIN xscore.gold.score_explanations e ON p.user_id = e.user_id
""")

user_context.write.format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable("arthasetu.gold.user_credit_context")

cnt = user_context.count()
print(f"  ✅ arthasetu.gold.user_credit_context → {cnt:,} rows")
print()
user_context.select("user_id", "occupation", "city", "credit_score", "score_tier") \
    .show(5, truncate=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: Build per-user scheme eligibility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("[2/4] Matching users to eligible government schemes...")
print("=" * 60)

# Cross-join users with schemes, then filter by eligibility
scheme_eligibility = spark.sql("""
    WITH schemes AS (
        SELECT * FROM arthasetu.silver.loan_schemes
    ),
    users AS (
        SELECT user_id, occupation, credit_score, score_tier, annual_income
        FROM arthasetu.gold.user_credit_context
    )
    SELECT
        u.user_id,
        u.credit_score,
        u.occupation,
        s.scheme_name,
        s.max_loan_inr,
        s.interest_pct,
        s.collateral,
        s.eligibility,
        -- Simple eligibility flag (can be refined)
        CASE
            WHEN s.scheme_name LIKE '%Shishu%'     AND u.credit_score >= 300 THEN true
            WHEN s.scheme_name LIKE '%SVANidhi%'    AND u.credit_score >= 400 THEN true
            WHEN s.scheme_name LIKE '%Kishor%'      AND u.credit_score >= 550 THEN true
            WHEN s.scheme_name LIKE '%Tarun%'       AND u.credit_score >= 700 THEN true
            WHEN s.scheme_name LIKE '%Vishwakarma%' AND u.credit_score >= 350 THEN true
            WHEN s.scheme_name LIKE '%Kisan%'       AND u.credit_score >= 400 THEN true
            WHEN s.scheme_name LIKE '%SHG%'         AND u.credit_score >= 350 THEN true
            WHEN s.scheme_name LIKE '%NABARD%'      AND u.credit_score >= 500 THEN true
            ELSE false
        END AS is_eligible
    FROM users u
    CROSS JOIN schemes s
""")

# Keep only eligible combinations
eligible_only = scheme_eligibility.filter(col("is_eligible") == True)

eligible_only.write.format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable("arthasetu.gold.scheme_eligibility")

cnt = eligible_only.count()
print(f"  ✅ arthasetu.gold.scheme_eligibility → {cnt:,} rows")
print()

# Show distribution
spark.sql("""
    SELECT scheme_name, COUNT(DISTINCT user_id) as eligible_users
    FROM arthasetu.gold.scheme_eligibility
    GROUP BY scheme_name
    ORDER BY eligible_users DESC
""").show(truncate=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Add Xscore profile summaries to RAG corpus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("[3/4] Adding Xscore aggregates to RAG corpus...")
print("=" * 60)

# Aggregate by occupation + state (not individual users)
xscore_agg = spark.sql("""
    SELECT
        'xscore_profile' AS content_type,
        'xscore' AS source,
        CONCAT(occupation, ' - ', state) AS title,
        state,
        CONCAT(
            'Credit profile for ', occupation, ' in ', state, ': ',
            'Average Xscore ', ROUND(AVG(credit_score), 0), '/900. ',
            'Typical income Rs ', ROUND(AVG(annual_income), 0), '/year. ',
            'Average UPI transactions: ', ROUND(AVG(total_upi_txns), 0), '. ',
            'Bill payment rate: ', ROUND(AVG(bill_ontime_rate) * 100, 0), '%. ',
            'Sample size: ', COUNT(*), ' borrowers.'
        ) AS content
    FROM arthasetu.gold.user_credit_context
    GROUP BY occupation, state
    HAVING COUNT(*) >= 3
""")

# Append to existing RAG corpus
existing_corpus = spark.table("arthasetu.gold.rag_corpus")
new_corpus = existing_corpus.union(
    xscore_agg.withColumn("chunk_id", monotonically_increasing_id().cast("string"))
)

new_corpus.write.format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable("arthasetu.gold.rag_corpus")

total = new_corpus.count()
print(f"  ✅ Updated arthasetu.gold.rag_corpus → {total:,} chunks")
spark.sql("""
    SELECT content_type, COUNT(*) as chunks
    FROM arthasetu.gold.rag_corpus
    GROUP BY content_type
    ORDER BY chunks DESC
""").show()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: Rebuild FAISS index with new corpus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("[4/4] Rebuilding FAISS index...")
print("=" * 60)

import subprocess
subprocess.run(["pip", "install", "sentence-transformers", "faiss-cpu", "-q"], check=True)

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
corpus_pd = spark.table("arthasetu.gold.rag_corpus").toPandas()

print(f"  Embedding {len(corpus_pd):,} chunks...")
BATCH = 256
embeddings = []
for i in range(0, len(corpus_pd), BATCH):
    batch = corpus_pd["content"].iloc[i:i + BATCH].tolist()
    embs = embed_model.encode(batch, show_progress_bar=False)
    embeddings.extend(embs.tolist())
    if (i + BATCH) % 1024 == 0 or i + BATCH >= len(corpus_pd):
        print(f"    {min(i + BATCH, len(corpus_pd)):,}/{len(corpus_pd):,}")

vectors = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(vectors)

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# Save to Volume
VOLUME_PATH = "/Volumes/arthasetu/gold/rag_files"
spark.sql("CREATE VOLUME IF NOT EXISTS arthasetu.gold.rag_files")

faiss.write_index(index, f"{VOLUME_PATH}/rag.index")
with open(f"{VOLUME_PATH}/rag_meta.pkl", "wb") as f:
    pickle.dump(
        corpus_pd[["chunk_id", "content_type", "source", "title", "state", "content"]]
        .to_dict("records"),
        f,
    )

print(f"  ✅ FAISS index → {index.ntotal:,} vectors")
print(f"  📁 Saved to {VOLUME_PATH}")

# Also save embeddings to Delta
corpus_pd["embedding"] = embeddings
spark.createDataFrame(
    corpus_pd[["chunk_id", "content_type", "source", "title", "state", "content", "embedding"]]
).write.format("delta").mode("overwrite").saveAsTable("arthasetu.gold.rag_embeddings")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 60)
print("INTEGRATION COMPLETE")
print("=" * 60)

tables = {
    "arthasetu.gold.user_credit_context": "Xscore profiles + scores joined",
    "arthasetu.gold.scheme_eligibility": "Per-user scheme matching",
    "arthasetu.gold.rag_corpus": "Unified RAG corpus (ArthaSetu + Xscore)",
    "arthasetu.gold.rag_embeddings": "Embeddings for all chunks",
    "arthasetu.gold.voice_sessions": "Voice interaction logs",
}

for table, desc in tables.items():
    try:
        cnt = spark.sql(f"SELECT count(*) as c FROM {table}").collect()[0]["c"]
        print(f"  ✅ {table:45s} {cnt:>8,} rows — {desc}")
    except Exception:
        print(f"  ⚠️  {table:45s}      N/A — {desc} (will be created on first use)")

print()
print(f"  📊 FAISS index: {index.ntotal:,} vectors")
print(f"  📁 Volume: {VOLUME_PATH}")
print("=" * 60)
print("✅ Ready — deploy Streamlit app: streamlit run arthasetu_integrated_app.py")

