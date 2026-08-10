from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, lit
import pandas as pd
import numpy as np

spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG arthasetu")

# ── STEP 1: Install deps ─────────────────────────
import subprocess
subprocess.run(["pip", "install", "sentence-transformers", "faiss-cpu", "-q"], check=True)
print("Dependencies ready")

# Import after installation
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ── STEP 2: Build Gold RAG corpus ───────────────
print("\n[1/3] Building Gold RAG corpus...")

corpus_parts = []

# Source 1: Loan schemes
try:
    schemes = spark.table("arthasetu.silver.loan_schemes") \
        .selectExpr(
            "'loan_scheme' as content_type",
            "'gov_schemes' as source",
            "scheme_name as title",
            "'All India' as state",
            "rag_text as content"
        )
    corpus_parts.append(schemes)
    print("  ✅ loan_schemes loaded")
except Exception as e:
    print(f"  ❌ loan_schemes skipped: {e}")

# Source 2: Finance Q&A
try:
    qa = spark.table("arthasetu.silver.finance_qa") \
        .selectExpr(
            "'finance_qa' as content_type",
            "'bhashbench' as source",
            "question as title",
            "'All India' as state",
            "rag_text as content"
        )
    corpus_parts.append(qa)
    print("  ✅ finance_qa loaded")
except Exception as e:
    print(f"  ❌ finance_qa skipped: {e}")

# Source 3: Rural borrower profiles (aggregated)
try:
    profiles = spark.table("arthasetu.silver.rural_profiles_agg") \
        .selectExpr(
            "'borrower_profile' as content_type",
            "'rural_loan' as source",
            "occupation as title",
            "city as state",
            "rag_text as content"
        )
    corpus_parts.append(profiles)
    print("  ✅ rural_profiles_agg loaded")
except Exception as e:
    print(f"  ❌ rural_profiles_agg skipped: {e}")

# Source 4: State credit context
try:
    states = spark.table("arthasetu.silver.state_context") \
        .selectExpr(
            "'state_context' as content_type",
            "'pmmy' as source",
            "state as title",
            "state",
            "rag_text as content"
        )
    corpus_parts.append(states)
    print("  ✅ state_context loaded")
except Exception as e:
    print(f"  ❌ state_context skipped: {e}")

# Check if we have any data
if not corpus_parts:
    raise RuntimeError("No silver tables found! Run silver processing first.")

# Merge all available parts
corpus = corpus_parts[0]
for part in corpus_parts[1:]:
    corpus = corpus.union(part)

corpus = corpus.withColumn("chunk_id", monotonically_increasing_id().cast("string"))

corpus.write.format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable("arthasetu.gold.rag_corpus")

total = corpus.count()
print(f"\n  ✅ arthasetu.gold.rag_corpus → {total:,} chunks")
spark.sql("""
    SELECT content_type, source, count(*) as chunks
    FROM arthasetu.gold.rag_corpus
    GROUP BY content_type, source
    ORDER BY chunks DESC
""").show()

# ── STEP 3: Embed corpus ─────────────────────────
print("\n[2/3] Embedding corpus...")
embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

corpus_pd = spark.table("arthasetu.gold.rag_corpus").toPandas()
print(f"  Embedding {len(corpus_pd):,} chunks in batches...")

BATCH = 256
embeddings = []
for i in range(0, len(corpus_pd), BATCH):
    batch = corpus_pd["content"].iloc[i:i+BATCH].tolist()
    embs  = embed_model.encode(batch, show_progress_bar=False)
    embeddings.extend(embs.tolist())
    print(f"  {min(i+BATCH, len(corpus_pd)):,}/{len(corpus_pd):,} done")

corpus_pd["embedding"] = embeddings

# Save embeddings to Gold Delta
spark.createDataFrame(
    corpus_pd[["chunk_id","content_type","source","title","state","content","embedding"]]
).write.format("delta") \
 .mode("overwrite") \
 .saveAsTable("arthasetu.gold.rag_embeddings")

print(f"  ✅ arthasetu.gold.rag_embeddings → {len(corpus_pd):,} rows")
print(f"  Embedding dim: {len(embeddings[0])}")

# ── STEP 4: Build FAISS index on Volumes ───────────
print("\n[3/3] Building FAISS index...")

vectors = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(vectors)

dim   = vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vectors)

# Create volume if it doesn't exist
spark.sql("CREATE VOLUME IF NOT EXISTS arthasetu.gold.rag_files")
print("  Volume ready: arthasetu.gold.rag_files")

# Use Unity Catalog Volumes
VOLUME_PATH = "/Volumes/arthasetu/gold/rag_files"
FAISS_PATH = f"{VOLUME_PATH}/rag.index"
META_PATH  = f"{VOLUME_PATH}/rag_meta.pkl"

faiss.write_index(index, FAISS_PATH)
with open(META_PATH, "wb") as f:
    pickle.dump(
        corpus_pd[["chunk_id","content_type","source","title","state","content"]]\
            .to_dict("records"),
        f
    )

print(f"  FAISS index saved → {FAISS_PATH}")
print(f"  Total vectors: {index.ntotal:,}")

# ── STEP 5: Test retrieval ───────────────────────
print("\n Testing RAG retrieval...")

def retrieve(query, state=None, top_k=5):
    vec = embed_model.encode([query]).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, idxs = index.search(vec, top_k * 3)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1: continue
        chunk = corpus_pd.iloc[idx].to_dict()
        chunk["score"] = round(float(score), 3)
        if state and state.lower() in chunk.get("state","").lower():
            chunk["score"] += 0.1
        results.append(chunk)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# Test queries
test_queries = [
    ("loan for vegetable vendor",      "Uttar Pradesh"),
    ("dairy farm loan",                "Bihar"),
    ("women self help group credit",   "Tamil Nadu"),
    ("artisan blacksmith loan",        "Maharashtra"),
]

for query, state in test_queries:
    print(f"\n  Query: '{query}' | State: {state}")
    results = retrieve(query, state=state, top_k=3)
    for r in results:
        print(f"    [{r['content_type']}] {r['title'][:50]} — {r['score']}")

# ── Final summary ────────────────────────────────
print("\n" + "="*55)
print("GOLD LAYER — COMPLETE")
print("="*55)
for t in ["rag_corpus", "rag_embeddings"]:
    cnt = spark.sql(f"SELECT count(*) as c FROM arthasetu.gold.{t}").collect()[0]['c']
    print(f"  ✅ arthasetu.gold.{t:25s} {cnt:,} rows")
print(f"  ✅ FAISS index in Volumes       {index.ntotal:,} vectors")
print(f"  📁 Location: {VOLUME_PATH}")
print("="*55)
print("Gold complete — run Voice Pipeline next")
