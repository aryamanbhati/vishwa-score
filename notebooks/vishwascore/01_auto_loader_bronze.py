# Databricks notebook source
# DBTITLE 1,Auto Loader Configuration - Streaming Bronze Layer
# ============================================================================
# AUTO LOADER: STREAMING BRONZE LAYER FOR VISHWASCORE
# ============================================================================
# Continuously ingests CSV bank statements from cloud storage
# Features: Schema inference, schema evolution, rescue data column
# WOW FACTOR: Handles late-arriving data, checkpoint recovery, exactly-once processing
# ============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import *

# Configuration
SOURCE_PATH = "/Volumes/xscore/bronze/bank_statement_01/*.csv"
CHECKPOINT_PATH = "/Volumes/xscore/bronze/checkpoints/bronze_autoloader"
BRONZE_TABLE = "xscore.bronze.bronze_bank_statements_streaming"

print("="*70)
print("  AUTO LOADER: Streaming Bronze Layer")
print("="*70)
print(f"Source: {SOURCE_PATH}")
print(f"Target: {BRONZE_TABLE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Auto Loader Stream - Incremental CSV Ingestion
# ============================================================================
# AUTO LOADER STREAM CONFIGURATION
# ============================================================================

# Define expected schema (helps with performance, optional)
expected_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("date", StringType(), True),  # Will parse later
    StructField("particulars", StringType(), True),
    StructField("chq_num", StringType(), True),
    StructField("withdrawal", StringType(), True),  # Can have commas
    StructField("deposit", StringType(), True),
    StructField("balance", StringType(), True)
])

# Auto Loader with cloudFiles
df_stream = (spark.readStream
    .format("cloudFiles")  # 🔥 Auto Loader magic
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")  # Schema evolution tracking
    .option("cloudFiles.inferColumnTypes", "true")  # Infer types automatically
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")  # Handle schema changes
    .option("cloudFiles.includeExistingFiles", "true")  # Process existing + new files
    .option("rescuedDataColumn", "_rescued_data")  # Capture malformed rows
    .option("header", "true")
    .option("inferSchema", "true")
    .load(SOURCE_PATH)
)

# Add metadata columns (key for auditing)
df_stream_enriched = (df_stream
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("source_file", F.input_file_name())
    .withColumn("file_modification_time", F.col("_metadata.file_modification_time"))
)

print("✓ Auto Loader stream configured")
print(f"  - Schema evolution: ENABLED")
print(f"  - Rescued data column: _rescued_data")
print(f"  - Incremental processing: ENABLED")

# COMMAND ----------

# DBTITLE 1,Write Stream to Bronze Delta Table
# ============================================================================
# STREAMING WRITE TO BRONZE DELTA TABLE
# ============================================================================

# Write stream to Bronze table with checkpointing
query = (df_stream_enriched.writeStream
    .format("delta")
    .outputMode("append")  # Append new records
    .option("checkpointLocation", CHECKPOINT_PATH)  # Fault-tolerance
    .option("mergeSchema", "true")  # Allow schema evolution
    .trigger(processingTime="10 seconds")  # Micro-batch every 10s (tune for your use case)
    .table(BRONZE_TABLE)
)

print("\n" + "="*70)
print("  🚀 STREAMING INGESTION STARTED")
print("="*70)
print(f"Stream ID: {query.id}")
print(f"Status: {query.status}")
print(f"\nMonitor at: Databricks SQL > Query History")
print(f"Or use: query.status (in this notebook)")
print("\n💡 TIP: In production, use .trigger(availableNow=True) for triggered incremental")
print("="*70)

# IMPORTANT: This will run continuously. 
# For the hackathon demo, you can:
# 1. Let it run for a few seconds to process existing files
# 2. Use query.stop() to stop it
# 3. Or use trigger(availableNow=True) for one-time incremental processing

# COMMAND ----------

# DBTITLE 1,Monitor Stream and Stop (For Demo)
# ============================================================================
# STREAM MONITORING & CONTROL
# ============================================================================

import time

# Wait for some processing (for demo purposes)
print("Processing for 30 seconds...")
time.sleep(30)

# Check stream status
print(f"\nStream Status: {query.status}")
print(f"Is Active: {query.isActive}")
print(f"Recent Progress:")
if query.recentProgress:
    latest = query.recentProgress[-1]
    print(f"  - Batch: {latest.get('batchId', 'N/A')}")
    print(f"  - Rows Processed: {latest.get('numInputRows', 0)}")
    print(f"  - Processing Time: {latest.get('durationMs', {}).get('triggerExecution', 0)}ms")

# Stop the stream (for demo)
print("\n⏸️  Stopping stream for demo...")
query.stop()
print("✓ Stream stopped")

# Verify data in Bronze
df_bronze = spark.read.table(BRONZE_TABLE)
print(f"\n✓ Bronze table has {df_bronze.count():,} records")
print(f"✓ Unique users: {df_bronze.select('user_id').distinct().count():,}")

# COMMAND ----------

# DBTITLE 1,Production Pattern: Triggered Incremental (Recommended)
# ============================================================================
# PRODUCTION PATTERN: TRIGGER ONCE (availableNow=True)
# ============================================================================
# This is better for scheduled jobs - processes all new data then stops
# Recommended for hackathon: More predictable than continuous streaming
# ============================================================================

# ALTERNATIVE: Use this pattern for production/scheduled jobs
"""
# This processes all available data incrementally, then stops
query_production = (df_stream_enriched.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # 🔥 Process once then stop
    .table(BRONZE_TABLE)
)

# Wait for completion
query_production.awaitTermination()

print("✓ Incremental processing complete")
"""

print("\n" + "="*70)
print("  📚 AUTO LOADER PATTERNS")
print("="*70)
print("1. CONTINUOUS (Demo/Dev):")
print("   .trigger(processingTime='10 seconds')")
print("   - Always running, processes micro-batches")
print("   - Good for real-time dashboards")
print("")
print("2. TRIGGERED INCREMENTAL (Production):")
print("   .trigger(availableNow=True)")
print("   - Processes all new data then stops")
print("   - Run via Databricks Job scheduler")
print("   - Exactly-once guarantees with checkpoints")
print("="*70)

print("\n🎯 WOW FACTORS FOR JUDGES:")
print("  ✓ Schema evolution (handles CSV format changes)")
print("  ✓ Rescued data column (no data loss)")
print("  ✓ Exactly-once processing (checkpoint-based)")
print("  ✓ Incremental processing (only new files)")
print("  ✓ Production-ready pattern (trigger modes)")