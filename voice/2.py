# ─────────────────────────────────────────────
# ArthaSetu — Unity Catalog Setup
# Run this ONCE before anything else
# ─────────────────────────────────────────────

from pyspark.sql import SparkSession
from databricks.sdk.runtime import dbutils

spark = SparkSession.builder.getOrCreate()

def setup_catalog():
    print("Setting up ArthaSetu Unity Catalog...")
    print()

    cmds = [
        "CREATE CATALOG IF NOT EXISTS arthasetu",
        "CREATE SCHEMA IF NOT EXISTS arthasetu.bronze",
        "CREATE SCHEMA IF NOT EXISTS arthasetu.silver",
        "CREATE SCHEMA IF NOT EXISTS arthasetu.gold",
        "CREATE VOLUME IF NOT EXISTS arthasetu.bronze.uploads",
    ]

    for cmd in cmds:
        spark.sql(cmd)
        print(f"  ✅ {cmd}")

    print()
    print("Schemas in arthasetu catalog:")
    spark.sql("SHOW SCHEMAS IN arthasetu").show()

    print("Verifying Volume...")
    try:
        dbutils.fs.ls("/Volumes/arthasetu/bronze/uploads/")
        print("  ✅ Volume ready")
    except Exception as e:
        print(f"  ❌ Volume error: {e}")

    print()
    print("✅ Setup complete — ready for data ingestion")


if __name__ == "__main__":
    setup_catalog()