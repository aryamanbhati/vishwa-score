# Databricks notebook source
# DBTITLE 1,RAG Configuration and Imports
# ============================================================================
# VISHWASCORE FINANCIAL LITERACY RAG
# ============================================================================
# 🎯 WOW FACTOR: GenAI-powered financial advisor using RBI knowledge base
# Uses: Databricks Vector Search + Foundation Model APIs (DBRX/Llama)
# ============================================================================

import os
import json
from typing import List, Dict
import pandas as pd

print("="*70)
print("  🤖 RAG-POWERED FINANCIAL LITERACY ASSISTANT")
print("="*70)
print("  ✓ Vector Search: Semantic search over RBI guidelines")
print("  ✓ Foundation Models: DBRX/Llama for natural language responses")
print("  ✓ RAG Pipeline: Retrieval → Context → Generation")
print("  ✓ Use Case: Answer user questions about improving VishwaScore")
print("="*70)

# Configuration
VECTOR_SEARCH_ENDPOINT = "vishwascore_vector_search_endpoint"
KNOWLEDGE_BASE_INDEX = "workspace.default.vishwascore_knowledge_base_index"
SOURCE_TABLE = "workspace.default.rbi_financial_guidelines"

print(f"\n✓ Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"✓ Knowledge Base Index: {KNOWLEDGE_BASE_INDEX}")
print(f"✓ Source Documents: {SOURCE_TABLE}\n")