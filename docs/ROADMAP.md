# Roadmap

Where the project stands and what's next. Written to be honest, not aspirational.

## Where we are now

- **Alt-credit scoring:** working. LightGBM regressor, R²=0.8938, RMSE=14.5 on the 100K synthetic-borrower / 27.3M-transaction dataset. Trained + registered in MLflow. Explorer dashboard live on Streamlit Cloud.
- **Voice + RAG:** working end-to-end in `Voice Ai/4_voice_pipeline.py`. FAISS index over govt loan schemes + Sarvam ASR / sarvam-m / Bulbul TTS. Tested across 4 languages inline.
- **Secrets purged:** Sarvam key + HF token were previously committed, now revoked and removed from all git history.
- **Repo hygiene:** file renames done; layout is still messy — two parallel workstreams (XScore + VishwaScore + ArthaSetu) coexist.

## Not yet done (honest gap list)

- ~~**Feature Store registration**~~ — Done. `06_feature_store_registration.py` registers the Gold table via `FeatureEngineeringClient` and logs the model with feature lineage.
- ~~**Databricks Vector Search**~~ — Done. `08_vector_search_rag.py` creates a delta-sync index over the scheme corpus using `databricks-gte-large-en` Foundation Model embeddings.
- ~~**Model Serving endpoint**~~ — Done. `07_model_serving_deploy.py` deploys the @Champion model as a REST endpoint with auto-scaling.
- **Latency instrumentation** — the voice pipeline has no p50 / p95 measurement.
- **BhashaBench eval** — the loader exists (`Voice Ai/Load BhashaBench Data.ipynb`); the actual eval run + numbers have not been re-verified for this repo.
- **Model card + fairness audit** — no slice metrics by persona / gender / region yet.
- **RAG eval harness** — hit@k / MRR / LLM-as-judge on a held-out Q&A set.
- **Extract notebooks into `src/vishwascore/` package + pytest CI.**
- **Consolidate parallel workstreams** — pick canonical directory layout, archive the rest.

## Cleanup done

- ✅ Stub notebooks deleted (`VishwaScore Feature Store Registration.py`, `VishwaScore Financial Literacy RAG.py`).
- ✅ `Xscore/` folder (byte-identical duplicates + one buggy app variant) deleted.
- ✅ Top-level junk removed: `QUICK_REFERENCE.txt`, `README_STREAMLIT.md`, empty `data`, `requirements.txt`.
- ✅ `Voice Ai/` renamed to `voice/`; bad-name files inside (spaces, parens, colons) renamed to snake_case.
- ✅ Top-level `VishwaScore *.py` notebooks moved into `notebooks/vishwascore/` with numeric prefixes.
- ✅ Empty `src/vishwa/` skeleton removed.
- ✅ `pyproject.toml` simplified — no phantom package.

## Still to do

- Two workstreams (XScore in `notebooks/data/` + `notebooks/model/` + `app/`, and VishwaScore in `notebooks/vishwascore/` + `vishwascore_streamlit_app.py`) still coexist for provenance; consider archiving XScore once VishwaScore is the sole story.
- Consolidate the three voice-app variants (`voice/voice_app.py`, `voice/arthasetu_app.py`, `voice/arthasetu_integrated_app.py`) into one canonical entrypoint.

## Priority for interview readiness

1. Fill in the BhashaBench + RAG eval numbers with real re-runs.
2. Instrument voice-pipeline latency, put p50 / p95 in the README results table.
3. Publish `docs/MODEL_CARD.md`.
4. Consolidate to a single canonical layout.
5. Extract notebooks → `src/vishwascore/` + pytest CI.
