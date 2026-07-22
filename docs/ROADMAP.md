# Roadmap

Post-hackathon polish work, tracked publicly so the repo shows momentum.

## Step 1 — Repo hygiene (done)

- [x] Rename files to remove spaces / version suffixes.
- [x] Restructure into `app/`, `notebooks/`, `src/vishwa/`, `docs/`, `tests/`, `infra/`.
- [x] Add `.gitignore`, `LICENSE` (MIT), `pyproject.toml`, `.env.example`.
- [x] Rewrite README with fixed clone URL, results table, roadmap.

## Step 2 — Extract notebooks into a real package

- [ ] `src/vishwa/ingestion/` — bronze / silver / gold Spark jobs (importable, testable).
- [ ] `src/vishwa/features/` — feature engineering + Feature Store writes.
- [ ] `src/vishwa/model/` — training, tuning, SHAP, refresh.
- [ ] `src/vishwa/rag/` — FAISS index build + retriever.
- [ ] `src/vishwa/voice/` — Sarvam ASR / TTS / LLM wrappers.
- [ ] `src/vishwa/app/` — Streamlit views (leave `app/app.py` as thin entrypoint).
- [ ] Notebooks become thin wrappers that `import vishwa.<module>` — reproducibility preserved.

## Step 3 — Evidence, not just claims

- [ ] `docs/MODEL_CARD.md`: dataset, intended use, metrics, slice metrics by gender / region / age band, fairness audit (demographic parity, equal opportunity), known limitations.
- [ ] `docs/RAG_EVAL.md`: 50-item Q&A eval set, hit@k, MRR, LLM-as-judge faithfulness.
- [ ] `docs/LATENCY.md`: instrumented p50 / p95 for ASR → retrieval → LLM → TTS on a fixed benchmark set.
- [ ] Replace `TODO` rows in the README results table with real numbers.

## Step 4 — Tests + CI

- [ ] `tests/` — smoke tests for feature builders, retriever top-k, model I/O contract.
- [ ] `.github/workflows/ci.yml` — ruff + pytest on PR.
- [ ] `pre-commit` config.

## Step 5 — Architecture decisions

- [ ] `docs/decisions/0001-lightgbm-vs-neural.md`
- [ ] `docs/decisions/0002-faiss-plus-vector-search.md`
- [ ] `docs/decisions/0003-synthetic-data-strategy.md`

## Step 6 — Demo assets

- [ ] 60–90s Loom / YouTube demo linked from README.
- [ ] Screenshots / GIF of the app in each of the four dashboard tabs.
