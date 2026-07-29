# AlphaDesk — Design Document

**Research & Signal Retrieval Platform for Quantitative Investment**

| | |
|---|---|
| Status | Proposal — **Direction A** (designed to be proposed and deployed inside the student's current employer) |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer–grade work: quantized document retrieval, two-stage retrieval→ranking, LLM research agent with citation grounding, and automated model iteration — built in class on public data, then replicated on internal data at work |

---

## 1. The Core Idea

The student is **currently employed at Changsheng Fund Management** and already owns two things there: an AutoML hyper-parameter agent, and a two-stage sector-rotation → stock-selection strategy.

AlphaDesk is designed so that **the same technical content lives naturally inside that role**. Instead of writing an outside project onto a real employer's record — which fails background checks — the student proposes AlphaDesk internally, builds it there, and the résumé describes what he genuinely did at work.

**Two tracks run in parallel:**

| | Class track (with mentor) | Work track (at Changsheng) |
|---|---|---|
| Corpus | Public: SEC EDGAR filings / A-share announcements, FiQA benchmark | Internal research reports, announcements, notes |
| Purpose | Learn the architecture, benchmark on public labels, build reference implementation | Apply it to the firm's real research workflow |
| Output | GitHub repo, benchmark tables, design docs | Internal tool + the résumé bullets |

The class track de-risks the work track: he arrives at his manager with a working prototype and measured numbers, not a proposal slide.

## 2. Alignment With Prior Experience

| Prior experience | Skill demonstrated | AlphaDesk module |
|---|---|---|
| **CAS** — patented cascaded binary-encoding image retrieval; 50k+ images as binary signatures; Precision@K / latency / storage across 20+ configs | **Embedding retrieval, quantization, index benchmarking** | **Vector index layer**: document embeddings in FAISS; IVF-PQ / HNSW / binary benchmark on Recall@K vs. latency vs. memory. *Identical methodology — image corpus becomes a document corpus.* |
| **Changsheng** — two-stage sector→stock strategy; AutoML hyper-parameter agent; backtested excess return after costs | **Two-stage funnel, automated tuning, backtest discipline** | **Retrieval → rerank** funnel for documents, and **screening → ranking** for signals; **Optuna + MLflow with backtest promotion gates** — a direct upgrade of work he already owns |
| **Sina Weibo** — scheduled SQL ETL, standardized metrics, monitoring pipelines with alerting | Pipelines, metric standardization, **monitoring & alerting** | **Airflow ingestion DAGs** for filings/news + **data-quality and embedding-drift monitors with alerts** |
| **College Application Agent** — EBR behind a FastAPI endpoint; LangGraph **Writer–Critic** loop | LLM engineering, API design, self-critique | **LLM research agent**: question → constrained retrieval → grounded answer with citations → **critic pass** verifying every claim traces to a retrieved passage |
| **DiDi** — Java backend modules; **A/B testing** | Backend, experimentation | **FastAPI services** + **champion/challenger** evaluation for ranking models |
| **Music Genre** — PCA, K-means features | Dimensionality reduction, clustering | Conceptual bridge to product quantization (PQ = clustering in subspaces) |

**Interview one-liner:** *"At CAS I hand-implemented binary hashing for image retrieval and measured the accuracy–memory–latency frontier. At Changsheng I rebuilt that idea for our research corpus with modern embeddings and a proper ANN index, then put a two-stage ranker and a grounded LLM agent on top of it."*

## 3. Overview

AlphaDesk is an internal research platform with three surfaces:

1. **Semantic research search** — natural-language query over filings, announcements, and research notes, returning ranked passages with source citations.
2. **Signal screening & ranking** — a two-stage pipeline that narrows a candidate universe, then ranks it with a learned model, evaluated by backtest rather than classification accuracy.
3. **Research Q&A agent** — an LLM that answers analyst questions strictly from retrieved passages, with a critic pass that rejects any unsupported claim.

Underneath: a quantized vector index, scheduled ingestion pipelines, automated hyper-parameter search with promotion gates, drift monitoring, and production-style serving.

## 4. Goals & Non-Goals

### Goals
- **Quantized document retrieval as the headline deliverable**: IVF-PQ / HNSW / binary-hash benchmark producing a Recall@K vs. QPS vs. memory Pareto table.
- **Two-stage retrieval**: bi-encoder recall → cross-encoder or LTR rerank, benchmarked on public labeled data (FiQA/BEIR) before touching internal data.
- **Signal ranking pipeline** with rigorous evaluation: IC / rank-IC, backtest excess return net of costs, turnover — the language the student's team already speaks.
- **Grounded LLM research agent** with citation verification and a measured unsupported-claim rate.
- **Automated iteration**: Optuna sweeps, MLflow registry, Airflow retraining with **backtest gates** blocking promotion on regression.
- **Production-style serving**: FastAPI services, Redis caching, monitoring, load tests.

### Non-Goals
- Trading execution, order routing, or live capital deployment.
- Training financial foundation models from scratch.
- Any use of proprietary data in the class track — the mentored repo uses **public data only**.

## 5. Datasets

### Class track (public, laptop-scale)

| Dataset | Role | Notes |
|---|---|---|
| **FiQA-2018** (via BEIR) | **Labeled retrieval benchmark** | Financial QA with query–passage relevance judgments; published BEIR baselines (BM25, dense retrievers) give a public number to beat — a leaderboard for the retrieval layer. |
| **SEC EDGAR** full-text filings (10-K / 10-Q / 8-K) | Realistic corpus | Free API, large, and messy in exactly the ways real documents are — good for the ETL and chunking work. |
| **A-share announcements** (CNINFO / exchange disclosures) | Optional domain match | Aligns the class corpus with what he actually sees at Changsheng. |
| **Market data** via `akshare` / `tushare` / `yfinance` | Signal-ranking track | Prices, fundamentals, industry classification for the screening → ranking → backtest pipeline. |

**Scoping**: one sector or index constituent list; 2–3 years of filings; a few hundred thousand passages after chunking — comfortably laptop-scale.

### Work track (internal)
Internal research corpus, subject to the firm's data-governance rules. **Nothing internal enters the public repo.** Handling that boundary correctly is itself a strong interview story about working with sensitive data.

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, SQL |
| Embeddings | BGE / GTE (text), optional finance-tuned encoders; inference only, precomputed in batch |
| **Vector index** | **FAISS** — Flat baseline, **IVF-PQ**, HNSW, binary; the benchmark is itself a deliverable |
| Lexical baseline | BM25 (rank_bm25 or OpenSearch) — hybrid fusion via RRF |
| Reranking | Cross-encoder (BGE-reranker) fine-tuned on FiQA; LightGBM LambdaMART as the interpretable comparison |
| Signal ranking | LightGBM / linear factor models with IC and backtest evaluation |
| LLM | GPT-4o-mini / Claude Haiku via API; Qwen2.5-7B via Ollama as local and offline fallback |
| Serving | FastAPI + Pydantic microservices; Redis (cache, session) |
| Pipelines | **Airflow** (ingestion, embedding, retraining DAGs), DuckDB + Parquet, PostgreSQL |
| Experimentation | **Optuna**, **MLflow** (tracking + registry) |
| Ops | Docker Compose, Prometheus + Grafana, Locust, GitHub Actions CI |

## 7. Architecture

```
 Analyst / Service client
   ├── GET  /v1/search        natural-language research search
   ├── POST /v1/ask           grounded Q&A with citations
   ├── GET  /v1/signals/rank  ranked candidate universe
   └── POST /v1/feedback      relevance feedback from analysts
                    │
            API Gateway (FastAPI) · auth · rate limit · request_id
                    │
              Orchestrator
                    ├─► RETRIEVAL SERVICE
                    │     ├ lexical  BM25
                    │     └ dense    BGE → FAISS IVF-PQ
                    │        → RRF fusion → top-200
                    │
                    ├─► RERANK SERVICE
                    │     cross-encoder / LambdaMART → top-20 passages
                    │
                    ├─► SIGNAL RANKING SERVICE
                    │     universe screening → feature assembly → ranker
                    │     → backtest-evaluated scores
                    │
                    └─► RESEARCH AGENT
                          question → constrained retrieval → answer
                          → CRITIC: every claim must map to a cited passage
                          → revise once → else abstain   [timeout → fallback]

 Offline:  Airflow → filing/news ingestion → chunk → embed → index build
           → training → Optuna sweeps → MLflow → BACKTEST GATE → promote
 Ops:      Prometheus/Grafana · data-quality & drift alerts · Locust
```

## 8. Component Design

### 8.1 Vector Index Engineering — the CAS continuation
- Chunk and embed the corpus; float32 Flat index as exact-search ground truth.
- Benchmark across 20+ configurations:
  - **IVF-PQ**: `nlist`, `nprobe`, subquantizer count `m`, bits per code
  - **HNSW**: `M`, `efConstruction`, `efSearch`
  - **Binary hashing**: the direct modern echo of the CAS method
- Deliverable: **Recall@50 vs. QPS vs. index memory** table + Pareto plot + a documented configuration choice. Compression framing: 768-dim float32 ≈ 3 KB/vector → PQ96 ≈ 96 bytes ≈ **32× smaller** — exactly the storage-vs-accuracy trade-off measured at CAS, now on an industrial index library.

### 8.2 Two-Stage Retrieval
- **Stage 1 recall**: hybrid BM25 + dense retrieval, RRF fusion — teaches why lexical and semantic retrieval are complementary (a top search-interview topic).
- **Stage 2 rerank**: cross-encoder fine-tuned on FiQA; LambdaMART comparison with interpretable features (BM25 score, dense similarity, recency, document type, source authority).
- Benchmarked on FiQA/BEIR against published baselines — a public number, not a self-reported one.

### 8.3 Signal Screening & Ranking — extending work he already owns
- Universe screening (stage 1) → learned ranking over factor and text-derived features (stage 2), mirroring his existing sector→stock structure but with a learned ranker replacing heuristics.
- **Text-derived features** are the novel contribution: sentiment and event tags extracted from filings/news by the LLM, joined onto the factor table — where retrieval and quant genuinely meet.
- Evaluation in his team's own language: **IC, rank-IC, backtest excess return net of costs, turnover, max drawdown** — not accuracy.

### 8.4 LLM Research Agent — the College-Agent continuation
- Question → constrained retrieval → answer generated **only** from retrieved passages, every claim carrying a citation.
- **Critic pass** (descendant of the LangGraph Writer–Critic loop): verifies each sentence maps to a cited passage, that numbers match the source, and that no entity outside the retrieved set is named; one revision, then abstain rather than guess.
- Metrics: citation-grounding rate, **unsupported-claim rate**, abstention rate, cost per query. In finance, "refuses to answer when evidence is missing" is a feature — and a strong interview point.

### 8.5 Automated Iteration — the Changsheng continuation
- Optuna sweeps over chunk size, embedding model, `nprobe`, reranker depth, and ranker hyper-parameters; every trial logged in MLflow.
- Airflow DAG: `ingest → chunk → embed → index rebuild → retrain → offline eval → BACKTEST GATE → promote or alert`. The gate is the crucial piece — it encodes "never promote on in-sample improvement alone", the discipline of quantitative research applied to ML deployment.

### 8.6 Pipelines, Serving & Monitoring — Weibo + DiDi continuation
- Scheduled ingestion with schema validation, dedup, and volume-anomaly detection; standardized metric definitions shared between training and serving.
- **Drift monitoring** on embedding and feature distributions with alert rules.
- **Champion/challenger** evaluation for ranking models — the model-level analogue of the page-layout A/B tests run at DiDi.
- FastAPI microservices, Redis caching, Locust load tests with P50/P99 reporting.

## 9. Evaluation Plan

| Area | Metrics |
|---|---|
| Index engineering | Recall@50 vs. QPS vs. memory across 20+ configs; compression ratio; Pareto plot |
| Retrieval | NDCG@10 / MRR / Recall@100 on **FiQA vs. published BEIR baselines**; hybrid vs. dense-only vs. BM25-only ablation |
| Signal ranking | IC, rank-IC, backtest excess return net of costs, turnover, drawdown vs. the existing heuristic baseline |
| LLM agent | Citation-grounding rate, unsupported-claim rate, abstention rate, cost/query |
| System | QPS, P50/P99 latency, cache hit rate |
| Ops | Gate pass/fail history; drift alerts fired vs. injected anomalies |

## 10. Milestones (mapped to the existing 19-hour, 15-session plan)

| Sprint | Sessions | Deliverable |
|---|---|---|
| **S0** — Setup | 3–4 | Agile workflow + AI tooling; repo, environment, corpus scoped |
| **S1** — Ingestion & quantized retrieval | 5–8 | Airflow ingestion, chunking, embeddings, **index benchmark table**, `/v1/search` live |
| **S2** — Reranking & signal ranking | 9–10 | Cross-encoder + LambdaMART on FiQA; screening→ranking pipeline with backtest evaluation; Optuna sweeps |
| **S3** — Research agent | 11–12 | Grounded Q&A, critic pass, agent metrics |
| **S4** — Serving, MLOps & monitoring | 13–14 | Microservices, caching, champion/challenger, drift alerts, load tests |
| — | 15 | Interview sprint |

Structure identical to the existing outline; only module content differs.

## 11. Résumé Bullets (to sit under **Changsheng Fund Management** — only after the work is genuinely done there)

- Built a **semantic research retrieval service** over the firm's document corpus, embedding and indexing passages in FAISS and benchmarking **IVF-PQ, HNSW, and binary-hash configurations across 20+ settings** to select an index sustaining X% Recall@50 at P99 < X ms with **X× lower memory** than exact search.
- Designed a **two-stage retrieval pipeline** — hybrid BM25 + dense recall fused by RRF, followed by a fine-tuned cross-encoder reranker — improving **NDCG@10 by X%** over the BM25 baseline.
- Extended the sector-rotation → stock-selection workflow with a **learned ranking model over factor and LLM-extracted text features**, improving rank-IC from X to X and raising backtested excess return to **7.2% annualized** net of transaction costs.
- Implemented an **AutoML iteration loop** (Optuna sweeps, MLflow registry, Airflow retraining) with **backtest promotion gates** blocking model releases on out-of-sample regression.
- Built a **grounded LLM research agent** with a generate–critique–revise loop that cites source passages for every claim and abstains when evidence is insufficient, holding unsupported-claim rate below X%.

## 12. How the Student Proposes This at Work

A short, concrete script — this is the step that makes the whole plan real:

1. **Lead with a pain point, not a technology.** "Finding the report where we discussed X takes me 20 minutes of keyword search" is a pitch; "I want to build a RAG system" is not.
2. **Bring a working demo on public data.** After Sprint 1 of the class track, he has a running search over SEC filings with measured numbers. A demo converts far better than a proposal.
3. **Ask for a small, bounded scope first.** One document type, one team, read-only. Expansion follows if it gets used.
4. **Clear data governance up front.** Ask explicitly what may be indexed and where it may be stored, before touching anything. This question earns trust rather than costing it.
5. **Instrument from day one.** Query counts, latency, analyst thumbs-up/down — so the résumé number comes from measurement, not estimation.

If the manager declines, the class-track version still stands on its own as a public-repo project; nothing is wasted.

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Manager declines the internal project | Class track is self-contained; falls back to a public-repo project entry with FiQA benchmark numbers |
| Internal data cannot be used | Class track is public-data-only by design; internal work stays internal and is described only in general terms |
| Less directly "recsys" for recommendation-focused roles | Retrieval + ranking transfers directly; also opens search-relevance and AIE roles |
| Backtest overfitting | Out-of-sample gates, walk-forward evaluation, turnover and cost accounting — the discipline is itself the teaching point |
| Corpus/ETL messiness eats the schedule | Fixed corpus scope agreed in Sprint 1; chunking strategy timeboxed |
| LLM cost | Cached queries, batched offline extraction, Ollama fallback; API budget < $20 |

## 14. Interview Leverage

- **Search / retrieval system design**: hybrid recall, reranking, ANN index families, product quantization mechanics, the accuracy–latency–memory frontier — with a three-year through-line back to the CAS patent implementation.
- **Applied ML judgment**: evaluating a model by out-of-sample backtest rather than in-sample accuracy is a maturity signal most new grads cannot demonstrate.
- **LLM engineering (AIE crossover)**: grounding enforcement, citation verification, abstention design, cost control — with the domain twist that being wrong is expensive.
- **SDE**: microservices, caching, P99 optimization, scheduled pipelines, monitoring and alerting.
- **Behavioral**: proposing, scoping, and shipping a tool inside a real organization, including data-governance conversations — genuinely rare in a new-grad candidate.
