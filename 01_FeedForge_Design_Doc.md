# FeedForge — Design Document

**LLM-Enhanced Multimodal Content Feed Recommendation System**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer Intern–grade project: end-to-end feed recommendation backend with LLM augmentation and production-style MLOps |

---

## 1. Overview

FeedForge is a news/short-content feed recommendation platform built as a **complete backend system**, not a standalone algorithm. It implements the industry-standard multi-stage pipeline — **multi-channel retrieval → CTR ranking → LLM re-ranking** — served through a microservice architecture with an online feature store, streaming event ingestion, offline/online evaluation, and monitoring.

The differentiating theme is **LLM-augmented recommendation**: using large language models for semantic feature generation, cold-start retrieval, and listwise re-ranking, mirroring how large tech companies are currently modernizing their recommendation stacks.

## 2. Goals & Non-Goals

### Goals
- Build a full retrieval → ranking → re-ranking pipeline on a real impression-log dataset.
- Integrate recommendation as a **service inside a backend system**: API gateway, feature store, candidate services, caching, latency budgets.
- Apply LLMs in three concrete, defensible roles: feature augmentation, cold-start semantic retrieval, and listwise re-ranking.
- Production-style MLOps: experiment tracking, model registry, containerized deployment, monitoring, scheduled retraining, A/B bucketing.
- Produce measurable results suitable for a resume: Recall@K / NDCG / AUC offline; P99 latency and QPS under load test.

### Non-Goals
- Training foundation models from scratch (LLMs are used via API or small open-weight models).
- Real user traffic (simulated via timestamped log replay — see §7.3).
- Kubernetes-grade infra; Docker Compose is sufficient, with K8s manifests as an optional stretch goal.

## 3. Dataset

| Dataset | Role | Notes |
|---|---|---|
| **MIND** (Microsoft News Dataset, MIND-small → MIND-large) | Primary | ~1M users, 160k news articles, **impression logs with explicit click/non-click labels** — natural CTR training data, no label fabrication needed. Includes title/abstract/category for content features. Public leaderboard available for benchmarking. |
| **MicroLens** (short-video, multimodal) | Optional upgrade | Adds real cover images + video metadata if a stronger multimodal story is desired. |
| **Criteo / Avazu** | Warm-up only | Pure tabular CTR data for a first DeepFM exercise before moving to MIND. |

**Train/eval split**: temporal (e.g., first N days for training, last day(s) for evaluation) to match industrial practice and avoid temporal leakage.

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (core), SQL |
| ML frameworks | PyTorch, scikit-learn |
| Recommendation models | Two-tower retrieval (in-batch negatives), DeepFM or DIN for CTR ranking |
| Embeddings | BGE / Sentence-BERT (text), CLIP (images, if MicroLens) |
| LLM | GPT-4o-mini / Claude Haiku via API, or Qwen2.5-7B-Instruct served locally via vLLM/Ollama |
| Vector search | FAISS (primary), Milvus (optional upgrade) |
| Serving | FastAPI (+ optional gRPC between internal services), Pydantic |
| Feature store / cache | Redis (online features, hot-user result cache) |
| Streaming | Kafka (or Redpanda) — event ingestion + log replay |
| Storage | Parquet + DuckDB (offline), PostgreSQL (metadata) |
| MLOps | MLflow (tracking + model registry), Airflow (retraining DAG), Docker Compose |
| Monitoring | Prometheus + Grafana (QPS, latency, cache hit rate, online CTR proxy) |
| Load testing | Locust |
| CI | GitHub Actions (lint, unit tests, image build) |

## 5. System Architecture

```
                        ┌────────────────────────────────────────────┐
  Log Replayer ──────►  │ Kafka (impression / click events)          │
                        └───────┬────────────────────────────────────┘
                                ▼
                     Stream consumer → feature updater ──► Redis (online feature store)
                                                            ▲
 Client ──► API Gateway (FastAPI)                           │
              │                                             │
              ▼                                             │
        Recommendation Orchestrator ────────────────────────┘
              │
              ├─► Retrieval Service ── two-tower ANN (FAISS)
              │                     ── content/semantic channel (embedding search)
              │                     ── popularity / recency channel
              │
              ├─► Ranking Service ──── DeepFM / DIN CTR model (top-500 → top-50)
              │
              ├─► LLM Re-rank Service ─ listwise re-rank of top-50 → top-10
              │                         (skippable via feature flag / timeout fallback)
              ▼
         Response (ranked feed + optional explanations)

 Offline: Parquet/DuckDB ─► training pipelines ─► MLflow registry ─► model rollout
 Ops:     Prometheus/Grafana dashboards, Airflow retraining DAG, Locust load tests
```

## 6. Component Design

### 6.1 Data & Feature Layer
- Parse MIND impression logs into `(user, item, context, label)` training records.
- Offline features: user click-history aggregates, category affinity, item CTR priors, precomputed text embeddings.
- Online features in Redis: rolling user history (last N clicks), real-time item popularity counters updated from the Kafka stream.
- Feature consistency: a shared feature-definition module used by both training and serving paths (talking point: training/serving skew).

### 6.2 Retrieval (Candidate Generation)
- **Two-tower model**: user tower (history-averaged / attention-pooled embeddings) × item tower (content + ID embeddings); trained with in-batch sampled softmax; served via a FAISS ANN index (corpus → top-500).
- **Semantic channel**: BGE embedding search over article text — doubles as the cold-start path for brand-new items.
- **Heuristic channels**: popularity and recency, merged with learned channels (talking point: multi-channel merge & dedup).

### 6.3 Ranking (CTR Model)
- DeepFM (baseline) or DIN (stretch: attention over the user behavior sequence).
- Trained directly on MIND's real click/non-click impression labels; metrics: AUC, GAUC, LogLoss.
- Exported to ONNX for faster CPU inference (optimization talking point).

### 6.4 LLM Augmentation (the "Modern RecSys" Layer)
1. **Feature augmentation (offline)**: an LLM generates semantic tags / interest labels / summaries per article; injected as categorical + embedding features into the ranker. Batched and cached — cost-controlled.
2. **Cold-start retrieval (online)**: new items with zero interactions are retrievable immediately via the semantic channel; LLM-generated user interest profiles ("likes fiscal-policy news and NBA trade rumors") improve matching.
3. **Listwise re-ranking (online)**: an LLM re-ranks top-50 → top-10 given a compact user profile and candidate summaries (RankGPT/LlamaRec-style prompting). Strict timeout (e.g., 300 ms budget) with fallback to CTR order; evaluated offline as NDCG delta vs. ranking-only.

### 6.5 Serving & System Design
- Service decomposition: gateway / orchestrator / retrieval / ranking / re-rank as separate FastAPI (or gRPC) services under Docker Compose.
- Redis caching of full results for hot users, with TTL + invalidation on new interactions.
- Latency engineering: batched inference in the ranker, precomputed embeddings, ONNX Runtime, connection pooling.
- **Deliverable metric**: sustained QPS and P50/P99 latency from Locust load tests, reported with/without cache and with/without the LLM stage.

### 6.6 MLOps
- MLflow: every training run tracked; models promoted through a registry (staging → production).
- Airflow DAG: weekly retrain → offline eval gate (AUC/Recall thresholds) → auto-promotion.
- A/B framework: deterministic user-hash bucketing; per-bucket online metrics (simulated CTR from replayed clicks) in Grafana.
- Monitoring: Prometheus metrics on QPS, latency percentiles, cache hit rate, model version, LLM fallback rate.

## 7. Evaluation Plan

### 7.1 Offline
- Retrieval: Recall@K, coverage, cold-start Recall@K (items unseen in training).
- Ranking: AUC / GAUC / LogLoss vs. LR and popularity baselines.
- Re-ranking: NDCG@10 / MRR delta from the LLM stage; ablation table (retrieval-only → +ranker → +LLM re-rank).

### 7.2 System
- Locust load test: QPS vs. P99 latency curves; cache-hit ablation.

### 7.3 Simulated Online
- **Timestamped log replay through Kafka** simulates production traffic: final-week events drive "online" metric dashboards and A/B bucket comparison. Honest framing: this simulates the serving loop, not real user feedback — itself a deliberate interview talking point (exposure bias, delayed feedback).

## 8. Milestones (mapped to 19 mentoring hours)

| # | Phase | Hours | Output |
|---|---|---|---|
| 0 | Kickoff & scoping | 1 | Repo, environment, dataset EDA assignment |
| 1 | RecSys fundamentals | 2 | Pipeline concepts, metrics; Criteo warm-up (optional) |
| 2 | Data + retrieval | 3 | Feature pipeline, two-tower + FAISS, semantic channel |
| 3 | Ranking + LLM augmentation | 4 | DeepFM/DIN, LLM features, cold-start, listwise re-rank |
| 4 | System integration | 4 | Microservices, Redis, caching, load-test numbers |
| 5 | MLOps | 3 | MLflow, Airflow retrain DAG, A/B buckets, Grafana |
| 6 | Resume & interview prep | 2 | Resume bullets, ML system design mock, deep-dive drill |

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM API cost | Batch + cache offline augmentation; small local model via Ollama as fallback |
| Infra scope creep | Docker Compose baseline; K8s/Milvus strictly stretch goals |
| MIND-large too heavy for a laptop | Start on MIND-small; sample MIND-large; DuckDB out-of-core processing |
| LLM re-rank latency | Hard timeout + fallback to CTR order; report fallback rate as a metric |

## 10. Interview Leverage

- **ML system design**: "Design a news/short-video feed" — the candidate has literally built one.
- **ML deep-dive**: negative sampling & exposure bias, two-tower vs. cross-feature models, training/serving skew, LLM-in-recsys trade-offs (latency/cost vs. NDCG).
- **SDE system design**: caching strategy, service decomposition, P99 latency optimization, graceful degradation of the LLM stage.
