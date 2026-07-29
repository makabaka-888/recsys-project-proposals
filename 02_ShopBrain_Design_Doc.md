# ShopBrain — Design Document

**Multimodal E-Commerce Recommendation Engine with LLM Conversational Shopping**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer Intern–grade project: search-and-recommendation e-commerce backend with sequential models, semantic IDs, and an LLM conversational entry point |

---

## 1. Overview

ShopBrain is an Amazon-style product recommendation engine built as a full backend service. It combines classical pipeline stages (retrieval → ranking) with two "new-generation" capabilities that differentiate it from textbook recsys projects:

1. **LLM query understanding & conversational recommendation** — a natural-language entry point ("I need a gift for my dad who runs marathons") that is parsed into structured intent, routed through retrieval/ranking, and answered with LLM-generated recommendation rationales.
2. **Semantic IDs (generative-recommendation trend)** — product embeddings discretized into semantic tokens (RQ-VAE / residual quantization), aligning the project with the generative recommendation direction pursued at Google/Meta/YouTube.

Multimodality comes from real product data: images, titles, descriptions, and reviews.

## 2. Goals & Non-Goals

### Goals
- End-to-end product recommendation: multi-channel retrieval, CTR/CVR-style ranking, personalized "similar items" and "buy-it-again" surfaces.
- Sequential recommendation (SASRec) over user purchase/review histories.
- LLM conversational layer: intent parsing → structured query → candidate retrieval → grounded, hallucination-checked recommendation responses.
- Semantic-ID prototype demonstrating generative retrieval on a subset of the catalog.
- Same production-style serving + MLOps bar as a real e-commerce team: microservices, Redis, vector DB, experiment tracking, monitoring, load tests.

### Non-Goals
- Payment/checkout/inventory systems (catalog + recommendation surfaces only).
- Real ad-auction or price optimization logic.
- Full-catalog generative retrieval (semantic-ID track is scoped to one product category).

## 3. Dataset

| Dataset | Role | Notes |
|---|---|---|
| **Amazon Reviews 2023** (McAuley Lab, UCSD) | Primary | 571M reviews, 48M items, with product **images, titles, descriptions, prices, categories**; widely used in recsys research. Work on 1–2 categories (e.g., *Electronics* + *Sports & Outdoors*) to stay laptop-friendly. |
| **Amazon-M2 / ESCI (Amazon KDD Cup)** | Optional | Query–product relevance data if a stronger search/query-understanding story is desired. |

**Label construction**: the dataset records interactions (reviews/ratings) rather than impressions, so ranking labels are built via **negative sampling** (random + popularity-corrected negatives; in-batch negatives for the two-tower). This is a deliberate teaching point: the model predicts interaction propensity, not true CTR — sample-selection bias and exposure bias are discussed and documented honestly.

**Split**: leave-one-out per user (standard for sequential recsys) plus a temporal holdout for pipeline-level evaluation.

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, SQL |
| ML frameworks | PyTorch, scikit-learn |
| Models | SASRec (sequential), two-tower retrieval, DeepFM ranker |
| Semantic IDs | RQ-VAE / residual vector quantization over item embeddings (single-category prototype) |
| Embeddings | CLIP (product images), BGE / Sentence-BERT (titles, descriptions, reviews) |
| LLM | GPT-4o-mini / Claude Haiku via API (intent parsing, response generation); Qwen2.5-7B via Ollama as a local fallback |
| Vector search | FAISS or Qdrant |
| Serving | FastAPI (REST) + Pydantic; optional gRPC between internal services |
| Cache / online store | Redis (session context, hot results, feature cache) |
| Storage | Parquet + DuckDB (offline), PostgreSQL (catalog metadata) |
| MLOps | MLflow, Airflow (or Prefect), Docker Compose, GitHub Actions CI |
| Monitoring | Prometheus + Grafana; structured logging of LLM token usage & fallback rate |
| Load testing | Locust |

## 5. System Architecture

```
 User ──► API Gateway (FastAPI)
            │
            ├── /feed, /similar-items, /you-may-also-like ────────┐
            │                                                     ▼
            │                                        Recommendation Orchestrator
            │                                             │
            │                                             ├─► Retrieval Service
            │                                             │     ├ two-tower ANN (FAISS/Qdrant)
            │                                             │     ├ SASRec next-item candidates
            │                                             │     ├ multimodal semantic search (CLIP+BGE)
            │                                             │     └ semantic-ID generative channel (prototype)
            │                                             │
            │                                             └─► Ranking Service (DeepFM)
            │
            └── /chat  ──► LLM Conversational Service
                             │  1. intent parsing → structured query (category,
                             │     price range, attributes, recipient, occasion)
                             │  2. calls Retrieval + Ranking with parsed constraints
                             │  3. grounded response generation with product cards
                             │     + recommendation rationales (grounding check:
                             │     only items returned by retrieval may be mentioned)
                             ▼
                       Response with citations to real catalog items

 Offline: ETL (Parquet/DuckDB) ─► training pipelines ─► MLflow registry
 Ops: Prometheus/Grafana, Airflow retraining, Locust load tests
```

## 6. Component Design

### 6.1 Catalog & Feature Layer
- ETL: raw Amazon Reviews 2023 → cleaned catalog (PostgreSQL) + interaction logs (Parquet).
- Item features: CLIP image embeddings, BGE text embeddings, price bucket, category path, rating stats, review-derived aspect tags (LLM-extracted offline: "battery life", "fits small").
- User features: interaction sequence, category affinity, price-sensitivity band.

### 6.2 Retrieval
- **Two-tower** (user × item) with in-batch negatives → FAISS top-500.
- **SASRec**: self-attention over the user's item sequence for next-item prediction; doubles as the "because you bought X" surface.
- **Multimodal semantic search**: text query or image → CLIP/BGE embedding → ANN; also the cold-start path for new products.
- **Semantic-ID prototype (stretch, one category)**: quantize item embeddings into token sequences (RQ-VAE), train a small transformer to generate semantic IDs conditioned on user history — a discussable generative-recommendation demo, benchmarked against the two-tower on the same category.

### 6.3 Ranking
- DeepFM over user, item, context, and cross features (including LLM-extracted aspect tags).
- Labels from negative sampling per §3; metrics AUC / NDCG@10 / HitRate@10 with popularity and BPR-MF baselines.

### 6.4 LLM Conversational Layer
- **Intent parser**: LLM → JSON schema (validated by Pydantic; retry-on-invalid) capturing category, budget, attributes, recipient, occasion.
- **Grounded generation**: responses may only reference product IDs returned by retrieval; a post-generation grounding check rejects hallucinated items (talking point: hallucination control in LLM products).
- **Multi-turn session state** in Redis: refinements like "cheaper" or "in black" mutate the structured query rather than restarting.
- Token/cost accounting logged per request.

### 6.5 Serving & System Design
- Gateway / orchestrator / retrieval / ranking / chat as separate services; Docker Compose.
- Redis: session store + hot-item cache + result cache with TTL.
- Latency budget: non-chat surfaces P99 < 150 ms (no LLM in the hot path); chat surface streams tokens for perceived latency.
- Load-test report: QPS vs. latency for /feed and /similar-items; concurrency behavior of the chat service.

### 6.6 MLOps
- MLflow experiment tracking + model registry for SASRec / two-tower / DeepFM.
- Airflow: nightly incremental ETL + weekly retrain with an offline eval gate.
- A/B bucketing by user hash; dashboard comparing ranker versions on replayed interactions.
- Prometheus/Grafana: service metrics + LLM-specific metrics (token spend, grounding-check failure rate, intent-parse retry rate).

## 7. Evaluation Plan

- **Retrieval**: Recall@K / coverage per channel; cold-start recall via held-out new items.
- **Sequential**: HitRate@10, NDCG@10 (leave-one-out), vs. GRU4Rec/BPR baselines.
- **Ranking**: AUC, NDCG@10; ablation with/without multimodal and LLM-derived features.
- **Semantic-ID prototype**: Recall@K vs. two-tower on the same category; qualitative analysis of token semantics.
- **Conversational**: intent-parse accuracy on a hand-labeled set (~200 queries); grounding-violation rate; human-eval rubric (relevance/helpfulness) on ~50 conversations.
- **System**: Locust QPS / P99 report.

## 8. Milestones (mapped to 19 mentoring hours)

| # | Phase | Hours | Output |
|---|---|---|---|
| 0 | Kickoff & scoping | 1 | Repo, environment, category selection, EDA assignment |
| 1 | RecSys fundamentals | 2 | Pipeline concepts, metrics, negative-sampling theory |
| 2 | ETL + retrieval | 3 | Catalog ETL, two-tower + FAISS, multimodal search |
| 3 | Sequential + ranking | 4 | SASRec, DeepFM, ablations; (stretch) semantic-ID prototype |
| 4 | LLM conversational layer | 3 | Intent parser, grounded generation, session state |
| 5 | System integration + MLOps | 4 | Microservices, Redis, load tests, MLflow/Airflow, dashboards |
| 6 | Resume & interview prep | 2 | Resume bullets, ML system design mock, deep-dive drill |

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset size (571M reviews) | Scope to 1–2 categories; DuckDB out-of-core; pre-filtered subsets published by McAuley Lab |
| Semantic-ID track too research-heavy | Explicitly a stretch goal; project is complete without it |
| No impression logs → biased labels | Documented negative-sampling strategy; discussed as an interview strength, not hidden |
| LLM hallucination in chat | Grounding check + constrained generation; violation rate reported as a metric |
| LLM API cost | Aggressive caching of intent parses; local Qwen fallback via Ollama |

## 10. Interview Leverage

- **ML system design**: "Design Amazon's recommendations / a shopping assistant" — direct experience.
- **ML deep-dive**: negative sampling & exposure bias, sequential vs. matrix-factorization models, multimodal fusion, semantic IDs & generative retrieval (a current-frontier topic interviewers notice).
- **SDE/system design**: session-state management, streaming responses, cache hierarchy, keeping LLMs out of the P99-critical path.
- **AIE crossover**: structured-output parsing, grounding/hallucination control, LLM cost engineering — lets the same project serve AI Engineer applications.
