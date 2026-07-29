# SearchCraft — Design Document

**E-Commerce Search Relevance Engine with API-First, End-to-End Service Design**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer / SDE dual-track project: human-labeled search relevance modeling (hybrid retrieval + learning-to-rank + LLM augmentation) wrapped in a production-grade, contract-first public API |

---

## 1. Overview

SearchCraft is an e-commerce product search platform built the way a real search team ships one: an **API-first service** whose contract is designed before any model is trained, backed by a relevance stack of **hybrid retrieval (lexical + semantic) → learning-to-rank → LLM re-ranking**, and closed with a **feedback-ingestion loop** that turns client-reported clicks into future ranking features.

Two properties distinguish this proposal from the other three:

1. **Human-labeled relevance data.** The primary dataset (Amazon ESCI) ships four-grade human relevance judgments per query–product pair, so ranking models train on ground-truth labels — no negative-sampling caveats.
2. **End-to-end API design as a first-class deliverable.** The project produces a versioned, documented, load-tested public API (OpenAPI 3.1 spec, cursor pagination, structured error model, rate limiting, idempotent ingestion), giving the student concrete SDE interview material that pure-ML projects lack.

## 2. Goals & Non-Goals

### Goals
- Design the full public API surface **contract-first** (OpenAPI spec written and reviewed before implementation; contract tests enforce it in CI).
- Hybrid retrieval: BM25 (OpenSearch) + dense bi-encoder (BGE + FAISS), fused with Reciprocal Rank Fusion.
- Learning-to-rank: LightGBM LambdaMART (industry mainstay) vs. a fine-tuned cross-encoder re-ranker; benchmarked on the public ESCI leaderboard task.
- LLM augmentation: query rewriting/expansion, LLM-as-relevance-judge for training-data distillation, listwise re-ranking.
- Feedback loop: a click/impression ingestion API whose events become CTR-derived ranking features — closing the search quality flywheel.
- Production bar: versioned API, auth, rate limiting, monitoring, load tests, CI with contract tests, MLOps for ranker retraining.

### Non-Goals
- Cart/checkout/payment functionality (search and catalog surfaces only).
- Crawling or catalog acquisition (datasets provide the corpus).
- Multi-tenant SaaS concerns (single-tenant assumed; discussed as an extension in the API design section).

## 3. Dataset

| Dataset | Role | Notes |
|---|---|---|
| **Amazon ESCI / Shopping Queries Dataset** (KDD Cup 2022) | Primary | ~130k real shopping queries, 2.6M query–product pairs, **four-grade human labels** (Exact / Substitute / Complement / Irrelevant), en/es/jp. Official tasks + public baselines enable leaderboard comparison. Product text (title, description, bullet points, brand, color) drives both lexical and semantic indexing. |
| **WANDS** (Wayfair) | Secondary / generalization check | 48k query–product pairs, three-grade labels; used to test whether the ranking stack transfers across catalogs. |
| **Click simulation** | Feedback-loop data | Synthetic click logs generated from ESCI labels via a position-biased click model (examination hypothesis: P(click) = P(examine at rank) × P(relevant)); clearly documented as simulated. Drives the feedback-ingestion API and CTR features, and enables a position-bias / counterfactual-learning discussion. |

**Split**: official ESCI train/test split for leaderboard comparability; additional query-level holdout for pipeline-level evaluation.

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, SQL |
| Lexical search | OpenSearch (BM25, filters, aggregations, highlighting) |
| Dense retrieval | BGE embeddings + FAISS (or OpenSearch k-NN for a single-engine variant) |
| Ranking | LightGBM (LambdaMART); cross-encoder re-ranker (BGE-reranker fine-tuned on ESCI) |
| LLM | GPT-4o-mini / Claude Haiku via API (query rewriting, relevance judging, listwise re-rank); Qwen2.5-7B via Ollama fallback |
| API framework | FastAPI + Pydantic v2; **OpenAPI 3.1 spec as the source of truth** |
| API tooling | Schemathesis (contract/property tests), Spectral (spec linting), ReDoc (docs site) |
| Auth & rate limiting | API keys + Redis token-bucket rate limiter |
| Cache / counters | Redis (hot-query cache, rate-limit state, CTR counters) |
| Eventing | Kafka (feedback event ingestion → feature aggregation) |
| Storage | PostgreSQL (catalog, API keys), Parquet + DuckDB (offline training data) |
| MLOps | MLflow (tracking + registry), Airflow (retrain + index rebuild DAGs), Docker Compose, GitHub Actions CI |
| Monitoring | Prometheus + Grafana (QPS, latency percentiles, cache hit rate, zero-result rate, API-error rates) |
| Load testing | Locust |

## 5. System Architecture

```
 Client ──► API Gateway (FastAPI)
              │   auth (API key) · rate limit (Redis token bucket)
              │   request validation (Pydantic ⇄ OpenAPI contract)
              │
              ├── GET  /v1/search ────► Query Understanding Service
              │                           spell-correct · LLM rewrite/expansion
              │                           filter parsing (category, price, brand)
              │                                │
              │                           Hybrid Retrieval
              │                             ├ BM25 (OpenSearch)
              │                             └ dense ANN (BGE + FAISS)
              │                             → RRF fusion → top-200
              │                                │
              │                           Ranking Service
              │                             LambdaMART features:
              │                               BM25 score · embedding sim · CTR priors
              │                               category match · price/brand signals
              │                             (optional cross-encoder / LLM re-rank of top-50)
              │                                ▼
              │                           paginated, faceted results
              │
              ├── GET  /v1/suggest ───► prefix + popularity + embedding suggestions
              ├── POST /v1/events ────► Kafka ─► aggregator ─► CTR features (Redis/Parquet)
              └── /v1/admin/* ────────► index management, relevance experiments

 Offline: ESCI/WANDS ─► feature builder ─► LambdaMART & cross-encoder training
          ─► MLflow registry ─► Airflow rollout (index rebuild + model promotion)
 Ops: Prometheus/Grafana · Locust · GitHub Actions (lint, tests, contract tests)
```

## 6. API Design (First-Class Deliverable)

The API is designed **contract-first**: the OpenAPI 3.1 spec is written in week 1, reviewed like a real design doc, then enforced by Schemathesis contract tests in CI. Implementation drift fails the build.

### 6.1 Endpoint Surface

| Method & Path | Purpose | Notes |
|---|---|---|
| `GET /v1/search` | Core search | `q`, `filter` (structured), `sort`, `cursor`, `limit`, `facets`; returns ranked hits + facet counts + `next_cursor` |
| `GET /v1/suggest` | Typeahead | prefix + popularity + semantic suggestions; strict latency budget (P99 < 40 ms) |
| `GET /v1/products/{id}` | Item lookup | canonical product resource referenced by search hits |
| `POST /v1/events` | Feedback ingestion | batched impression/click/add-to-cart events; **idempotent via `Idempotency-Key` header**; 202 Accepted + async processing |
| `POST /v1/admin/indexes/{name}/rebuild` | Index lifecycle | async job pattern: returns a job resource; `GET /v1/admin/jobs/{id}` polls status |
| `POST /v1/admin/experiments` | Relevance experiments | register ranker variants + traffic split for A/B bucketing |

### 6.2 Contract Decisions (teaching material — each an interview talking point)

- **Versioning**: URL-path major version (`/v1/`); additive changes only within a major; deprecation policy documented in the spec.
- **Pagination**: opaque cursor (encoded search-after state), not offset — correct under live index updates and cheap at depth; offset pagination's failure modes discussed explicitly.
- **Filtering**: structured filter grammar (`filter=category:electronics AND price:[10,100]`) parsed server-side into OpenSearch DSL — never raw query passthrough (injection surface).
- **Error model**: RFC 9457 `application/problem+json` envelope with stable machine-readable `type` codes; validation errors enumerate field paths.
- **Rate limiting**: per-key token bucket in Redis; `429` with `Retry-After`; limits advertised via `X-RateLimit-*` headers.
- **Idempotency**: `Idempotency-Key` on event ingestion; duplicate submissions return the original result — at-least-once client retries stay safe.
- **Async jobs**: long-running operations (index rebuild) return `202` + a job resource instead of blocking — the standard long-task API pattern.
- **Auth**: API keys with scopes (`search:read`, `events:write`, `admin:*`); key hashing at rest.
- **Observability contract**: every response carries `X-Request-Id`; structured logs join gateway → retrieval → ranking spans per request.

### 6.3 Example Exchange (synthetic values)

```http
GET /v1/search?q=wireless+earbuds+for+running&filter=price:[20,80]&limit=2 HTTP/1.1
X-Api-Key: sk_demo_xxx
```

```json
{
  "query": {
    "raw": "wireless earbuds for running",
    "rewritten": "wireless sport earbuds sweatproof",
    "applied_filters": [{"field": "price", "range": [20, 80]}]
  },
  "hits": [
    {"product_id": "B0DEMO123", "title": "AquaBeat Sport Earbuds", "score": 12.41,
     "price": 49.99, "rank_features": {"bm25": 8.2, "dense_sim": 0.83, "ctr_prior": 0.061}},
    {"product_id": "B0DEMO456", "title": "RunTone Wireless Earbuds", "score": 11.87,
     "price": 39.99, "rank_features": {"bm25": 7.9, "dense_sim": 0.81, "ctr_prior": 0.054}}
  ],
  "facets": {"brand": [{"value": "AquaBeat", "count": 14}]},
  "next_cursor": "eyJzZWFyY2hfYWZ0ZXIiOlsxMS44NywiQjBERU1PNDU2Il19",
  "request_id": "req_01J8DEMO"
}
```

`rank_features` in responses is a debug-mode option — gating it per key scope is itself a nice API-design discussion (leaking ranking internals vs. partner debuggability).

## 7. Component Design

### 7.1 Indexing Pipeline
- ETL: ESCI product corpus → normalized catalog (PostgreSQL) → OpenSearch index (language-aware analyzers, synonyms) + FAISS dense index (BGE embeddings, precomputed in batch).
- Index rebuild as an Airflow DAG triggered via the admin API; blue/green index alias swap for zero-downtime rollover (talking point).

### 7.2 Query Understanding
- Spell correction (symspell) → LLM query rewrite/expansion, cached by normalized query (high hit rate: query distributions are Zipfian — a talking point on cache design).
- Structured filter parsing with strict validation (no raw DSL passthrough).

### 7.3 Retrieval & Ranking
- Hybrid: BM25 top-100 ∪ dense top-100 → RRF → top-200.
- **LambdaMART** over ~20 features (lexical scores, embedding similarity, CTR priors from the feedback loop, category/brand/price match, product quality signals). Trained on ESCI graded labels with an NDCG objective.
- **Cross-encoder** re-ranker fine-tuned on ESCI as the neural comparison; **LLM listwise re-rank** of top-20 as the modern-stack option (timeout + fallback, same pattern as the other proposals).
- Ablation table: BM25-only → +dense → +LambdaMART → +cross-encoder/LLM.

### 7.4 LLM-as-Judge Distillation
- An LLM labels additional query–product pairs on the ESCI rubric; agreement with human labels is measured (Cohen's κ) before distilled data augments training — mirrors how industry teams scale relevance labeling today, and honestly quantifies LLM-judge reliability.

### 7.5 Feedback Loop
- `/v1/events` → Kafka → aggregation job → per-(query, product) CTR priors with position-bias correction (inverse-propensity weighting using the click model from §3).
- Closes the loop: the next LambdaMART retrain consumes CTR features; A/B buckets compare ranker versions on replayed simulated clicks.

## 8. Evaluation Plan

- **Relevance (offline)**: NDCG@10 / MRR / ERR on the official ESCI test split, compared against published KDD Cup baselines; WANDS transfer evaluation.
- **Ablations**: retrieval fusion variants; LambdaMART feature-importance analysis (interpretable — good interview material).
- **LLM components**: rewrite win-rate (side-by-side NDCG with/without rewriting); LLM-judge agreement κ vs. human labels.
- **Search-quality ops metrics**: zero-result rate, suggest acceptance rate (simulated).
- **API/system**: Locust load test — QPS vs. P50/P99 for `/v1/search` and `/v1/suggest`; rate-limiter correctness under burst; contract-test pass rate in CI.

## 9. Milestones (mapped to 19 mentoring hours)

| # | Phase | Hours | Output |
|---|---|---|---|
| 0 | Kickoff & scoping | 1 | Repo, environment, ESCI EDA assignment |
| 1 | **API contract design** | 2 | OpenAPI spec authored + reviewed; error model, pagination, versioning decided; contract tests scaffolded |
| 2 | Indexing + hybrid retrieval | 3 | OpenSearch + FAISS indexes, RRF fusion, `/v1/search` v0 |
| 3 | Learning-to-rank | 4 | LambdaMART + cross-encoder, ESCI benchmark numbers, ablations |
| 4 | LLM augmentation | 3 | Query rewrite, LLM-judge distillation with κ analysis, listwise re-rank |
| 5 | Feedback loop + ops | 4 | `/v1/events` + Kafka + CTR features, rate limiting, monitoring, load tests, Airflow retrain |
| 6 | Resume & interview prep | 2 | Resume bullets, search system design mock, API-design deep-dive drill |

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Two engines (OpenSearch + FAISS) = infra overhead | Single-engine variant: OpenSearch k-NN plugin serves both lexical and dense; FAISS optional |
| ESCI scale (2.6M pairs) | Start with the en-US "small" task subset; full set only for final training runs |
| Simulated clicks mistaken for real feedback | Explicitly documented click model; framed as a position-bias/counterfactual teaching device |
| Contract-first feels slow to the student | Timeboxed to 2 hours with a reviewed template; payoff made explicit in mock interviews |
| LLM cost | Zipfian query cache; local Qwen fallback; judge distillation batched offline |

## 11. Interview Leverage

- **Search system design**: "Design product search for an e-commerce site" — a top-frequency question at Amazon/Airbnb/DoorDash/Instacart search teams; the candidate has built every stage, including the feedback loop.
- **API design interviews** (common SDE round): versioning, cursor vs. offset pagination, idempotency, rate limiting, async job patterns, error contracts — all implemented, not just recited.
- **ML deep-dive**: learning-to-rank objectives (pointwise/pairwise/listwise, LambdaMART), hybrid retrieval trade-offs, position bias & counterfactual learning, LLM-as-judge reliability measurement.
- **Differentiation vs. the other three proposals**: human-labeled ground truth (no negative-sampling caveat), and the strongest pure-SDE story of the four — the natural pick if the student weights SDE ≥ MLE.
