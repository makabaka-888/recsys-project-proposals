# SnapShop — Design Document

**Visual-First Multimodal Product Discovery & Recommendation Platform**

| | |
|---|---|
| Status | Proposal (replaces ShopBrain) |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer–grade work experience: quantized multimodal retrieval, two-stage ranking funnel, automated model iteration, and an LLM shopping agent — served as production-style microservices |

---

## 1. Why This Project (Alignment Thesis)

Every module below is a **direct continuation of work the candidate has already done**. The résumé stops reading like a list of unrelated internships and starts reading like one engineer going deeper on one problem for three years.

| Prior experience | Skill already demonstrated | SnapShop module that extends it |
|---|---|---|
| **CAS** — implemented a patented cascaded binary-encoding image-retrieval method (PCA hash projections); indexed 50k+ images as binary signatures; benchmarked Precision@K / latency / storage across 20+ configurations | Embedding retrieval, **quantization**, ANN index benchmarking | **Visual retrieval layer**: CLIP embeddings + FAISS **IVF-PQ / HNSW / binary-hash** benchmark on Recall@K vs. P99 latency vs. memory. *Same methodology, modern industrial stack, 10× the corpus.* |
| **Changsheng Fund** — two-stage weekly strategy (sector rotation → stock selection); AutoML agent generating hyper-parameters; backtested excess return after costs | **Multi-stage funnel design**, automated tuning, disciplined offline evaluation | **Retrieval → ranking funnel** + **Optuna sweep agent** driving model iteration + offline replay evaluation with lift over baselines |
| **Sina Weibo** — scheduled SQL ETL, standardized engagement metrics, monitoring pipelines with influencer scoring and alerting | Data pipelines, metric standardization, **monitoring & alerting** | **Airflow feature pipelines** over 30M+ transactions + Redis online store + **data-quality and embedding-drift monitors that alert** |
| **DiDi** — Java/Servlet backend modules; **A/B tests on page-layout changes** | Backend engineering, **experimentation** | **FastAPI microservices** + deterministic user-hash **A/B bucketing** with guardrail metrics |
| **College Application Agent** — EBR retrieval behind a versioned FastAPI endpoint; LangGraph **Writer–Critic** loop | LLM engineering, API design, self-critique loops | **LLM shopping agent**: intent parsing → constrained retrieval → **generate–critique–revise** loop with grounding enforcement |
| **Music Genre Classification** — PCA to 9 components, K-means cluster features | Dimensionality reduction, clustering | Conceptual bridge to **product quantization** (PQ = clustering in subspaces) and cold-start segmentation |

**One-line interview answer to "why this work?"**
*"At CAS I built image retrieval with hand-rolled binary hashing and measured the accuracy–memory–latency trade-off. SnapShop is that same problem at production scale: modern multimodal embeddings, a proper ANN index, and everything downstream of retrieval — ranking, an LLM agent, serving, and monitoring."*

## 2. Overview

SnapShop is a **visual-first product discovery platform** for fashion e-commerce — the pattern behind Pinterest Lens, Amazon StyleSnap, and Alibaba Pailitao. Users can search **by photo, by text, or by both**, receive personalized recommendations, and refine through a conversational agent.

Behind those three surfaces sits a full recommendation stack: a quantized multimodal vector index, a multi-channel retrieval layer, a learned ranker, an LLM agent with a critic loop, and the operational machinery (scheduled pipelines, experiment tracking, A/B bucketing, monitoring, load testing) that makes it a system rather than a notebook.

## 3. Goals & Non-Goals

### Goals
- **Visual similarity search**: photo → visually and semantically similar products, with cold-start support for items that have zero purchase history.
- **Quantized index engineering as a headline deliverable**: systematic benchmark of IVF-PQ / HNSW / binary hashing on Recall@K, P99 latency, and index memory — a defensible trade-off table, not a single number.
- **Two-stage funnel**: multi-channel retrieval (visual, semantic, behavioral two-tower, sequential) → learned ranker over multimodal + behavioral features.
- **Automated model iteration**: Optuna sweeps + MLflow registry + Airflow retraining with evaluation gates that block regressions.
- **LLM shopping agent** with structured intent parsing and a generate–critique–revise loop that grounds every recommendation in retrieved catalog items.
- **Production-style serving**: FastAPI microservices, Redis caching, A/B bucketing, Prometheus/Grafana monitoring, Locust load tests.

### Non-Goals
- Training vision or language foundation models from scratch (CLIP/BGE used as frozen encoders; LLMs via API or small local models).
- Checkout, payments, inventory.
- Real production traffic — the serving loop is exercised by timestamped log replay, and this is stated plainly wherever results are reported.

## 4. Dataset

| Dataset | Role | Notes |
|---|---|---|
| **H&M Personalized Fashion Recommendations** (Kaggle, 2022) | **Primary** | ~105k articles **with product images**, ~1.37M customers, ~31.8M transactions spanning 2018-09 → 2020-09, plus rich metadata (product type, colour, department, garment group, text descriptions). The rare public dataset with **images and real transaction logs together** — exactly what a visual recommender needs. Competition metric MAP@12 gives a public benchmark to compare against. |
| **Amazon Reviews 2023** (McAuley Lab) | Secondary / generalization | Cross-catalog check that the retrieval stack transfers beyond fashion. |

**Scoping for a laptop**: one `index_group` (e.g. Ladieswear) → ~35–40k articles; last 6–12 months of transactions → ~3–5M rows; sampled customer cohort. Images downloaded only for the scoped subset.

**Label construction**: transactions are purchases, not impressions, so ranking labels use **negative sampling** (random + popularity-corrected; in-batch for the two-tower). The model predicts purchase propensity, not true CTR — documented explicitly, and the exposure-bias discussion is treated as an interview asset rather than a weakness.

**Split**: temporal (train on earlier weeks, evaluate on the final weeks) — matching the competition setup and avoiding leakage.

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, SQL |
| ML | PyTorch, scikit-learn, LightGBM |
| Encoders | **CLIP** (product images), **BGE / Sentence-BERT** (titles, descriptions) — inference only, embeddings precomputed |
| Retrieval models | Two-tower (in-batch negatives), SASRec (sequential) |
| **Vector index** | **FAISS** — Flat baseline, **IVF-PQ**, HNSW, binary (LSH/ITQ-style) — the benchmark is itself a deliverable |
| Ranking | DeepFM (or LightGBM ranker) over multimodal + behavioral + cross features |
| LLM | GPT-4o-mini / Claude Haiku via API; Qwen2.5-7B via Ollama as local fallback |
| Serving | FastAPI + Pydantic microservices; Redis (online features, session, result cache) |
| Pipelines | **Airflow** (ETL, embedding jobs, retraining DAGs), DuckDB + Parquet, PostgreSQL |
| Experimentation | **Optuna** (hyper-parameter sweeps), **MLflow** (tracking + model registry) |
| Ops | Docker Compose, Prometheus + Grafana, Locust, GitHub Actions CI |

## 6. Architecture

```
 Client
   ├── POST /v1/search/visual   image  → visually similar products
   ├── GET  /v1/search          text   → semantic + lexical results
   ├── GET  /v1/feed            personalized recommendations
   ├── POST /v1/chat            conversational refinement
   └── POST /v1/events          impression / click / purchase feedback
                    │
            API Gateway (FastAPI)  ·  auth · rate limit · request_id
                    │
              Orchestrator
                    ├─► RETRIEVAL SERVICE
                    │     ├ visual channel      CLIP  → FAISS IVF-PQ
                    │     ├ semantic channel    BGE   → FAISS
                    │     ├ behavioral channel  two-tower → FAISS
                    │     └ sequential channel  SASRec next-item
                    │        → reciprocal rank fusion → top-500
                    │
                    ├─► RANKING SERVICE
                    │     DeepFM / LightGBM over visual, text, behavioral,
                    │     price, popularity and cross features → top-50
                    │
                    └─► LLM AGENT SERVICE
                          intent parse (Pydantic) → constrained retrieval
                          → generate → CRITIC check (grounding, constraints)
                          → revise → response      [timeout → fallback]

 Offline:  Airflow DAGs → ETL → embedding batch jobs → feature store (Redis/Parquet)
           → training → Optuna sweeps → MLflow registry → gated promotion
 Ops:      Prometheus/Grafana · drift & data-quality alerts · A/B bucketing · Locust
```

## 7. Component Design

### 7.1 Vector Index Engineering — the CAS continuation
The flagship technical deliverable, and the module most directly inherited from prior work.

- Encode the scoped catalog with CLIP (image) and BGE (text); store float32 as the baseline.
- Build and benchmark index families against identical ground truth (exact Flat search):
  - **Flat** — exact; the accuracy ceiling and the latency reference line.
  - **IVF-PQ** — sweep `nlist`, `nprobe`, subquantizer count `m`, bits per code.
  - **HNSW** — sweep `M`, `efConstruction`, `efSearch`.
  - **Binary hashing** — the modern echo of the CAS method, included deliberately for the comparison narrative.
- Report a trade-off table across 20+ configurations: **Recall@50 vs. queries/sec vs. index memory**, plus compression ratio (512-dim float32 = 2 KB/vector → PQ64 = 64 bytes ≈ **32× smaller**).
- Deliverable: a Pareto-frontier plot and a documented "chosen configuration and why" — mirroring the CAS accuracy–efficiency selection, with an order of magnitude more data and an industrial index library.

### 7.2 Multi-Channel Retrieval
- **Two-tower**: user tower (purchase-history pooling + customer attributes) × item tower (ID + CLIP + BGE embeddings), in-batch sampled softmax, tuned temperature.
- **Sequential (SASRec)**: next-article prediction over purchase sequences; powers "wear it with" / "you might buy next".
- **Cold-start path**: brand-new articles have no transactions but do have images and text — retrievable from the first minute via the visual/semantic channels. Cold-start Recall@K is measured and reported separately.
- **Fusion**: reciprocal rank fusion + dedup across channels; per-channel contribution logged for attribution analysis.

### 7.3 Ranking
- DeepFM (or a LightGBM ranker as the interpretable comparison) over:
  - user features (category affinity, price band, recency/frequency, sales-channel mix)
  - item features (compressed CLIP/BGE embeddings, colour, garment group, price, popularity priors)
  - cross features (user affinity × item attribute, retrieval channel and channel score)
- Trained with popularity-corrected negative sampling; evaluated by AUC, NDCG@10, and MAP@12 against the public competition benchmark.

### 7.4 Automated Model Iteration — the Changsheng continuation
- **Optuna** sweeps over embedding dimension, learning rate, negative-sampling ratio, tower depth, and PQ settings; every trial logged to MLflow.
- Airflow DAG: `incremental ETL → embed new items → retrain → offline eval → gate (no metric regression) → promote or alert`.
- Directly parallels the AutoML hyper-parameter agent built at Changsheng, and the "backtest before you promote" discipline of quantitative research.

### 7.5 LLM Shopping Agent — the College-Agent continuation
- **Intent parsing**: natural language → Pydantic-validated JSON (`category, colour, price_range, occasion, style_attributes, size`), retry-on-invalid.
- **Constrained retrieval**: parsed constraints become hard filters plus embedding search — the agent can only ever see real catalog items.
- **Critic loop** (direct descendant of the LangGraph Writer–Critic pattern): a critic pass verifies that (a) every referenced article ID exists in the retrieved candidate set, (b) stated attributes match catalog metadata, (c) all user constraints are satisfied. Failures trigger one revision, then fall back to a template response.
- Metrics: intent-parse field-level F1, **grounding-violation rate**, constraint-satisfaction rate, tokens and cost per request.

### 7.6 Pipelines, Serving & Monitoring — the Weibo + DiDi continuation
- **Airflow ETL** over the transaction log: sessionization, standardized engagement/purchase metrics, feature materialization to Redis (online) and Parquet (offline) — the same "scheduled ETL + standardized metrics" pattern as Weibo, one abstraction level up.
- **Monitoring & alerting**: data-quality checks (null rates, schema drift, volume anomalies) and **embedding/feature drift** detection with alert rules — the direct analogue of Weibo's risk-post monitoring pipelines.
- **A/B framework**: deterministic user-hash bucketing, variant→model-version config, guardrail metrics — extending DiDi's page-layout A/B testing from UI to models.
- **Serving**: microservice decomposition, Redis result caching with TTL + invalidation, batched inference, LLM kept off the P99-critical path with timeout fallback; Locust load-test report (QPS vs. P50/P99, with and without cache).

## 8. Evaluation Plan

| Area | Metrics |
|---|---|
| Index engineering | Recall@50 vs. QPS vs. memory across 20+ configs; compression ratio; Pareto plot |
| Retrieval | Recall@100/500 per channel and fused; **cold-start Recall@K**; channel attribution |
| Ranking | AUC, NDCG@10, **MAP@12 vs. public leaderboard**; ablations (−visual, −text, −sequence) |
| LLM agent | Intent-parse F1, grounding-violation rate, constraint-satisfaction rate, cost/request |
| System | QPS, P50/P99 latency, cache hit rate, LLM fallback rate |
| Ops | Retraining gate pass/fail history; drift alerts fired vs. injected anomalies |

## 9. Milestones (mapped to the existing 19-hour, 15-session plan)

| Sprint | Sessions | Deliverable |
|---|---|---|
| **S0** — Setup | 3–4 | Agile workflow + AI tooling; repo, environment, dataset scoped |
| **S1** — Data & visual retrieval | 5–8 | Airflow ETL, CLIP/BGE embedding jobs, **index benchmark table**, `/search/visual` live |
| **S2** — Behavioral retrieval & ranking | 9–10 | Two-tower + SASRec channels, DeepFM ranker, MAP@12 benchmark, Optuna sweeps |
| **S3** — LLM agent | 11–12 | Intent parsing, constrained retrieval, critic loop, agent metrics |
| **S4** — Serving, MLOps & experimentation | 13–14 | Microservices, caching, A/B bucketing, monitoring/alerts, load tests |
| — | 15 | Interview sprint |

The 15-session / 19-hour structure from the existing course outline is unchanged; only module content shifts.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| H&M image set is large (~25–35 GB full) | Scope to one index group (~8–10 GB); use resized images; embeddings computed once and cached |
| CPU-only CLIP encoding is slow | One-time batch job on a free Colab/Kaggle GPU (~40k images in minutes); embeddings committed to disk |
| 31.8M transactions exceed RAM | DuckDB out-of-core over Parquet; temporal + customer-cohort subsampling |
| Index benchmark balloons in scope | Config grid fixed up front; automated sweep script rather than manual runs |
| Purchases ≠ impressions (label bias) | Documented negative-sampling strategy; exposure bias discussed openly as interview material |
| LLM cost | Cached intent parses, batched offline attribute extraction, Qwen-via-Ollama fallback; total API budget < $20 |

## 11. Résumé Bullets (experience voice; quantities filled in as sprints land)

- Built a **multimodal retrieval service** indexing 100k+ product images and descriptions as CLIP/BGE embeddings in FAISS, benchmarking **IVF-PQ, HNSW, and binary-hash configurations across 20+ settings** to select an index sustaining **X% Recall@50 at P99 < X ms with X× lower memory** than exact search.
- Designed a **two-stage recommendation funnel** — visual, semantic, two-tower behavioral, and sequential (SASRec) retrieval merged by reciprocal rank fusion, followed by a DeepFM ranker over multimodal and behavioral features — improving **MAP@12 by X%** over a popularity baseline.
- Automated model iteration with **Optuna hyper-parameter sweeps and MLflow tracking across X runs**, wired into **Airflow retraining DAGs with evaluation gates** that block promotion on metric regression.
- Built **scheduled Airflow ETL and feature pipelines** over 30M+ transaction records with a Redis online feature store, plus **data-quality and embedding-drift monitors** that alert on distribution shift.
- Developed an **LLM shopping agent** with Pydantic-validated intent parsing and a **generate–critique–revise loop** grounding every recommendation in retrieved catalog items, holding **hallucinated product mentions below X%**.
- Deployed the platform as **FastAPI microservices** with Redis caching and deterministic **user-hash A/B bucketing**, sustaining **X QPS at P99 < X ms** under Locust load testing with Prometheus/Grafana dashboards.

*(Use 4–5 of the six; keep #1 and #5 — they carry the strongest continuity with CAS and the College Application Agent.)*

## 12. Placement Note (read before putting this under EXPERIENCE)

The bullets above are written in work-experience voice and will read correctly in an experience section — **provided the entry names a real organization and a real engagement**. Legitimate options:

1. **The mentoring organization as the employer**, with a truthful title (`ML Engineering Practicum` / `Machine Learning Intern`) — valid where an actual engagement record exists.
2. **A real collaboration** — a startup, lab, or open-source org that agrees to host the work.
3. **Open-source project entry** — a public GitHub org with real commit history is verifiable and increasingly well regarded.

What must be avoided is inventing an employer or dates: North American employers verify prior employment, and a fabricated entry is grounds for a rescinded offer. The technical substance here does not need a false label — a public repo with real commits often survives scrutiny better than an unverifiable internship line.

## 13. Interview Leverage

- **ML system design**: "Design visual search / a fashion recommender" — every stage built first-hand.
- **Retrieval depth (rare and differentiating)**: ANN index families, product quantization mechanics, the accuracy–latency–memory Pareto frontier — traceable back to hand-implementing binary encoding from a patent at CAS.
- **Ranking**: negative sampling and exposure bias, multimodal feature fusion, ablation methodology.
- **LLM engineering (AIE crossover)**: structured-output reliability, grounding enforcement, critic loops, cost control.
- **SDE**: microservice decomposition, cache strategy, P99 optimization, graceful degradation, A/B infrastructure.
