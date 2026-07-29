# TuneFlow — Design Document

**Sequential Music Recommendation Platform with LLM Playlist Generation**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer Intern–grade project: Spotify-style streaming recommendation backend with sequential models, audio/text multimodal features, and natural-language playlist generation |

---

## 1. Overview

TuneFlow is a music streaming recommendation backend covering the three core surfaces of a real streaming product:

1. **Personalized home feed** — "made for you" track and playlist recommendations.
2. **Sequential radio / autoplay** — next-track prediction that respects listening context and session flow.
3. **Natural-language playlist generation** — "lo-fi for coding on a rainy day" → LLM intent parsing → retrieval + ranking over the catalog → a coherent, ordered playlist with an LLM-written description.

The project pairs classical recommendation modeling (collaborative filtering → sequential transformers) with multimodal item understanding (audio features + text metadata) and the same production-style serving and MLOps bar as an industry team.

This proposal deliberately connects to the student's existing Spotify music-genre classification project (50k tracks, PCA + AdaBoost), extending an analysis-level project into a full production-style system — a natural growth story for the resume.

## 2. Goals & Non-Goals

### Goals
- Full pipeline on playlist/listening data: candidate retrieval → ranking → sequence-aware ordering.
- Sequential modeling as the core ML depth: SASRec / BERT4Rec for next-track prediction; session-aware recommendations.
- Playlist continuation as a headline task (aligned with the Spotify RecSys Challenge 2018 formulation, enabling public benchmark comparison).
- LLM layer: natural-language playlist requests parsed to structured constraints (mood, tempo, era, genre, activity); grounded playlist assembly with generated titles/descriptions.
- Production-style serving: microservices, Redis session state, vector search, latency budgets, monitoring, retraining pipeline.

### Non-Goals
- Audio content ingestion/streaming (metadata + precomputed audio features only; no actual audio serving).
- Licensing/rights management.
- Training audio encoders from scratch (precomputed features or off-the-shelf embeddings such as MERT/CLAP are used).

## 3. Dataset

| Dataset | Role | Notes |
|---|---|---|
| **Spotify Million Playlist Dataset (MPD)** | Primary | 1M user-curated playlists, ~2.3M unique tracks with title/artist/album metadata. The canonical **playlist continuation** benchmark (RecSys Challenge 2018) with published results to compare against. |
| **Music4All / LFM-1b (Last.fm)** | Complement | Timestamped listening histories for sequential/next-track modeling; Music4All adds genre tags, lyrics snippets, and audio descriptors. |
| **Spotify audio features** (danceability, energy, valence, tempo, etc.) | Item features | The student already worked with these fields in the genre-classification project; reused here as ranking features and LLM constraint targets. |

**Labels**: playlists and listening sequences provide positive co-occurrence / next-item labels; negatives via random + popularity-corrected sampling (documented bias discussion, as in industry practice without impression logs).

**Split**: playlist continuation — hold out the last N tracks per playlist; sequential — leave-one-out per user plus a temporal holdout.

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+, SQL |
| ML frameworks | PyTorch, scikit-learn |
| Models | ALS/BPR matrix factorization (baseline), two-tower retrieval, SASRec / BERT4Rec (core), LightGBM or DeepFM ranker |
| Audio/text embeddings | Spotify audio-feature vectors; BGE / Sentence-BERT (titles, artists, tags); optional CLAP/MERT audio embeddings for a deeper multimodal story |
| LLM | GPT-4o-mini / Claude Haiku via API (intent parsing, playlist naming/description); Qwen2.5-7B via Ollama as local fallback |
| Vector search | FAISS |
| Serving | FastAPI + Pydantic; optional gRPC internally |
| Cache / session store | Redis (listening-session state, hot-user caches) |
| Streaming | Kafka — listening-event replay for simulated live traffic |
| Storage | Parquet + DuckDB (offline), PostgreSQL (catalog) |
| MLOps | MLflow, Airflow, Docker Compose, GitHub Actions CI |
| Monitoring | Prometheus + Grafana |
| Load testing | Locust |

## 5. System Architecture

```
 Listener ──► API Gateway (FastAPI)
                │
                ├── /home ───────────► Orchestrator ─► Retrieval Service
                │                                        ├ two-tower ANN (FAISS)
                │                                        ├ item-item CF ("fans also like")
                │                                        └ content channel (audio+text embeddings;
                │                                          cold-start path for new tracks)
                │                                      ─► Ranking Service (LightGBM/DeepFM)
                │
                ├── /radio/next ─────► Sequence Service (SASRec/BERT4Rec)
                │                       reads session state from Redis;
                │                       session-aware next-track prediction
                │
                └── /playlist/generate ─► LLM Playlist Service
                                           1. parse request → constraints
                                              (mood, tempo range, era, genre, activity)
                                           2. constrained retrieval over catalog
                                           3. sequence ordering (energy arc / transitions)
                                           4. grounded title + description generation
                                              (only real retrieved tracks appear)

 Events: Log Replayer ─► Kafka ─► session/feature updater ─► Redis
 Offline: Parquet/DuckDB ─► training ─► MLflow registry ─► rollout
 Ops: Prometheus/Grafana, Airflow retraining DAG, Locust
```

## 6. Component Design

### 6.1 Data & Feature Layer
- ETL: MPD JSON → normalized catalog (PostgreSQL) + playlist/track interaction tables (Parquet).
- Item features: audio-feature vector, genre tags, artist popularity, text embeddings of title/artist.
- User/session features: recent listening sequence, genre/tempo affinity profile, time-of-day context.
- Redis holds live session state (current radio session's played tracks, skips) updated from the Kafka stream.

### 6.2 Retrieval
- **Baselines first**: item-item collaborative filtering and ALS/BPR — cheap, strong, and pedagogically important before transformers.
- **Two-tower**: playlist/user tower × track tower, in-batch negatives, FAISS serving.
- **Content channel**: audio + text embeddings for cold-start tracks and constraint-filtered retrieval (feeds the LLM playlist surface).

### 6.3 Sequential Modeling (Core ML Depth)
- SASRec as the main model; BERT4Rec as a comparison (masked vs. autoregressive training — a strong interview discussion).
- Session-aware inputs: skip signals treated as weak negatives; recency-weighted history.
- Evaluated on next-track HitRate@10 / NDCG@10 against GRU4Rec and Markov-chain baselines.

### 6.4 Playlist Continuation (Benchmark Task)
- MPD challenge formulation: given playlist title + first K tracks, predict the held-out remainder.
- Metrics: R-Precision, NDCG, and clicks (RecSys 2018 official metrics) — enables direct comparison with published leaderboard results.

### 6.5 LLM Playlist Generation
- **Intent parsing**: LLM → JSON constraints validated by Pydantic (mood, tempo BPM range, era, genres, activity, target length); retry-on-invalid.
- **Constrained retrieval**: constraints map to filters over audio features + embedding similarity; candidates come only from the catalog (grounding by construction).
- **Ordering**: sequence model + simple "energy arc" heuristics (warm-up → peak → cool-down) for playlist coherence — a fun, demoable talking point.
- **Generation**: LLM writes playlist title/description referencing actual selected tracks; token cost logged.

### 6.6 Serving, System Design & MLOps
- Services: gateway / orchestrator / retrieval / ranking / sequence / LLM-playlist under Docker Compose.
- Redis session store enables stateful /radio/next with low latency; P99 target < 120 ms for non-LLM surfaces.
- Kafka log replay of timestamped listening events simulates live traffic for dashboards and A/B bucket comparison (honest framing: simulated serving loop, not real feedback).
- MLflow registry + Airflow weekly retrain with offline eval gates; Prometheus/Grafana for QPS, latency percentiles, cache hit rate, model version.
- Locust load-test report for /home and /radio/next.

## 7. Evaluation Plan

- **Retrieval**: Recall@K per channel; cold-start recall on held-out new tracks.
- **Sequential**: HitRate@10 / NDCG@10 vs. GRU4Rec and Markov baselines; skip-aware ablation.
- **Playlist continuation**: R-Precision / NDCG / clicks vs. RecSys 2018 published results.
- **LLM playlist quality**: constraint-satisfaction rate (measurable: do selected tracks fall inside the requested tempo/genre bounds?); human-eval rubric on coherence/fit for ~50 generated playlists.
- **System**: Locust QPS vs. P50/P99 latency; cache ablation.

## 8. Milestones (mapped to 19 mentoring hours)

| # | Phase | Hours | Output |
|---|---|---|---|
| 0 | Kickoff & scoping | 1 | Repo, environment, MPD EDA assignment |
| 1 | RecSys fundamentals + CF baselines | 3 | Metrics, item-item CF, ALS/BPR baselines |
| 2 | Retrieval | 3 | Two-tower + FAISS, content/cold-start channel |
| 3 | Sequential modeling | 4 | SASRec (+BERT4Rec comparison), playlist continuation benchmark |
| 4 | LLM playlist generation | 3 | Intent parsing, constrained retrieval, ordering, generation |
| 5 | System integration + MLOps | 3 | Microservices, Redis sessions, Kafka replay, MLflow/Airflow, dashboards |
| 6 | Resume & interview prep | 2 | Resume bullets, ML system design mock, deep-dive drill |

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MPD scale (2.3M tracks) on a laptop | Subsample to a few hundred-k playlists for training; DuckDB out-of-core ETL; full catalog only in the FAISS index |
| Audio embeddings (CLAP/MERT) too heavy | Optional; Spotify audio-feature vectors alone are sufficient for the multimodal story |
| Music domain seen as less "industrial" than feed/e-commerce | Framed via the public RecSys Challenge benchmark + identical system architecture to industrial stacks; transferable in interviews |
| No impression logs → biased labels | Documented negative-sampling strategy; interview talking point |
| LLM cost | Cache parsed intents; local Qwen fallback |

## 10. Interview Leverage

- **ML system design**: "Design Spotify's recommendations / autoplay" — direct experience, including session-state handling that most candidates cannot discuss concretely.
- **ML deep-dive**: sequential architectures (SASRec vs. BERT4Rec vs. GRU4Rec), skip-signal handling, negative sampling & exposure bias, cold-start via content embeddings.
- **SDE/system design**: stateful low-latency services, Redis session management, streaming event pipelines, cache strategy.
- **Story line**: extends the student's existing Spotify genre-classification project into a production system — a coherent "analysis → engineering" growth narrative across the resume.
