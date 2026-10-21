# FeedForge — Design Document

**LLM-Augmented News Feed Recommendation — One End-to-End Loop, from Impression Log to Served Feed and Back**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Positioning | ML Engineer Intern–grade project: a complete feed recommendation backend — retrieval, ranking, LLM augmentation, streaming features, experimentation, model lifecycle — built and measured as one system |
| Dataset | MIND (Microsoft News Dataset): real impression logs with click / non-click labels |
| Core stack | PyTorch · FAISS · ONNX Runtime · FastAPI · Redis · Redpanda (Kafka API) · Parquet + DuckDB · MLflow · Prometheus + Grafana · Docker Compose |
| Hardware | One laptop, CPU only, 16 GB RAM. No stage requires a GPU; a GPU only shortens training. |

---

## 1. The Product Thesis

A feed recommender is not a model. It is a loop: a request is served, the user reacts, the reaction becomes a feature within seconds and a training example within a day, and the next model has to beat the current one before it is allowed to serve. Most portfolio projects build one box of that loop — usually the ranking model — and stop at a notebook.

**FeedForge builds the whole loop on real impression logs and treats every hand-off between components as an explicit, tested contract.** News is chosen deliberately because it is the hardest common case for that loop: articles live for a day or two, so most candidates have little or no interaction history, and a system that leans on item IDs fails quietly.

Five claims the project sets out to prove — or disprove — with measurements:

1. **On news, content-based item towers beat ID-based ones.** On a time-respecting split, a large share of dev-window articles never appear in training (EDA measures the exact share in week 1). A two-tower model whose item tower reads content can retrieve an article the second it is published.
2. **Real negatives beat sampled ones.** MIND logs what was shown and not clicked. Training the retriever with impression negatives on top of logQ-corrected in-batch negatives is an ablation most public datasets cannot support.
3. **The LLM earns its place only where it is measured to.** Three LLM roles — offline enrichment, cold-user interest profiles, listwise re-ranking — are ablated separately under a hard cost cap, and the re-ranker stays off the synchronous path unless it fits the latency budget. A negative result is reported as one.
4. **Training/serving skew is tested, not hoped against.** A parity test recomputes served features through the offline pipeline and fails CI when they disagree.
5. **The system degrades instead of failing.** Every stage has a deadline and a defined fallback; a chaos test kills dependencies under load and asserts the feed still returns.

## 2. Goals & Non-Goals

### Goals
- **Walking skeleton first.** The full loop — request → served-impression log → stream → features → retrain → promote — runs end to end with trivial models by the end of week 2. Every later sprint upgrades one component behind an unchanged contract.
- **Multi-stage serving**: five-channel retrieval → CTR ranking (LR → DeepFM → DIN) → policy layer (near-duplicate removal, diversity, exploration) → optional LLM re-rank.
- **Feature freshness**: a click is visible to the user's next request within seconds, measured as P99 feature lag.
- **Point-in-time-correct offline features** from the same feature definitions the online path uses, with a parity test between the two.
- **Three LLM roles**, each ablated, each metered against a $20 cost cap.
- **Model lifecycle**: registry, multi-segment evaluation gate, shadow scoring, canary bucket, one-command rollback.
- **Experimentation**: deterministic layered bucketing, A/A validation, sample-ratio-mismatch checks, user-clustered confidence intervals.
- **Measured results**: offline metrics on three protocols (§8), P50/P99 latency and QPS per configuration, feature lag, degradation and fallback rates, LLM cost per 1k sessions.

### Non-Goals
- Training foundation models. LLMs are used through an API or as small open-weight models.
- Real user traffic. Traffic is simulated by timestamped log replay (§6.2), with an explicit account of what replay can and cannot measure.
- Position-bias correction. MIND does not document display position, so the bias is stated rather than modeled from a guessed field.
- Kubernetes. Docker Compose is the ceiling; K8s manifests are a stretch goal.
- Images and video in the core. MicroLens is an optional multimodal track.

## 3. Dataset — and the Facts That Shape the Design

### 3.1 Sources

| Dataset | Scale | Role |
|---|---|---|
| **MIND-small** | ~50k users, ~65k articles, ~230k impressions | Default development set; every pipeline runs on a laptop in minutes |
| **MIND-large** | ~1M users, ~161k articles, ~15M impressions | Scale test for replay throughput, feature-store memory and the index study; sampled for training |
| Criteo / Avazu | tabular CTR | Optional DeepFM warm-up before MIND |
| MicroLens | short video with cover images | Optional multimodal track (CLIP image tower) |

MIND ships `behaviors.tsv` (impression id, user id, timestamp, click history, and the impression's shown articles each labeled 1/0), `news.tsv` (category, subcategory, title, abstract, URL, Wikidata entities), and 100-d TransE entity embeddings. The data is downloaded by a script and **never committed**; CI runs on a synthetic generator that emits MIND-shaped files (§9).

### 3.2 Five facts that drive the design

| Fact | Design consequence |
|---|---|
| **Articles are short-lived.** Most articles shown in a later window never appear in an earlier one. | Item towers are content-based; an item-ID embedding is an optional, hashed, dropout-regularized extra. Cold-item metrics are reported next to every headline number. |
| **`news.tsv` has no publish time.** | `first_seen_ts` = the earliest impression that shows the article. The candidate corpus at time *t* is the set of articles first seen inside a freshness window before *t* (default 48 h, tuned so the window covers ≥ 95% of clicked articles on validation). |
| **`history` is a pre-window snapshot** (confirm in EDA). | Clicks inside the window must be reconstructed from impression labels. The stream feature updater does exactly that during replay, and the offline builder does the same thing point-in-time. |
| **Labels are conditional on exposure.** Negatives are shown-and-not-clicked; unshown articles have no label. | Impression-level ranking (Protocol A) is clean. Full-corpus retrieval (Protocol B) treats unshown articles as unknown, so its recall is a lower bound and is labeled as such. |
| **Test-split labels are hidden** behind the official leaderboard. | All development and ablations use the dev split as the test set. A leaderboard submission is a stretch, and only if the leaderboard still accepts submissions — confirm before planning around it. |

### 3.3 Splits

Strictly by time, never by random row:

| Split | MIND-small source | Use |
|---|---|---|
| Train | `train/behaviors.tsv` minus its last day | Model training |
| Validation | last day of `train/behaviors.tsv` | Early stopping, calibration, window and threshold tuning |
| Test / replay | `dev/behaviors.tsv` | Offline Protocols A and B, and the "online day" the replayer serves (Protocol C) |

For the staleness study (§7.9), MIND-large's multi-day training window is carved into rolling train → replay → retrain → replay segments.

## 4. Stack

### 4.1 Choices

| Layer | Core | Compared against / stretch | Why |
|---|---|---|---|
| Env & tasks | Python 3.11+, `uv`, entry points in `pyproject.toml` | — | One lockfile; `uv run ff-…` behaves the same on every laptop |
| Models | PyTorch: two-tower, LR, DeepFM, DIN | NRMS replica as the published-baseline reference | The classic retrieval → ranking ladder, each rung a measured row |
| Text encoder | Frozen small sentence encoder (bge-small-en-v1.5 class, 384-d), precomputed | Qwen3-Embedding-0.6B; fine-tuned title encoder | Small enough to embed the full corpus on CPU in minutes |
| Inference | ONNX Runtime (CPU) | PyTorch eager as the latency baseline | Batched CPU scoring; export parity is unit-tested |
| Vector search | FAISS, exact search over the freshness window | IVF / HNSW / IVF-PQ over the full corpus (§7.4 study) | At window size exact search costs about a millisecond; the study shows where ANN starts to pay |
| LLM | Provider-agnostic LLM gateway: a hosted small-tier model, or Qwen3-4B-Instruct via Ollama | logprob scoring on a llama.cpp server | Cost cap, cache and schema validation in one place |
| Serving | FastAPI + Pydantic contracts, pooled `httpx` clients, `orjson` | gRPC on the orchestrator → ranking hop | Contracts live in one package shared by every service |
| Online store & cache | Redis | — | Features, sorted-set channels, cursors, re-rank cache |
| Event bus | Redpanda (Kafka API, single binary) behind an `EventBus` interface | Redis Streams adapter (fallback); Apache Kafka | Kafka semantics in ~1 GB of RAM |
| Offline storage | Parquet + DuckDB (`ASOF JOIN` for point-in-time features) | — | Out-of-core on MIND-large; no warehouse needed |
| MLOps | MLflow (tracking + registry, SQLite backend); pipelines as plain entry points; Airflow DAG as a thin wrapper | Prefect | Pipelines run without Airflow on a laptop and in CI |
| Observability | Prometheus + Grafana; JSON logs carrying `request_id` | OpenTelemetry + Jaeger | Per-stage latency and the funnel are visible from day one |
| Load & chaos | Locust; compose-level fault injection | — | Latency, QPS and degradation are deliverables |
| CI | GitHub Actions: lint, unit, contract, leakage, parity, compose e2e on fixtures | — | Correctness on every PR; performance tracked in a versioned benchmark table |

### 4.2 Laptop footprint (MIND-small, core profile)

| Process | RAM |
|---|---|
| Redis (all online state) | ~0.2 GB |
| Redpanda (`--smp 1 --memory 1G`) | ~1 GB |
| Prometheus + Grafana | ~0.5 GB |
| MLflow | ~0.2 GB |
| Gateway, orchestrator, retrieval, ranking, LLM gateway, stream workers | ~1.5–2 GB (vector tables, ONNX models) |
| **Core total** | **≈ 3.5–4 GB** |
| Airflow (compose profile `ops`, optional) | +1.5–2 GB |

MIND-large replay runs on a 200k-user sample so Redis stays around 1 GB.

## 5. Architecture

### 5.1 Three time scales

| Path | Latency | What runs there | Writes |
|---|---|---|---|
| **Online** | per request, P99 ≤ 200 ms | gateway → orchestrator → retrieval → ranking → policy | `feed.served` log, pagination cursor |
| **Nearline** | seconds to minutes | feature updater, item pipeline, index appliers, LLM worker, metrics joiner, log sink | Redis features, window indexes, cached re-ranks, Parquet |
| **Offline** | hourly to weekly | dataset build, training, evaluation, index generation, gate, promote | Parquet datasets, MLflow runs, release bundles |

Design rule: **anything that can be precomputed moves one tier slower.** The online path reads and scores. It never waits on an LLM, never embeds an article, and never blocks on a write other than the response itself.

### 5.2 System diagram

```
──────────────── ONLINE · per request · P99 ≤ 200 ms ─────────────────────────────

 Client / Replayer
    │  GET /v1/feed · POST /v1/events
    ▼
 Gateway ── auth · rate limit · request_id · deadline = now + 200 ms
    ▼
 Orchestrator
    ├─ 1  one pipelined Redis round trip: user features · seen set · cached re-rank
    ├─ 2  bucket assignment (layered hash)
    ├─ 3  Retrieval svc ─ user tower, per request ─┬─ two-tower  FAISS window
    │                                              ├─ semantic   FAISS window
    │                                              ├─ category   Redis zset
    │                                              ├─ trending   Redis zset
    │                                              └─ fresh      Redis zset
    │        merge · dedup · filter (clicked, seen, out of window) → ~500
    ├─ 4  Ranking svc ─ item features · DIN / DeepFM (ONNX, one batch) · calibrate → top-50
    ├─ 5  policy: near-dup removal · MMR diversity · exploration slot (+ propensity)
    ├─ 6  apply cached LLM re-rank if still valid, else keep ranker order
    └─ 7  respond top-10 + cursor ─ publish feed.served (fire-and-forget)

──────────────── NEARLINE · seconds ──────────────────────────────────────────────

 Redpanda: user.events · item.events · item.vectors · feed.served · llm.jobs
   feature-updater    user.events  → recent clicks, category affinity, item CTR,
                                     trending / fresh zsets                 → Redis
   item-pipeline      item.events  → text + two-tower vectors              → item.vectors
                                     + one enrichment job                  → llm.jobs
   retrieval replica  item.vectors → its own window indexes (own consumer group)
   llm-worker         llm.jobs     → topics · interest profiles · async re-ranks → Redis
   metrics-joiner     feed.served ⨝ logged labels → matched-click CTR per arm → Prometheus
   log-sink           every topic  → Parquet, hourly partitions

──────────────── OFFLINE · hourly → weekly ───────────────────────────────────────

 build_dataset (point-in-time, DuckDB ASOF) → train two-tower → build index generation
   → train ranker → calibrate → evaluate (Protocols A/B/C × segments) → gate
   → MLflow registry → shadow → canary bucket → promote · or roll back

 Ops: Prometheus · Grafana · alert rules · Locust · chaos test · debug trace store
```

### 5.3 Component map

The table is the system's contract at a glance: what each component owns, what it reads and writes, how much of the request budget it gets, and what happens when it fails.

| ID | Component | Runs as | Owns | Reads | Writes | Online budget (P99) | Degrades to |
|---|---|---|---|---|---|---|---|
| C1 | Gateway | service | auth, rate limit, `request_id`, deadline | — | — | 2 ms | static popular feed |
| C2 | Orchestrator + policy | service | request lifecycle, buckets, policy layer, cursors | Redis, experiment config | `feed.served`, cursors | whole request ≤ 200 ms | Redis-only popular feed |
| C3 | Retrieval | service (holds indexes) | channels, merge, filters, index generations | window indexes, Redis zsets, `item.vectors` | — | 25 ms | channels that answered |
| C4 | Ranking | service | item-feature assembly, CTR model, calibration | resident vectors, Redis item hashes | — | 60 ms | fused retrieval order |
| C5 | LLM gateway + worker | service + worker | providers, prompts, cache, schema validation, cost cap | `llm.jobs` | Redis (topics, profiles, re-ranks) | off the sync path | ranker order |
| C6 | Feature platform | library + worker | feature registry, online updater, point-in-time builder, parity test | `user.events`, `item.events` | Redis, Parquet | 5 ms read | per-feature defaults |
| C7 | Item pipeline | worker | text + two-tower vectors, enrichment trigger | `item.events` | `item.vectors`, `llm.jobs` | servable ≤ 10 s after publish | fresh channel only |
| C8 | Event backbone + replayer | Redpanda + CLI | topics, schemas, dead-letter topics, replay clock | MIND Parquet | `user.events`, `item.events`, feed requests | — | — |
| C9 | Experimentation | library + config + job | layered bucketing, assignment logging, analysis | `experiments.yaml`, Parquet | reports | < 1 ms | control arm |
| C10 | Training & lifecycle | pipelines | datasets, training, evaluation, gates, registry, rollout | Parquet | MLflow, release bundles | — | keep current bundle |
| C11 | Observability | Prometheus, Grafana, trace store | metrics, alerts, dashboards, request traces | `/metrics`, logs | dashboards, alerts | — | — |

## 6. End-to-End Flows

Components are only half of a design; the other half is what crosses between them. Four flows cover everything that moves through the system.

### 6.1 Request lifecycle and latency budget

Budget measured on the development laptop, CPU only, under replay load:

| Step | Component | Work | P99 budget |
|---|---|---|---|
| 1 | Gateway | validate, authenticate, assign `request_id`, set the deadline | 2 ms |
| 2 | Orchestrator | one pipelined Redis round trip (user features, seen set, cached re-rank); bucket hash | 5 ms |
| 3 | Retrieval | user tower over ≤ 50 resident item vectors; five channels in parallel; merge, dedup, filter | 25 ms |
| 4 | Ranking | one Redis pipeline for ~500 item hashes, vectorized feature assembly, one batched ONNX pass, calibration | 60 ms |
| 5 | Orchestrator | policy layer; apply the cached LLM order if valid | 5 ms |
| 6 | Orchestrator | write the cursor snapshot, respond; publish `feed.served` off the critical path | 3 ms |
| — | Internal hops | three HTTP calls over pooled connections, `orjson` payloads (~20 KB to ranking) | ~20 ms |
| | **Target / SLO** | | **~120 ms target · 200 ms SLO** |

**Deadline propagation.** Every internal call carries `X-Deadline-Ms`, the remaining budget. A callee that cannot finish in time returns what it has — retrieval returns the channels that completed — and a caller that runs out of budget falls back instead of waiting. A **circuit breaker** per dependency stops the orchestrator from spending budget on a dependency that keeps failing.

**Degradation ladder** — every rung is exercised by the chaos test (§9):

| Failure | Detected by | Response | Counted as |
|---|---|---|---|
| LLM re-rank missing or stale | cache miss / overlap check | ranker order | `llm_rerank_total{result="miss"}` |
| One retrieval channel slow | 15 ms channel deadline | merge the channels that answered | `degraded_total{reason="channel"}` |
| Retrieval service down | breaker open | trending + fresh read directly from Redis | `degraded_total{reason="retrieval"}` |
| Ranking slow or down | deadline / breaker | order by fused retrieval score | `degraded_total{reason="ranking"}` |
| Redis down | connection error | cold-user defaults + in-process popular snapshot (refreshed every 60 s) | `degraded_total{reason="features"}` |
| Everything | — | static popular feed held in gateway memory — **a feed request never returns 5xx** | `degraded_total{reason="static"}` |

Responses carry `degraded: [...]`, so a degraded feed is visible rather than silent.

### 6.2 Event lifecycle — log replay without grading our own homework

The replayer reads the dev window in timestamp order on an accelerated clock (×K) and does three things per logged impression:

```
 MIND dev log, time-ordered, clock × K
   │
   ├─► item.events   article first seen               → item pipeline (C7), fresh zset
   │
   ├─► GET /v1/feed  one request per logged impression → OUR served list
   │                    └─► feed.served { request_id, impression_id, items, pctr,
   │                                      propensity, buckets, bundle, degraded }
   │
   └─► user.events   the ORIGINAL impression's clicks and non-clicks
                        → feature updater: the world evolves as it really did

 metrics-joiner:  feed.served ⨝ logged labels on impression_id
   labeled slots    served items that were in the original impression → CTR measured here
   matched clicks   labeled slots that were clicked
   unlabeled slots  everything else → outcome unknown; reported, never counted as negatives
```

Why the split matters: if user state evolved from clicks simulated on our own served list, the replay would invent the outcomes it is meant to measure. Driving state from logged behavior keeps the world honest; scoring our policy only on labeled slots keeps the metric honest; and the **unlabeled share** says how far our policy has moved from the logging policy — the limit of what any replay can tell you, printed next to every replay number.

**Delivery semantics**
- At-least-once delivery. Every event carries an `event_id`, and counters dedupe with a short-TTL `SET NX` on it, so a redelivered click is never counted twice.
- Topic key `user_id` gives per-user ordering for user state.
- **All decay and TTL logic runs on event time from the replay clock, never on `time.time()`** — otherwise a ×60 replay makes every article look sixty times staler than it is. Requests carry `X-Event-Time` in replay mode, and a unit test pins the clock.
- Events later than a 5-minute watermark go to Parquet only, never to online counters.
- Malformed events go to a per-topic dead-letter topic; a non-empty DLQ fails the e2e test.

**From served impression to training example.** Training data comes from MIND's training window, with point-in-time features rebuilt offline. The served-feature log is used for two things: the parity test (§7.2), and a demonstration of **log-and-wait** — joining the features actually used at serve time with labels that arrive later inside an attribution window, which is how production systems build training data without recomputing features at all.

### 6.3 New-item lifecycle

| Time after publish | What happens | Article reachable via |
|---|---|---|
| t₀ | `item.events` received; `first_seen_ts` stored; added to the `fresh` zset | fresh |
| ≤ 1 s | micro-batch (≤ 64 items or 500 ms) embedded — text vector + two-tower item vector — and published on `item.vectors`; every retrieval replica appends it to its window indexes | **+ two-tower, semantic** |
| seconds–minutes | LLM enrichment lands (taxonomy topics, normalized entities); `llm_ready = 1` | (better ranking) |
| first impressions | hourly impression / click buckets accumulate; CTR smoothed with a category prior | + trending, past a minimum-impression threshold |
| window end (48 h) | dropped at the next index generation; trimmed from zsets | — |

Two measured numbers: **time-to-servable** (P95 ≤ 10 s) and **time-to-enriched**.

**The missing-feature trap.** Offline, every article has LLM topics because the backfill ran in batch. Online, a fresh article has none for its first minutes. A ranker trained only on fully enriched rows has never seen "no topics" — classic training/serving skew. FeedForge trains the ranker with LLM features **masked at the rate observed in serving**, plus an explicit `llm_ready` indicator, so a brand-new article is scored as "not enriched yet" rather than as an anomaly.

### 6.4 Model lifecycle

```
 train ─► evaluate (A/B/C × segments) ─► gate ─► registry: Staging
       ─► shadow: score live replay traffic, log only — compare score distributions (PSI), latency
       ─► canary: 5% bucket, guardrails watched
       ─► Production: atomic alias swap        rollback = repoint the alias, one command
```

**The release unit is a bundle, not a model**: `{two-tower vN, index generation built by vN, ranker vM trained on vN's retrieval features, calibrator, feature-definition version}`. Two version-skew bugs become impossible by construction:
- **Embedding skew** — querying an index built by one item tower with a different user tower. Each index generation records the model that built it; retrieval loads (tower, index) as a pair and swaps them atomically.
- **Feature-input skew** — the ranker consumes retrieval scores as features, so swapping the two-tower model silently shifts the ranker's input distribution. They ship together.

## 7. Component Design

Each component is specified the same way: **what it owns · its interface · design · what it degrades to · what it emits · done when.** "Done when" is the acceptance test for the sprint that builds it.

### 7.1 C8 — Event Backbone & Log Replayer

**Topics**

| Topic | Key | Producer | Consumers | Payload (schema `…/v1`) |
|---|---|---|---|---|
| `user.events` | `user_id` | replayer, `POST /v1/events` | feature-updater, log-sink | `event_id, type (impression\|click), user_id, item_id, impression_id, event_ts` |
| `item.events` | `item_id` | replayer, `POST /v1/admin/items` | item-pipeline, feature-updater, log-sink | `item_id, title, abstract, category, subcategory, entities, first_seen_ts` |
| `item.vectors` | `item_id` | item-pipeline | every retrieval replica (own consumer group), ranking, log-sink | `item_id, text_vec, tower_vec, model_version` |
| `feed.served` | `user_id` | orchestrator | metrics-joiner, log-sink | see §7.6 |
| `llm.jobs` | job key | item-pipeline, orchestrator | llm-worker | `kind (enrich\|profile\|rerank), input_hash, payload, deadline` |

- Event schemas are Pydantic models in `contracts/`, exported to JSON Schema. Producers validate before publishing; consumers send an unknown major version to `<topic>.dlq`.
- The bus sits behind an `EventBus` interface; Redis Streams is the fallback adapter if Redpanda proves too heavy for the laptop.

**Replayer**
- Clock: `t_log = t_start + (wall − wall_start) × K`. Modes: `--speed K` (time-faithful), `--qps N` (load-test mode, ignores timestamps), `--users-sample p`.
- **Deterministic**: the same seed and data produce the identical request sequence, so two configurations can be compared on exactly the same traffic (paired replay, §7.8).
- **State bootstrap**: before replay starts, the point-in-time builder materializes every user's state as of `t_start` into Redis — the same code path that builds training features.

**Done when**: a full MIND-small dev-window replay at ×60 completes with consumer lag P99 < 5 s and every DLQ empty.

### 7.2 C6 — Feature Platform

**One definition, two paths.** Each feature is declared once; the online updater and the offline builder call the same transformation on the same event shape:

```python
Feature(
    name="user.category_affinity",
    entity="user",
    dtype="float32[n_categories]",
    source="user.events",
    transform=decayed_category_counts,  # pure function of (state, event), shared online/offline
    half_life="24h",                    # event time, never wall clock
    default="zeros",
    freshness_sla="5s",
    version=2,
)
```

**Feature groups**

| Group | Examples | Computed |
|---|---|---|
| User | last-50 clicked items, decayed category affinity, click count, active days, LLM interest profile + embedding | stream (seconds); profile nearline |
| Item | category, subcategory, entity ids, text vector, age in hours, smoothed decayed CTR, impression count, LLM topics, `llm_ready` | stream + item pipeline |
| Context | hour of day, weekday, page number | per request |
| Cross | user–item category affinity, max / mean cosine between history and candidate | ranking, per request |
| Provenance | channel bitmask, per-channel rank and score | retrieval, per request |

**Online store layout (Redis)**

| Key | Type | Content | Bound |
|---|---|---|---|
| `u:{uid}:clicks` | list | last 50 clicked item ids (`LPUSH` + `LTRIM`) | 30-day idle TTL |
| `u:{uid}:cat` | hash | decayed category counts + last-update event time (decay applied lazily on read) | 30-day idle TTL |
| `u:{uid}:seen` | set | items served in the last 24 h | 24 h TTL |
| `u:{uid}:profile` | hash | LLM profile text, its embedding, click count at generation | regenerated every 10 new clicks |
| `i:{iid}` | hash | static attributes, LLM topics, `llm_ready` | window + 1 day |
| `i:{iid}:ctr` | hash | hourly impression / click buckets | 72 h |
| `z:trending`, `z:trending:{cat}` | zset | smoothed decayed CTR | top 2,000 |
| `z:fresh` | zset | item id scored by `first_seen_ts` | trimmed to the window |
| `rr:{uid}` | string | async LLM re-rank: candidate set, order, bundle | 10 min TTL |
| `cur:{request_id}` | string | ranked top-50 snapshot for pagination | 10 min TTL |

**Point-in-time builder.** A DuckDB `ASOF JOIN` attaches to each training impression the feature state strictly before its timestamp. A **leakage test** builds a hand-made timeline in which a click lands one second after an impression and asserts the click is absent from that impression's features.

**Parity test.** For a 1% sample of requests, ranking returns the exact feature matrix it scored and the orchestrator attaches it to `feed.served`. The offline builder recomputes the same (user, item, event time) rows, and the mismatch rate is tracked per feature. CI fails above 0.1% on fixtures; Grafana tracks it during replay.

- **Degrades to**: per-feature defaults when Redis is unreachable (the cold-user path).
- **Emits**: `feature_lag_seconds` (event time → visible in Redis), `feature_parity_mismatch_ratio{feature}`, updater throughput.
- **Done when**: feature lag P99 < 5 s at ×60 replay; parity mismatch < 0.1% for every feature.

### 7.3 C7 — Item Pipeline & LLM Enrichment

- **Vectors**: `title. abstract` (+ category) through the frozen sentence encoder → 384-d text vector; text vector + category + entity embeddings through the two-tower item tower (ONNX) → 64-d tower vector. The corpus backfill runs offline in batch; new articles arrive in micro-batches.
- **Publish, don't share memory.** Vectors go out on `item.vectors`. Each retrieval replica consumes with its own consumer group (broadcast) and applies them to its in-memory index, so embeddings are computed once and applied everywhere. A restarted replica recovers as **snapshot + log**: load the latest index generation from disk, then replay `item.vectors` from the offset recorded in that snapshot.
- **Enrichment**: one `llm.jobs` entry per article. The output is schema-validated JSON — `topics` drawn from a **fixed taxonomy** (~200 topics, derived once by clustering the backfill output, then frozen) plus normalized entities. Free-form tags would explode the ranker's vocabulary; a frozen taxonomy keeps them a multi-hot feature. Invalid JSON gets one retry, then is marked failed. Results are cached by content hash, and the backfill uses batch endpoints.
- **Degrades to**: with the pipeline down, new articles still reach users through the fresh channel; ranking sees `llm_ready = 0` and default content features.
- **Emits**: `time_to_servable_seconds`, `time_to_enriched_seconds`, `enrichment_schema_valid_ratio`.
- **Done when**: time-to-servable P95 ≤ 10 s; schema-valid rate ≥ 98%; backfill spend inside its share of the cap.

### 7.4 C3 — Retrieval Service

**Channels**

| Channel | Signal | Store | Default quota | Why it exists |
|---|---|---|---|---|
| two-tower | learned user × item | FAISS window index (64-d) | 200 | main personalized channel |
| semantic | user profile embedding (LLM profile, or mean of history text vectors) × item text vector | FAISS window index (384-d) | 100 | light and cold users; diversity relative to the learned channel |
| category | the user's top-3 categories → trending in each | Redis zsets | 100 | cheap, robust, explainable |
| trending | smoothed decayed CTR | Redis zset | 50 | cold users; safety net |
| fresh | newest articles in the window | Redis zset | 50 | item cold start; exploration pool |

Merge = union, dedup by item id, keep each channel's rank and score as features plus a provenance bitmask, then filter out clicked, seen-in-24h and out-of-window articles → ~500 candidates. Quotas are experiment parameters.

**Two-tower model**
- **User tower** — attention pooling over the last 50 clicked articles' content vectors, concatenated with category affinity → MLP → 64-d. Computed **per request** from a resident content-vector table covering every article (≈ 125 MB in fp16 on MIND-large), so the user embedding is never staler than the latest click.
- **Item tower** — text vector + category / subcategory embeddings + mean entity embedding (MIND's TransE vectors) [+ optional hashed ID embedding with dropout] → MLP → 64-d, L2-normalized.
- **Loss** — sampled softmax over in-batch negatives with **logQ correction** for the popularity skew of in-batch sampling (Yi et al., RecSys 2019) and a tuned temperature; **impression negatives** (shown, not clicked) added as hard negatives.

| Ablation | Measured on |
|---|---|
| in-batch → + logQ → + impression negatives | Protocol B Recall@100, overall and cold-item |
| content-only item tower vs. + ID embedding vs. ID-only | cold-item Recall@100 — the evidence for claim 1 |
| attention vs. mean pooling in the user tower | Recall@100 for light vs. heavy users |

**Index design**
- At a 48-hour window the live corpus is a few thousand articles (EDA confirms), where **exact inner-product search costs about a millisecond** — so exact search is the default.
- **Copy-on-write generations**: the applier builds the next small index and swaps the reference, so a search never sees a half-written index. Expired articles drop out at the hourly generation rebuild, and a bundle promotion forces a rebuild with the new item tower.
- **ANN study** on the full MIND-large corpus: Flat vs. IVF vs. HNSW vs. IVF-PQ — Recall@100 against exact, QPS, memory, build time. For news the honest conclusion may be "not needed at this window size", and stating that with a measurement is a stronger answer than reaching for HNSW by reflex.

**Channel accounting**: *unique click recall* per channel — clicked articles that only that channel retrieved. A channel that contributes almost nothing unique is removed, and the latency saved is recorded.

- **Degrades to**: the channels that answered within their deadline; heuristic channels alone if the learned ones fail.
- **Emits**: per-channel latency, candidate count and unique hits; merged candidate count; index generation and bundle version.
- **Done when**: Protocol B Recall@100 beats the popularity + recency baseline overall and on cold items; P99 ≤ 25 ms under replay load.

### 7.5 C4 — Ranking Service

- **Model ladder**, each rung a row in the results table: logistic regression with hashed crosses → **DeepFM** → **DIN** (target attention over the user's history content vectors against each candidate). An **NRMS** replica is trained as the published-baseline reference.
- **Training data**: MIND training impressions — real shown-and-not-clicked negatives — with point-in-time features. If negatives are downsampled at rate *w*, predictions are corrected with p′ = p / (p + (1 − p) / w) before calibration.
- **Calibration**: isotonic regression fit on validation, reported as ECE with a reliability diagram. It matters because the policy layer and experiments combine pCTR with other terms; uncalibrated scores make those weights meaningless.
- **Serving**: ranking owns item-feature assembly. Vectors are resident in-process (kept current from `item.vectors`); per-item hashes (enrichment, counters) come from one Redis pipeline. Assembly is vectorized NumPy feeding **one** batched ONNX forward pass — no per-candidate Python loops. PyTorch vs. ONNX output parity is a unit test (max abs diff < 1e-5).
- **Metrics**: Protocol A AUC, MRR, nDCG@5, nDCG@10 (MIND's official set) plus GAUC and LogLoss, sliced by user activity (cold / light / heavy) and article age.
- **Degrades to**: fused retrieval-score order.
- **Emits**: batch size, assembly and inference latency, score histogram (feeds the shadow PSI), bundle version.
- **Done when**: DeepFM or DIN beats LR and popularity on Protocol A; the gap to published NRMS-family numbers is stated and explained (they fine-tune text encoders end to end, FeedForge serves a frozen encoder on CPU); batch-500 P99 ≤ 60 ms.

### 7.6 C1 + C2 — Gateway, Orchestrator & Policy Layer

**API surface**

| Method & path | Purpose | Notes |
|---|---|---|
| `GET /v1/feed` | ranked feed | `user_id`, `limit ≤ 20`, `cursor`; in replay mode `X-Event-Time` makes the whole serving path use replay time |
| `POST /v1/events` | client feedback | batched; `Idempotency-Key`; `202 Accepted` |
| `GET /v1/items/{id}` | article lookup | |
| `POST /v1/admin/items` | publish an article | used by the replayer |
| `GET /v1/debug/requests/{request_id}` | stage-by-stage trace | candidates per channel, scores, timings, degradations; 1% sampled or forced by header; 1 h TTL |
| `GET /healthz`, `/readyz`, `/metrics` | ops | `readyz` stays red until indexes and models are loaded |

Errors use `application/problem+json`, and every response carries `X-Request-Id`.

**Example exchange (synthetic values)**

```http
GET /v1/feed?user_id=U90001&limit=3 HTTP/1.1
X-Api-Key: ff_demo_xxx
```

```json
{
  "request_id": "req_01JDEMO7Q",
  "items": [
    {"item_id": "N90412", "position": 1, "pctr": 0.083, "category": "sports",  "channels": ["two_tower", "category"]},
    {"item_id": "N90377", "position": 2, "pctr": 0.071, "category": "finance", "channels": ["semantic"]},
    {"item_id": "N90598", "position": 3, "pctr": 0.024, "category": "travel",  "channels": ["fresh"], "exploration": true}
  ],
  "cursor": "req_01JDEMO7Q:3",
  "bundle": "b-20261103.2",
  "buckets": {"retrieval": "control", "ranking": "din", "policy": "mmr_0.7", "llm": "async"},
  "llm_rerank": "applied",
  "degraded": []
}
```

The matching `feed.served` event — the record every downstream consumer joins on (`channel_mask` bits: two-tower 1, semantic 2, category 4, trending 8, fresh 16):

```json
{
  "schema": "feed.served/v1",
  "event_id": "evt_01JDEMO7R",
  "request_id": "req_01JDEMO7Q",
  "impression_id": "41207",
  "user_id": "U90001",
  "event_ts": "2019-11-15T09:12:44Z",
  "items": [
    {"item_id": "N90412", "pos": 1, "pctr": 0.083, "channel_mask": 5,  "propensity": 1.0},
    {"item_id": "N90377", "pos": 2, "pctr": 0.071, "channel_mask": 2,  "propensity": 1.0},
    {"item_id": "N90598", "pos": 3, "pctr": 0.024, "channel_mask": 16, "propensity": 0.04}
  ],
  "buckets": {"retrieval": "control", "ranking": "din", "policy": "mmr_0.7", "llm": "async"},
  "bundle": "b-20261103.2",
  "degraded": [],
  "feature_snapshot": null
}
```

**Pagination**: the cursor points at the `cur:{request_id}` snapshot of the ranked top-50, so page 2 is consistent with page 1 and costs one Redis read rather than a recomputation. When the snapshot expires, a fresh request runs and the seen filter prevents repeats.

**Policy layer** — the last stage, deliberately simple and auditable:
- **Near-duplicate removal**: text-vector cosine above a tuned threshold within the slate → keep the higher-scored article. Several outlets covering the same story is the norm in news.
- **Diversity**: MMR with λ chosen from a measured curve (Protocol C labeled-slot CTR vs. intra-list diversity), plus a hard rule of at most two consecutive articles from one subcategory.
- **Exploration**: one slot filled by sampling fresh-channel candidates in proportion to score, with the **propensity logged**. MIND's own logging propensities are unknown, so off-policy evaluation on MIND is impossible; logging propensities for our own exploration slot is the habit that makes inverse-propensity evaluation possible on the next round of logs.
- **LLM order**: `rr:{uid}` is applied if its candidate set overlaps the current top-20 by ≥ 80% — the LLM's relative order is applied to the overlap, and the other articles keep their ranker positions. Otherwise it is ignored.

**Microservices, measured rather than assumed.** Every service is also importable in-process; `FF_MODE=monolith` runs the identical code with function calls instead of HTTP. Locust measures both, and the P99 difference is the price of the network hops — the number that answers "why not a monolith?" in an interview.

**Done when**: every degradation rung is observable in metrics; a static feed is served with every dependency killed; the monolith vs. services comparison is in the benchmark table.

### 7.7 C5 — LLM Gateway and the Three LLM Roles

**LLM gateway** — the only code in the system that talks to a model provider:
- Adapters for a hosted API and for Ollama behind one interface; each role's config picks its model.
- Response cache keyed by (prompt-template version, input hash, model).
- Output validated against a JSON schema; one repair retry; then a typed failure.
- Concurrency limit, rate limiting, retries with backoff and jitter.
- **Cost meter with a hard cap**: projected spend is checked before a batch is submitted, per-role spend is a Prometheus counter, and the gateway refuses work that would exceed the budget.
- Temperature 0 and pinned model versions for every evaluation run.

**Three roles**

| Role | Trigger | Input → output | Path | Measured by |
|---|---|---|---|---|
| **1. Enrichment** | article published; backfill | title + abstract → taxonomy topics, normalized entities | nearline | ranker AUC with vs. without, overall and on articles < 6 h old |
| **2. Interest profile** | every 10 new clicks per user | recent clicked titles → two-sentence interest profile → embedded | nearline | semantic-channel Recall@100 for light users |
| **3. Listwise re-rank** | after serving page 1 (async), or inline in one experiment arm | profile + top-20 candidate summaries → permutation | nearline / sync arm | nDCG@10 delta (Protocol A), labeled-slot CTR (Protocol C), fallback rate, latency, cost per 1k sessions |

**Keeping the re-ranker off the critical path.** Producing a 20-item permutation means a prompt of roughly 1.5k tokens and tens of output tokens. Whether that fits a 200 ms request budget is measured in S4 rather than assumed — on a hosted small model or a local 4B model it usually will not. Three modes sit behind one interface, each an experiment arm:

1. **Async (default)**: page 1 is served in ranker order; the orchestrator enqueues a re-rank of the top-20; the result lands in `rr:{uid}` and applies to the next page or refresh. One LLM call per active session rather than per request.
2. **Sync with a deadline** (RankGPT-style generation): an inline call under its own, looser SLO arm (e.g., 1 s), with fallback rate and P99 reported — the honest cost of putting an LLM in the request path.
3. **Score instead of generate** (stretch): one forward pass reading next-token log-probabilities over candidate labels (LlamaRec-style verbalizer) on a local llama.cpp server — prefill-bound, no generation.

**Distillation** (stretch): LLM orders for ~20k offline lists become training targets for a small pairwise re-ranker, or a feature for the main ranker, which then serves synchronously inside budget. The measured question is how much of the LLM's delta survives distillation.

**Prompt hygiene**: candidate order is randomized before prompting (listwise LLM rankers show position bias — Hou et al., 2023), offline evaluation averages two permutations, and output is repaired into a valid permutation (missing or duplicate ids appended in ranker order) with the repair rate reported.

**Honest framing**: an LLM sees titles, not collaborative signal, and can lose to a trained ranker at click prediction. The experiment is built to report a negative delta as cleanly as a positive one.

**Token volumes (MIND-small) and the $20 cap**

| Job | Volume | Approx. tokens |
|---|---|---|
| Enrichment backfill | ~65k articles × (~250 in, ~40 out) | ~16M in · ~3M out |
| Interest profiles | evaluation users only, ~10k × (~600 in, ~60 out) | ~6M in · ~0.6M out |
| Re-rank, offline eval | 2k impressions × 2 permutations × (~1.5k in, ~80 out) | ~6M in · ~0.3M out |
| Distillation labels (stretch) | 20k lists × (~1.5k in, ~80 out) | ~30M in · ~1.6M out |

Price the chosen tier before S4 and size the jobs to the cap: batch endpoints and prompt caching of the shared instruction prefix cut cost substantially, and enrichment can be restricted to articles that appear in the training sample or the dev window. Local Qwen3 via Ollama costs nothing but is too slow on a CPU laptop for a 65k-article backfill; it handles online new articles and evaluation slices, or the backfill runs on a GPU machine if one is available.

### 7.8 C9 — Experimentation

- **Assignment**: `bucket = hash(layer_salt, user_id) mod 1000`, with independent layers for retrieval, ranking, policy and LLM, so experiments in different layers overlap without interfering (overlapping-experiment design, Tang et al., KDD 2010). Arms live in a versioned, hot-reloaded `experiments.yaml`; every `feed.served` records its assignment.
- **Validate before trusting**: the first experiment is an **A/A test** — across many A/A re-splits the false-positive rate should match α. Every analysis runs a **sample-ratio-mismatch** chi-square test first and refuses to report when it fails.
- **Analysis**: labeled-slot CTR per arm, with confidence intervals from a **user-clustered bootstrap**, since requests from one user are correlated. Guardrails: P99 latency, degradation rate, unlabeled share.
- **Paired replay**: the replayer is deterministic, so the same traffic can be served once per arm. Split traffic teaches the online mechanics; the paired version gives far lower variance for the actual decision. Both are reported, and the difference in interval width is itself a result.
- **Done when**: the A/A test passes; the SRM check gates every report; one real comparison (DeepFM vs. DIN) is analyzed both ways, with intervals.

### 7.9 C10 — Training Pipelines & Model Lifecycle

- **Pipelines are plain entry points** — `ff-build-dataset`, `ff-train-two-tower`, `ff-build-index`, `ff-train-ranker`, `ff-calibrate`, `ff-evaluate`, `ff-gate`, `ff-promote`, `ff-rollback`. The Airflow DAG chains them; a laptop or CI runs them directly.
- **Reproducibility**: every MLflow run records git SHA, data snapshot hash, feature-definition version, config and seeds. Artifacts include the ONNX graph, index generation, calibrator, metrics JSON and an evaluation report.
- **Evaluation gate** — promotion requires all of:

| Check | Rule |
|---|---|
| Protocol A (AUC, nDCG@10) | ≥ production, overall |
| Segments (cold / light / heavy users; articles < 6 h old; each top-level category) | no regression beyond a set tolerance |
| Protocol B Recall@100 (when the two-tower changed) | ≥ production, overall and cold-item |
| Calibration | ECE no worse than production beyond tolerance |
| Latency | batch-500 P99 ≤ 60 ms on the reference laptop |
| Contract | feature-definition version readable by the online store |
| Shadow | score-distribution PSI and latency reviewed; a large PSI blocks auto-promotion and asks a human |

- **Rollout**: shadow → 5% canary → production. A guardrail breach in canary (latency, degradation rate, labeled-slot CTR beyond tolerance) triggers automatic rollback by repointing the alias.
- **Staleness study** (stretch): over a multi-day replay carved from MIND-large's training window, compare a frozen bundle against a daily-retrained one. How fast does a news model go stale, in nDCG per day?
- **Done when**: one bundle goes train → gate → shadow → canary → production and back via `ff-rollback`, with every step visible in MLflow and Grafana.

### 7.10 C11 — Observability

| Metric | Type | Labels |
|---|---|---|
| `request_duration_seconds` | histogram | service, stage, bundle |
| `funnel_candidates` | histogram | stage (retrieved / merged / ranked / served), channel |
| `degraded_total` | counter | reason |
| `feature_lag_seconds` | histogram | feature group |
| `feature_parity_mismatch_ratio` | gauge | feature |
| `consumer_lag_messages` | gauge | topic, consumer group |
| `llm_calls_total`, `llm_tokens_total`, `llm_cost_usd_total` | counter | role, model, result |
| `labeled_ctr`, `unlabeled_share` | gauge | experiment layer, arm |
| `score_psi` | gauge | model, reference |
| `bundle_info` | gauge | component versions |

- **Dashboards**: service health · funnel (500 → 50 → 10, per channel) · freshness · model quality · experiments · LLM cost.
- **Alerts**: P99 over SLO for 5 min; degradation rate > 5%; feature lag P99 > 5 s; consumer lag growing; LLM spend > 80% of cap.
- **Debug trace store**: the per-request stage trace (§7.6) — the fastest answer to "why did this user see this article?".

## 8. Evaluation Plan

### 8.1 Three protocols, because no single one is honest on its own

| Protocol | Question | Unit | Metrics | Comparable to |
|---|---|---|---|---|
| **A — Impression ranking** (MIND official) | Given what was shown, is it ordered well? | each dev impression's shown articles | AUC, MRR, nDCG@5, nDCG@10; GAUC, LogLoss | published MIND results (NRMS, NAML, LSTUR and successors) — pull exact numbers from the papers and the Microsoft `recommenders` repo before claiming parity |
| **B — Full-corpus retrieval** (ours) | From everything in the window, are the clicked articles found? | dev impressions with ≥ 1 click; corpus = articles in the freshness window at impression time | Recall@50/100/200, cold-item Recall@K, window coverage | internal baselines only; recall is a lower bound (unshown ≠ irrelevant) |
| **C — End-to-end replay** (ours) | Does the whole served feed do better? | every replayed request | matched-click hit@10, labeled-slot CTR, **unlabeled share**, intra-list diversity, catalog coverage, mean article age | other FeedForge configurations, and the logging policy on the same requests |

### 8.2 Ablation ladder (filled in as sprints land)

Each row scores Protocol A with its own final scoring function applied to the impression's shown articles.

| Configuration | A: nDCG@10 | B: Recall@100 (all / cold) | C: labeled CTR · unlabeled share | P99 |
|---|---|---|---|---|
| Popularity + recency | | | | |
| + two-tower (content item tower) | | | | |
| + semantic + category channels | | | | |
| + DeepFM | | | | |
| + DIN | | | | |
| + LLM enrichment features | | | | |
| + policy layer (dedup, MMR, exploration) | | | | |
| + LLM re-rank (async) | | | | |

### 8.3 System results

| Measurement | Configurations |
|---|---|
| QPS vs. P50 / P99 curve | services vs. monolith; cache warm vs. cold; LLM sync arm on vs. off |
| Feature lag P99 | replay ×10, ×60, ×300 |
| Time-to-servable, time-to-enriched | P50 / P95 |
| Degradation behavior | each chaos scenario: error rate (target 0), degradation counter, P99 |
| LLM economics | cost per 1k sessions, cache hit rate, fallback rate per mode |

Performance is measured locally and checked in as a **versioned benchmark table**, so a regression shows up in a diff. CI proves correctness, not speed.

**Honest-reporting rules**: Protocol C is never compared with published numbers; every replay metric is printed with its unlabeled share; LLM results state model, prompt version and temperature; a hypothesis that fails is reported as failed.

## 9. Testing Strategy

| Level | What it proves | Runs |
|---|---|---|
| Unit + property tests | merge never returns duplicates; policy never returns clicked or seen articles; decay uses event time | every PR |
| Contract tests | shared Pydantic models; Schemathesis against the gateway's OpenAPI spec; event schemas validated end to end | every PR |
| Leakage test | the point-in-time builder never sees the future (§7.2) | every PR |
| Parity test | online and offline features agree (§7.2) | every PR on fixtures; continuously during replay |
| Model behavior tests | candidate order does not change scores (invariance); adding sports clicks raises the sports share of retrieval (directional); PyTorch ≡ ONNX | every PR |
| E2E golden test | `docker compose up` on a synthetic MIND-shaped fixture (200 users, 2k articles, 2 h of events): replay → feeds served → features updated within 5 s → `feed.served` joined → metrics exported → DLQs empty | every PR |
| Chaos test | under load, kill in turn the LLM worker, one retrieval channel, ranking, Redis → zero 5xx, the right `degraded_total` reason, P99 recorded | nightly / before a release |
| Load test | Locust scenarios from §8.3 | locally; results checked in |

The synthetic fixture generator pays off twice: MIND is never committed, and a tiny deterministic dataset makes the full-stack test fast enough to run on every pull request.

## 10. Repository Layout

```
feedforge/
├── contracts/    Pydantic API + event schemas shared by every service — the single source of truth
├── features/     feature registry · online updater · point-in-time builder · parity check
├── data/         MIND download + ingest → Parquet · first_seen derivation · splits · fixture generator
├── models/
│   ├── two_tower/   towers, losses (logQ, impression negatives), ONNX export
│   └── ranker/      LR · DeepFM · DIN · NRMS reference · calibration
├── llm/          gateway client · provider adapters · versioned prompts · output schemas · cost meter
├── services/     gateway · orchestrator (+ policy) · retrieval · ranking · llm_gateway
├── stream/       replayer · feature-updater · item-pipeline · llm-worker · metrics-joiner · log-sink
├── pipelines/    build_dataset · train_* · build_index · calibrate · evaluate · gate · promote · rollback
├── eval/         protocols A/B/C · ablations · experiment analysis · report generation
├── ops/          compose files + profiles · Grafana dashboards · Prometheus rules · Locust · chaos scripts
├── benchmarks/   versioned latency / QPS / lag tables
└── tests/        unit · contract · leakage · parity · behavior · e2e
```

**One-command demo**: `uv run ff-demo` brings up compose, bootstraps state, replays ten minutes at ×60, opens Grafana, then kills the ranking service so the degradation shows live. The five-minute demo is part of the deliverable.

## 11. Milestones (19 mentoring hours)

| Sprint | Hours | Deliverable | Exit criterion |
|---|---|---|---|
| **S0 — Kickoff & walking skeleton** | 2 | Repo, `uv`, compose (Redis, Redpanda, Prometheus, Grafana); MIND ingest + EDA (unseen-article share, window coverage, `history` semantics); **every component stubbed**: popularity retrieval, LR ranker, replayer, feature updater, metrics joiner | One replay runs end to end and Grafana shows QPS, P99 and the funnel |
| **S1 — Data & feature platform** | 3 | Feature registry, point-in-time builder, online updater, state bootstrap, leakage + parity tests, fixture generator, e2e test in CI | Parity < 0.1%; feature lag P99 < 5 s; CI green |
| **S2 — Retrieval** | 4 | Content two-tower (logQ, impression negatives); semantic / category / trending / fresh channels; merge + quotas; window index generations via `item.vectors`; ANN study | Protocol B ablations; Recall beats the baseline overall and on cold items |
| **S3 — Ranking & policy** | 3 | DeepFM → DIN, calibration, ONNX serving, NRMS reference; policy layer | Protocol A table; batch-500 P99 ≤ 60 ms; **go/no-go below** |
| **S4 — LLM layer** | 3 | LLM gateway with cost cap; enrichment + taxonomy; interest profiles; re-rank in async and sync arms | Each role ablated; cost per 1k sessions reported; spend under cap |
| **S5 — Lifecycle, experiments, hardening** | 2 | Registry + gate + shadow + canary + rollback; layered A/B with A/A and SRM; Locust; chaos test; monolith vs. services | Benchmark table complete; chaos test passes |
| **S6 — Interview sprint** | 2 | Résumé bullets with real numbers; ML system design mock ("design a news feed"); deep-dive drill; five-minute demo rehearsed | — |

**S0 go/no-go**: if the skeleton is not running end to end by the end of week 2, switch the event bus to the Redis Streams adapter and leave Airflow out. The loop matters more than the brand names on it.

**S3 go/no-go**: with the real two-tower and ranker in place, measure `/v1/feed` P99 at the ×60 replay rate on the development laptop. If it exceeds 200 ms, respond in this order: cut merged candidates 500 → 300, drop the channel with the lowest unique recall, tune ONNX threads, then run ranking in-process. The constraint surfaces in week 6, not week 11.

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Infrastructure eats the schedule | Walking skeleton in S0; event bus behind an interface with a Redis Streams fallback; Airflow optional; Compose is the ceiling |
| Laptop memory | Core footprint ≈ 4 GB on MIND-small; MIND-large replay on a 200k-user sample; heavy profiles opt-in |
| LLM re-rank too slow for the request path | Async mode by default; sync only as a measured experiment arm; distillation as the in-budget path |
| LLM cost | Gateway cost cap; batch endpoints and prefix caching; enrichment limited to articles actually used; local Qwen fallback |
| LLM re-rank does not beat the ranker | Framed as a hypothesis; a negative delta is a valid, reported result |
| Replay metrics over-interpreted | Labeled-slot metrics only, unlabeled share always shown, never compared with published numbers |
| Training/serving skew | Shared feature definitions, parity test in CI, LLM-feature masking at the serving rate |
| Version skew between towers, index and ranker | Bundles promoted atomically; each index generation records the model that built it |
| MIND licensing / repository hygiene | Data downloaded by script, never committed; CI runs on synthetic fixtures |
| MIND leaderboard unavailable | Dev split is the test set throughout; the leaderboard is only a stretch |

## 13. Résumé Bullets (experience voice; numbers filled in as sprints land)

- Built an **end-to-end news feed recommender** on Microsoft MIND impression logs — five-channel retrieval, DIN CTR ranking and an LLM augmentation layer behind a deadline-propagating microservice architecture — serving a ranked feed at **P99 X ms / X QPS** on a CPU-only laptop, with a six-rung degradation ladder verified by chaos tests (zero 5xx with any dependency killed).
- Designed a **two-tower retriever with a content-based item tower** for short-lived news, trained with logQ-corrected in-batch negatives plus real impression negatives, raising **cold-item Recall@100 from X to X** over an ID-based tower; new articles become retrievable **X s** after publication.
- Built a **streaming feature platform** (Redpanda → Redis) from a single feature registry shared by the online and point-in-time offline paths, with **P99 feature freshness of X s** and an online/offline **parity test in CI** holding mismatch under **X%**.
- Measured **three LLM roles** separately under a hard cost cap — article enrichment (**+X AUC** on articles under 6 h old), cold-user interest profiles (**+X Recall@100**), and asynchronous listwise re-ranking (**X nDCG@10** at **$X per 1k sessions**) — keeping the LLM off the synchronous path.
- Shipped a **model lifecycle and experimentation stack**: atomic bundle promotion through a multi-segment evaluation gate, shadow scoring, canary buckets and one-command rollback; layered A/B bucketing validated by A/A tests and SRM checks, with **paired log replay cutting confidence-interval width X×**.

## 14. Interview Leverage

- **ML system design** — "Design a news / short-video feed": the candidate has built every box, can draw the online / nearline / offline split, and knows each stage's latency budget from measurement.
- **ML deep dive** — content vs. ID embeddings under item churn; in-batch vs. impression negatives and logQ correction; exposure bias and why replay metrics need an unlabeled share; calibration under negative downsampling; target attention (DIN); position bias in listwise LLM ranking.
- **Data & feature engineering** — point-in-time correctness, the missing-feature trap, event time vs. wall-clock time, parity testing.
- **SDE system design** — deadline propagation, circuit breakers, graceful degradation, copy-on-write indexes, snapshot + log recovery, idempotent consumers, cursor pagination, and monolith vs. microservices with the latency cost measured.
- **MLOps** — release bundles and version skew, segment-level evaluation gates, shadow → canary → rollback, model staleness.
- **Experimentation** — layered bucketing, A/A tests, SRM, clustered bootstrap, why paired replay has lower variance, and why logging propensities matters for the next model.
- **Product judgement** — when an LLM belongs in the request path, and when it belongs one tier slower.
