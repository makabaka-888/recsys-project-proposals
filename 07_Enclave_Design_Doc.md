# Enclave — Design Document

**Zero-Egress Retrieval & Answer Engine — Fully Self-Hosted, Open-Weight Only**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Deployment | 100% local. No inference API, no managed vector DB, no telemetry. CPU-only viable; single consumer GPU optional. |
| Models | Qwen3-Embedding-0.6B (retrieval) · Qwen3-Reranker-0.6B (ranking) · Qwen3-4B-Instruct (synthesis) — all Apache-2.0, all local |

---

## 1. The Product Thesis

Some organizations cannot send their documents to an inference API. Legal, healthcare, finance, defense, and any company with a strict data-residency policy are all in this position — and it is precisely the segment that gets the worst search tooling, because the good tooling is SaaS.

**Enclave is a retrieval and answering engine where every model weight runs inside the customer's network.** No document, query, or embedding ever crosses the boundary.

The engineering thesis follows directly from that constraint:

> **Reach API-competitive retrieval quality with ~1.2B total parameters of open weights, by spending effort on architecture and domain adaptation instead of model scale.**

Three claims the project sets out to prove with measurements, not assertions:

1. **Hybrid retrieval + cross-encoder reranking closes most of the gap** to much larger models. A 0.6B reranker over a well-built candidate set beats a large embedder used alone.
2. **A 0.6B embedder fine-tuned on the customer's own domain beats a frontier API embedder that cannot be fine-tuned** — and that never sees the data. Small models are the only ones you can actually afford to specialize.
3. **Zero egress is verifiable**, not a marketing claim: an automated test asserts the system produces answers with all outbound network access blocked.

That third point is what makes this read as real product work rather than a course exercise.

## 2. Goals & Non-Goals

### Goals
- **Hybrid retrieval**: BM25 + dense (Qwen3-Embedding-0.6B) fused by reciprocal rank fusion.
- **Cross-encoder reranking**: Qwen3-Reranker-0.6B over the fused candidate set, with rerank depth chosen from a measured quality/latency curve.
- **Grounded answer synthesis**: local Qwen3-4B-Instruct with inline citations, plus a verifier pass that rejects unsupported sentences and abstains when evidence is thin.
- **Iterative retrieval** for multi-hop questions: decompose → retrieve → check sufficiency → retrieve again, bounded hop count.
- **Domain adaptation, locally**: hard-negative mining from feedback logs → LoRA fine-tune of the embedder and reranker → measured lift on a held-out domain set.
- **Honest evaluation** against public IR benchmarks (BEIR subsets) with published baselines, not self-reported numbers.
- **Resource engineering**: quantization, Matryoshka dimension truncation, ANN index selection, batching — each a measured trade-off.
- **Operability**: single `docker compose up`, health checks, metrics, structured logs, async ingestion jobs, and an egress-blocked integration test in CI.

### Non-Goals
- Training embedding or language models from scratch (open weights, adapted with LoRA).
- Multimodal input — this is a **pure-text** system by design.
- Multi-tenant SaaS control plane. Single-tenant, on-prem; multi-tenancy is discussed as a design extension only.
- GPU clusters. If it needs more than one consumer GPU, it is out of scope.

## 3. Model Choices

| Role | Model | Why | Approx. footprint |
|---|---|---|---|
| Dense retrieval | **Qwen3-Embedding-0.6B** | Apache-2.0; strong multilingual MTEB performance for its size; **instruction-aware** (task prefix changes the embedding — useful for query/document asymmetry); **Matryoshka** output dims (truncate 1024 → 512/256 and measure the loss) | ~1.2 GB fp16, ~0.6 GB int8 |
| Reranking | **Qwen3-Reranker-0.6B** | Apache-2.0 cross-encoder; scores (instruction, query, document) jointly, which is why it beats any bi-encoder at the same size; the natural stage-2 model | ~1.2 GB fp16, ~0.6 GB int8 |
| Lexical retrieval | **BM25** (DuckDB FTS or Tantivy) | Zero model cost; strong on rare terms, exact identifiers, and acronyms — precisely where dense retrieval is weakest | — |
| Answer synthesis | **Qwen3-4B-Instruct** (GGUF Q4_K_M via Ollama / llama.cpp) | Fits comfortably in RAM; good instruction following and citation formatting; swappable for 1.7B on constrained hosts or 8B where a GPU exists | ~2.5 GB Q4_K_M |
| Verifier | Qwen3-Reranker-0.6B, reused | A reranker is a relevance scorer — reusing it to score (claim, cited passage) is cheap and avoids a fourth model | reused |

**Model registry is local**: weights pulled once at build time into an image layer or a mounted volume, pinned by revision hash. Runtime never reaches Hugging Face.

*Verify current model cards before Sprint 1 — the Qwen3 embedding/reranker family ships variants at 0.6B / 4B / 8B, and dimension and context specifics should be confirmed against the published card rather than taken from this doc.*

## 4. Datasets & Evaluation Corpora

### Measured benchmark (public, labeled)

| Dataset | Size | Role |
|---|---|---|
| **SciFact** (BEIR) | ~5k docs, ~300 queries | Fast iteration loop — a full eval runs in minutes |
| **NFCorpus** (BEIR) | ~3.6k docs, ~320 queries | Medical domain; tests out-of-domain generalization |
| **FiQA** (BEIR) | ~57k docs, ~650 queries | Longer, opinionated passages; harder reranking |
| **CQADupStack** (BEIR) | per-subforum ~10–40k docs | Technical Q&A with duplicate-link labels — realistically messy phrasing |
| **HotpotQA** (subsampled corpus) | subsample to ~200k passages | Multi-hop retrieval and the iterative-retrieval module |

BEIR is the right choice because **published baselines exist for BM25, dense retrievers, and cross-encoder rerankers**. Every number this project reports sits next to a number someone else reported on the same split.

### Demo corpus (realistic, unlabeled)
Open-source project documentation (e.g. PostgreSQL, Kubernetes, Python) — genuinely messy: nested headings, code blocks, tables, version drift, heavy cross-referencing. Good for the ingestion and chunking work, and it makes a demo that looks like an internal knowledge base.

### Domain-adaptation set
Hard negatives mined from the demo corpus plus feedback labels collected through the UI. Used for the LoRA fine-tune in Sprint 4 and evaluated on a held-out slice.

## 5. Architecture

```
                      ┌─────────────────────────────────────────┐
                      │  NO OUTBOUND NETWORK ACCESS AT RUNTIME  │
                      └─────────────────────────────────────────┘

 Client
   ├── POST /v1/ask         question → grounded answer + citations
   ├── GET  /v1/search      hybrid search → ranked passages
   ├── POST /v1/ingest      document ingestion job (async, 202 + job id)
   ├── POST /v1/feedback    thumbs up/down, per-passage relevance
   └── GET  /v1/eval/runs   benchmark run history
                    │
          API Gateway (FastAPI) · auth · rate limit · request_id
                    │
              Orchestrator
                    │
   ┌────────────────┼─────────────────────────────────────┐
   │                │                                     │
 QUERY              RETRIEVAL (stage 1)              RERANK (stage 2)
 UNDERSTANDING      ├ BM25          (DuckDB FTS)     Qwen3-Reranker-0.6B
 local LLM:         └ dense         Qwen3-Emb-0.6B   cross-encode top-100
 rewrite ·            → FAISS ANN                    → top-10 passages
 decompose          → RRF fusion → top-100           (depth from the
 · instruction                                        measured knee)
   prefix
                    │
              SUFFICIENCY CHECK ──── insufficient ──► another hop
                    │                                (max N, then abstain)
                    ▼
              SYNTHESIS  Qwen3-4B-Instruct
              answer with inline [doc:chunk] citations
                    │
              VERIFIER  reranker scores (sentence, cited passage)
              unsupported sentence → revise once → else abstain
                    ▼
                 Response

 Offline:  ingest → parse → structure-aware chunk → dedup (MinHash)
           → embed (batch) → FAISS index build → alias swap
           feedback → hard-negative mining → LoRA fine-tune → eval gate → promote
 Ops:      Prometheus/Grafana · structured logs · Locust · egress test in CI
```

## 6. Component Design

### 6.1 Ingestion & Chunking
- Parsers for Markdown, HTML, PDF (PyMuPDF), plain text — all local libraries.
- **Structure-aware chunking**: split on heading boundaries, keep code blocks and tables intact, carry the heading path as chunk metadata (`Postgres > Indexes > GIN`). Compared against naive fixed-size chunking with overlap; the comparison is a measured ablation, because chunking usually moves retrieval quality more than model choice does.
- **Deduplication** via MinHash/SimHash — documentation corpora are full of near-duplicates that otherwise flood the top-k.
- Idempotent ingestion keyed on content hash; re-running a job is safe.
- Index build produces a new FAISS index, then an **alias swap** for zero-downtime rollover.

### 6.2 Query Understanding
- **Instruction prefixes**: Qwen3-Embedding is instruction-aware, so queries and documents are embedded with different task instructions. Nearly free, typically worth real points — an easy early win that also teaches asymmetric encoding.
- **Local LLM query rewriting**: expand acronyms, resolve pronouns from conversation context, split conjunctive questions. Cached by normalized query.
- **Decomposition** for multi-hop questions, feeding the iterative loop.

### 6.3 Stage 1 — Hybrid Retrieval
- **BM25** over DuckDB FTS (or Tantivy) — no server to run, embarrassingly local.
- **Dense** — Qwen3-Embedding-0.6B, embeddings precomputed in batch, served from FAISS.
- **RRF fusion** of the two ranked lists, then dedup.
- Ablation table: BM25-only vs. dense-only vs. hybrid. The interesting result is *where* each wins — dense on paraphrase, BM25 on rare identifiers — which is the answer to a very common search-interview question.

### 6.4 Stage 2 — Cross-Encoder Reranking
- Qwen3-Reranker-0.6B scores each (instruction, query, passage) pair.
- **Rerank depth is a first-class experiment**: NDCG@10 as a function of candidates reranked (10 / 25 / 50 / 100 / 200) against P99 latency. Find the knee, document the choice. This is the highest-leverage latency decision in the system.
- Throughput engineering: dynamic batching by sequence length, truncation policy, INT8/ONNX export for CPU, optional GPU path.

### 6.5 Iterative Retrieval (multi-hop)
- After reranking, a **sufficiency check**: reranker top-score threshold plus a local-LLM judgement of whether the retrieved evidence can answer the question.
- If insufficient, generate a follow-up sub-query targeting what is still missing, retrieve again, accumulate evidence. Hard cap on hops; on exhaustion the system **abstains** rather than guessing.
- Evaluated on HotpotQA: answer EM/F1 and supporting-passage recall, single-hop vs. iterative.

### 6.6 Synthesis & Verification
- Qwen3-4B-Instruct generates the answer with inline `[doc:chunk]` citations, constrained to the reranked passages.
- **Verifier pass** reuses the reranker: each generated sentence is scored against the passage it cites; sentences below threshold are dropped or trigger one revision. Reported as **unsupported-claim rate** and **abstention rate**.
- Design stance worth defending in interviews: for this customer segment a confident wrong answer is far more expensive than "I don't know" — so abstention is a feature with a target rate, not a failure.

### 6.7 Domain Adaptation — the differentiating module
1. Collect feedback: per-passage thumbs up/down from the UI, plus click-through on search results.
2. **Hard-negative mining**: passages retrieved highly but marked irrelevant, plus BM25/dense disagreement cases, which are reliably informative.
3. **LoRA fine-tune** the embedder with a contrastive objective (InfoNCE over mined hard negatives) and the reranker with a pairwise/listwise objective.
4. **Evaluation gate**: promote the adapter only if held-out domain NDCG@10 improves *and* BEIR out-of-domain performance has not regressed beyond a set tolerance — i.e. explicitly test for catastrophic forgetting.
5. Everything runs locally: LoRA on a 0.6B model is minutes on a consumer GPU, hours on CPU. **This is only possible because the models are small — the core argument of the project.**

### 6.8 Resource & Latency Engineering
Each of these is a measured experiment with a documented decision:

| Knob | Experiment | Trade-off measured |
|---|---|---|
| Quantization | fp16 vs. INT8/ONNX for embedder & reranker | NDCG@10 loss vs. latency and RAM |
| **Matryoshka dims** | 1024 → 512 → 256 → 128 | Recall@100 loss vs. index size and search speed |
| ANN index | Flat vs. IVF-PQ vs. HNSW | Recall vs. QPS vs. memory |
| Rerank depth | top-10 … top-200 | NDCG@10 vs. P99 latency |
| LLM size | 1.7B vs. 4B vs. 8B (GGUF quants) | Answer quality vs. tokens/sec vs. RAM |

Output: a Pareto write-up with a recommended CPU-only profile and a GPU profile.

### 6.9 Deployment & Operability
- **`docker compose up`** brings up `api`, `worker` (ingestion/embedding), `postgres` (metadata, feedback), `redis` (cache), and either `ollama`/`vllm` or in-process llama.cpp for the LLM.
- Model weights baked into an image layer or mounted volume, **pinned by revision hash**; runtime has no registry access.
- **Egress test in CI**: run the stack on an internal-only Docker network with no default route, execute a golden query set, assert correct answers and zero outbound connection attempts. This test *is* the product guarantee.
- Prometheus metrics: per-stage latency histograms, QPS, cache hit rate, rerank depth, abstention rate, hops per query, tokens/sec. Grafana dashboards for service health and answer quality.
- Locust load tests reported per profile (CPU-only vs. GPU).
- Hardware tiers documented: **minimum 16 GB RAM, CPU-only**; recommended 8–12 GB VRAM GPU. Total model disk ≈ 5 GB.

## 7. Evaluation Plan

| Area | Metrics | Compared against |
|---|---|---|
| Retrieval | NDCG@10, Recall@100, MRR on SciFact / NFCorpus / FiQA / CQADupStack | **Published BEIR baselines** (BM25, dense, reranked) |
| Pipeline ablation | BM25 → dense → hybrid → +rerank → +query rewrite → +LoRA | Each stage's marginal contribution |
| Chunking | structure-aware vs. fixed-size | NDCG@10 delta |
| Multi-hop | Answer EM/F1, supporting-passage recall | Single-hop baseline on HotpotQA |
| Answer quality | Citation-grounding rate, unsupported-claim rate, abstention rate | Held-out annotated set (~200 questions) |
| Domain adaptation | Held-out domain NDCG@10 before/after LoRA; BEIR regression check | Pre-tuning checkpoint |
| System | P50/P99 per stage, QPS, RAM/VRAM, index size, tokens/sec | CPU vs. GPU profile |
| Guarantee | Egress test pass/fail | — |

**Honest-reporting rules baked into the plan**: LLM-as-judge results are labeled as such with their known biases; the annotated set is small and that is stated; no comparison is claimed against an API model the project never actually ran.

## 8. Milestones (19 hours / 15 sessions)

| Sprint | Sessions | Deliverable |
|---|---|---|
| **S0** | 3–4 | Agile workflow, AI tooling, repo, local model pulls, `docker compose` skeleton |
| **S1** | 5–8 | Ingestion + structure-aware chunking; BM25 + Qwen3-Embedding + FAISS; RRF hybrid; **BEIR baseline numbers**; `/v1/search` live |
| **S2** | 9–10 | Qwen3-Reranker integration; **rerank-depth, quantization and MRL trade-off study**; full ablation table |
| **S3** | 11–12 | Synthesis with citations, verifier pass, iterative retrieval, abstention policy; HotpotQA multi-hop eval |
| **S4** | 13–14 | Hard-negative mining + LoRA fine-tune with regression gate; monitoring, load tests, **egress test in CI**; ops runbook |
| — | 15 | Interview sprint |

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reranker latency dominates on CPU | Depth study finds the knee; INT8/ONNX export; Redis caching; rerank only when stage-1 scores are ambiguous |
| 0.6B embedder underperforms larger models | That gap is the point — measure it, then close it with hybrid + rerank + LoRA and report the remaining delta honestly |
| LoRA overfits the small feedback set | Held-out domain eval **and** BEIR regression gate; adapter promoted only if both pass |
| Full BEIR suite is slow to evaluate | Iterate on SciFact/NFCorpus (minutes per run); larger sets only for milestone runs |
| Chunking work quietly eats the schedule | Timeboxed in S1 with a fixed comparison design (two strategies, not five) |
| Local LLM answer quality is weak | Retrieval quality carries most of the value; synthesis model is swappable; abstention protects against confident nonsense |
| "Local-only" tempts scope creep into infra | Docker Compose is the ceiling; Kubernetes and multi-tenancy are explicitly out of scope |

## 10. Résumé Bullets (experience voice; numbers filled in as sprints land)

- Built a **fully self-hosted retrieval and answering engine** with zero outbound network dependencies, combining BM25 and Qwen3-Embedding-0.6B dense retrieval (RRF fusion) with Qwen3-Reranker-0.6B cross-encoder reranking, reaching **NDCG@10 of X on BEIR SciFact/FiQA** — within X points of published baselines from models Xx larger.
- Ran a **systematic resource/quality trade-off study** across quantization (fp16 vs. INT8), Matryoshka embedding dimensions (1024→256), ANN index families (Flat / IVF-PQ / HNSW), and rerank depth (10–200), cutting index memory **Xx** and P99 latency **X%** at a cost of **X points** of NDCG@10.
- Improved domain retrieval quality **X%** by mining hard negatives from user feedback and **LoRA fine-tuning the 0.6B embedder and reranker locally**, gated on a held-out domain set plus a BEIR out-of-domain regression check to prevent catastrophic forgetting.
- Implemented **grounded answer synthesis** with a local Qwen3-4B model, inline citations, and a reranker-based verifier that rejects unsupported sentences, holding unsupported-claim rate below **X%** with an **X% abstention rate** on unanswerable questions.
- Added **iterative multi-hop retrieval** with sufficiency checking and bounded hops, raising HotpotQA supporting-passage recall from **X to X** over single-pass retrieval.
- Shipped the system as **Docker Compose services** (FastAPI, worker, Postgres, Redis, local LLM runtime) with Prometheus/Grafana monitoring, Locust load tests, and a **CI test that verifies correct operation with all egress blocked**.

## 11. Why This Reads as Real Company Work

- **The constraint is a customer requirement, not a budget limitation.** "We chose open weights because the data cannot leave the network" is a product decision with a buyer behind it. That framing survives any interview follow-up.
- **The guarantee is tested.** Most candidates claim an architecture; an egress-blocked CI test proves one. Interviewers notice tests that encode product promises.
- **The trade-off studies are what the job actually is.** Choosing rerank depth and embedding dimension against a latency budget is the daily work of an applied ML engineer, and almost no new-grad project shows it.
- **Domain fine-tuning is a genuine differentiator.** Very few candidates have trained a retrieval model on mined hard negatives with a forgetting-regression gate. It is also the clean commercial argument for why small open models matter.
- **Honest evaluation.** Public benchmarks with published baselines, and stated limitations on LLM-as-judge and annotation size. Measured, bounded claims earn more trust than round numbers.

## 12. Interview Leverage

- **Search / IR system design**: hybrid recall, cross-encoder reranking, fusion, chunking strategy, index selection — with real numbers behind each choice.
- **Efficiency engineering**: quantization, Matryoshka representation learning, ANN trade-offs, batching — the accuracy/latency/memory frontier, measured.
- **Retrieval model training**: contrastive learning, hard-negative mining, LoRA, catastrophic-forgetting evaluation. The deepest ML content here, and the rarest in a new-grad candidate.
- **LLM engineering**: grounding, verification, abstention policy, iterative/agentic retrieval, and cost control without an API to lean on.
- **SDE / platform**: containerized services, async ingestion jobs, alias-swap index rollover, caching, monitoring, CI that tests a product guarantee.
- **Product judgement**: knowing which market needs on-prem, and why a 0.6B model you can fine-tune beats a frontier model you cannot.
