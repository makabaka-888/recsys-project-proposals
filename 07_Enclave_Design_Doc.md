# Enclave — Design Document

**Zero-Egress Retrieval & Answer Engine — Fully Self-Hosted, Open-Weight, Cross-Platform**

| | |
|---|---|
| Status | Proposal |
| Target duration | ~10–12 weeks (19 mentoring hours + independent work) |
| Platforms | **Windows and macOS from one repository**, plus Linux for release. Verified by a three-OS CI matrix. |
| Deployment | 100% local. No inference API, no managed vector database, no telemetry. CPU-only viable; NVIDIA or Apple GPU used when present. |
| Core stack | PyTorch · sentence-transformers · Qwen3-Embedding-0.6B · Qwen3-Reranker-0.6B · Ollama (llama.cpp) · Postgres + pgvector · FastAPI |

---

## 1. The Product Thesis

Some organizations cannot send their documents to an inference API. Legal, healthcare, finance, defense, and any company under a strict data-residency policy are all in that position — and it is exactly the segment with the worst search tooling, because the good tooling is SaaS.

**Enclave is a retrieval and answering engine where every model weight runs inside the customer's own network.** No document, query, or embedding crosses the boundary.

Two engineering theses follow from that constraint:

> **1. Reach API-competitive retrieval quality with ~1.2B parameters of open weights**, by spending effort on architecture and domain adaptation instead of model scale.
>
> **2. Ship one codebase that runs on whatever the customer already owns** — Windows workstation, Apple Silicon laptop, or Linux server — with no per-platform fork.

Four claims the project proves with measurements rather than assertions:

1. **Hybrid retrieval plus cross-encoder reranking closes most of the gap** to much larger models. A 0.6B reranker over a well-built candidate set beats a large embedder used alone.
2. **A 0.6B embedder fine-tuned on the customer's domain beats a frontier API embedder that cannot be fine-tuned** — and that never sees the data. Small models are the only ones you can afford to specialize.
3. **Zero egress is verifiable**, not marketing: an automated test asserts the system answers correctly with all outbound network access blocked.
4. **Cross-platform is verifiable too**: the same test suite passes on `windows-latest`, `macos-14` (arm64), and `ubuntu-latest`.

Claims 3 and 4 are what make this read as product work rather than a course exercise — both are enforced by CI, not by a paragraph in a README.

## 2. Goals & Non-Goals

### Goals
- **Hybrid retrieval** in a single store: Postgres full-text search (lexical) + pgvector (dense), fused by reciprocal rank fusion.
- **Cross-encoder reranking** with Qwen3-Reranker-0.6B, rerank depth chosen from a measured quality/latency curve per platform.
- **Grounded answer synthesis** with a local LLM via Ollama, inline citations, a verifier pass that rejects unsupported sentences, and abstention when evidence is thin.
- **Iterative retrieval** for multi-hop questions: decompose → retrieve → check sufficiency → retrieve again, bounded hops.
- **Local domain adaptation**: hard-negative mining from feedback → LoRA fine-tune of embedder and reranker → measured lift, gated against out-of-domain regression.
- **Evidence-based model selection**: Qwen3 embedding/reranker benchmarked against the BGE family, not chosen by reputation.
- **Honest evaluation** on public IR benchmarks (BEIR subsets) with published baselines.
- **One repository, two developer platforms**, proven by a green CI matrix.

### Non-Goals
- Training embedding or language models from scratch (open weights, adapted with LoRA).
- Multimodal input — **pure text** by design.
- Multi-tenant SaaS control plane; single-tenant on-prem, with multi-tenancy discussed only as an extension.
- GPU clusters. If it needs more than one consumer GPU, it is out of scope.
- Platform-specific forks or Mac-only / Windows-only dependencies.

## 3. Stack Choices — and Why Each Is Portable

### 3.1 Models

| Role | Primary | Benchmarked against | Why |
|---|---|---|---|
| Dense retrieval | **Qwen3-Embedding-0.6B** | **BGE-m3**, multilingual-E5 | Apache-2.0; strong MTEB results for its size; **instruction-aware** (different task prefixes for query vs. document); **Matryoshka** dims allow 1024→256 truncation. BGE-m3 is the long-standing industry default, so it is the baseline to beat rather than an afterthought. |
| Reranking | **Qwen3-Reranker-0.6B** | **bge-reranker-v2-m3** | Cross-encoder scoring of (instruction, query, document) jointly — the reason a 0.6B reranker outperforms any same-size bi-encoder. |
| Lexical retrieval | **Postgres full-text search** (`tsvector`, `ts_rank_cd`) | rank_bm25 in-process | No extra service, no extra container, identical on every OS. Strong exactly where dense retrieval is weak: rare identifiers, acronyms, version strings. |
| Answer synthesis | **Ollama** serving **Llama-3.1-8B-Instruct** (default) or **Qwen3-4B-Instruct** (low-RAM) | each other | Ollama wraps llama.cpp with an identical HTTP API on Windows, macOS, and Linux, and uses CUDA or Metal automatically. GGUF is the de facto local format. Comparing the two models is a cheap, concrete result. |
| Verifier | Qwen3-Reranker-0.6B, reused | — | A reranker is a relevance scorer; reusing it to score (claim, cited passage) avoids a fourth model. |

**On model families:** for embedding and reranking there is no competitive Llama-family model at this size — the recognized options are Qwen3 and BGE, and the project benchmarks both. For synthesis, Llama-3.1-8B-Instruct is the default because it is the most widely recognized open model in the North American market, with Qwen3-4B as the low-memory alternative. The synthesis layer is configuration, not architecture.

*Confirm current model cards before Sprint 1 — dimensions, context length, and available sizes should come from the published card, not from this document.*

### 3.2 Runtime — One Code Path, Three Backends

```python
# The entire platform abstraction for the encoder models
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
```

| Platform | Encoder / reranker | LLM |
|---|---|---|
| **Windows + NVIDIA** | PyTorch CUDA | Ollama (CUDA) |
| **Windows, CPU only** | PyTorch CPU, or ONNX Runtime INT8 | Ollama (CPU) |
| **macOS Apple Silicon** | PyTorch MPS | Ollama (Metal) |
| **Linux server** | PyTorch CUDA or ONNX INT8 | Ollama or llama.cpp |

No MLX, no platform-specific inference code. Apple's MLX is faster on M-series but is macOS-only, which would violate the single-codebase requirement — a trade the project deliberately declines and documents.

**Optional ONNX Runtime INT8 path** for CPU hosts, exported once and used on both Windows and macOS. Measured on both, because arm64 and x86 quantization kernels differ and parity should not be assumed.

### 3.3 Storage — Postgres Does Both Halves of Retrieval

**Postgres 16 + pgvector** in one container handles metadata, lexical search, and vector search:

- `pgvector` HNSW index for dense retrieval
- Native `tsvector` full-text index for lexical retrieval
- Chunks, documents, and feedback in the same transactional store

Why this over FAISS as the default:

1. **Identical behaviour on every OS.** The container is the same image everywhere. FAISS is installable on arm64 but has historically been the most fragile dependency in this kind of stack, and dependency friction on the student's own laptop is a real project risk.
2. **Industry recognition.** pgvector is the default vector store at a large number of companies; "hybrid search in Postgres" is a pattern an interviewer will recognize immediately.
3. **One system, one transaction.** Filters, joins, metadata, and vectors in the same query, with no dual-write consistency problem to invent.

**FAISS and usearch remain in the project** — as comparison points inside the index benchmark (§6.7), not as load-bearing dependencies. The study still answers "which index family, at what recall, at what memory", it just no longer decides whether the project installs.

Redis is used for query and rerank caching.

### 3.4 Tooling — Chosen for Windows/macOS Parity

| Concern | Choice | Why not the obvious alternative |
|---|---|---|
| Python env & deps | **`uv`** | Fast, lockfile-based, first-class Windows support; avoids conda-vs-pip drift across machines |
| Task running | **Python entry points** in `pyproject.toml` (or `just`) | `make` is awkward on Windows and would push developers onto different commands per OS |
| Paths | `pathlib` everywhere; no string concatenation | Backslash/forward-slash bugs are the classic cross-platform failure |
| Line endings | **`.gitattributes`** with `* text=auto eol=lf` | Otherwise CRLF churn produces noisy diffs and breaks shell scripts in containers |
| Services | `docker compose` for Postgres + Redis only | Model inference stays native so it can reach the GPU on both platforms (see §6.8) |
| Config | Pydantic Settings + `.env` | One schema, validated identically on both OSes |

## 4. Datasets & Evaluation Corpora

### Measured benchmark (public, labeled)

| Dataset | Size | Role |
|---|---|---|
| **SciFact** (BEIR) | ~5k docs, ~300 queries | Fast iteration — a full eval run takes minutes |
| **NFCorpus** (BEIR) | ~3.6k docs, ~320 queries | Medical domain; out-of-domain generalization check |
| **FiQA** (BEIR) | ~57k docs, ~650 queries | Longer, opinionated passages; harder reranking |
| **CQADupStack** (BEIR) | ~10–40k docs per subforum | Technical Q&A with duplicate labels; realistically messy phrasing |
| **HotpotQA** (subsampled) | ~200k passages | Multi-hop retrieval and the iterative module |

BEIR is chosen because **published baselines exist** for BM25, dense retrievers, and cross-encoder reranking. Every number this project reports sits beside a number someone else reported on the same split.

### Demo corpus (realistic, unlabeled)
Open-source project documentation (PostgreSQL, Kubernetes, Python) — nested headings, code blocks, tables, version drift, heavy cross-referencing. Drives the ingestion and chunking work and demos as an internal knowledge base.

### Domain-adaptation set
Hard negatives mined from the demo corpus plus feedback labels from the UI; used for the LoRA fine-tune and evaluated on a held-out slice.

## 5. Architecture

```
                ┌────────────────────────────────────────────────────┐
                │  NO OUTBOUND NETWORK ACCESS AT RUNTIME             │
                │  Verified on Windows, macOS and Linux in CI        │
                └────────────────────────────────────────────────────┘

 Client
   ├── POST /v1/ask        question → grounded answer + citations
   ├── GET  /v1/search     hybrid search → ranked passages
   ├── POST /v1/ingest     ingestion job (async, 202 + job id)
   ├── POST /v1/feedback   thumbs up/down, per-passage relevance
   └── GET  /v1/eval/runs  benchmark run history
                  │
        API Gateway (FastAPI) · auth · rate limit · request_id
                  │
            Orchestrator
                  │
  ┌───────────────┼────────────────────────────────────────┐
  │               │                                        │
 QUERY          RETRIEVAL (stage 1) — one Postgres      RERANK (stage 2)
 UNDERSTANDING  ├ lexical   tsvector / ts_rank_cd       Qwen3-Reranker-0.6B
 Ollama LLM:    └ dense     pgvector HNSW              cross-encode top-N
 rewrite ·        Qwen3-Embedding-0.6B                  → top-10
 decompose ·      → RRF fusion → top-N                  (N from the measured
 instruction                                             knee, per platform)
 prefixes
                  │
            SUFFICIENCY CHECK ──── insufficient ──► another hop
                  │                                 (max N, then abstain)
                  ▼
            SYNTHESIS  Ollama → Llama-3.1-8B / Qwen3-4B
            answer with inline [doc:chunk] citations
                  │
            VERIFIER  reranker scores (sentence, cited passage)
            unsupported → revise once → else abstain
                  ▼
               Response

 Offline:  ingest → parse → structure-aware chunk → dedup (MinHash)
           → embed (batch) → pgvector index → analyze
           feedback → hard-negative mining → LoRA → eval gate → promote
 Ops:      Prometheus/Grafana · Locust · egress test · 3-OS CI matrix
```

## 6. Component Design

### 6.1 Ingestion & Chunking
- Parsers for Markdown, HTML, PDF (PyMuPDF), plain text — all pure-Python or wheel-available on both platforms.
- **Structure-aware chunking**: split on heading boundaries, keep code blocks and tables intact, carry the heading path as metadata (`Postgres > Indexes > GIN`). Compared against fixed-size chunking with overlap as a measured ablation — chunking usually moves retrieval quality more than model choice does.
- **Deduplication** via MinHash — documentation corpora are full of near-duplicates that otherwise flood the top-k.
- Idempotent ingestion keyed on content hash; re-running a job is safe.
- Index maintenance: build into a new table or partition, then swap — no downtime, and it works the same in Postgres on both platforms.

### 6.2 Query Understanding
- **Instruction prefixes**: Qwen3-Embedding is instruction-aware, so queries and documents get different task instructions. Nearly free, usually worth real points, and it teaches asymmetric encoding.
- **LLM query rewriting** via Ollama: expand acronyms, resolve pronouns from context, split conjunctive questions. Cached in Redis by normalized query.
- **Decomposition** for multi-hop questions, feeding the iterative loop.

### 6.3 Stage 1 — Hybrid Retrieval in Postgres
- **Lexical**: `tsvector` column with a GIN index, ranked by `ts_rank_cd`. A `rank_bm25` in-process implementation is kept as a scoring comparison, since Postgres FTS ranking is not literally BM25.
- **Dense**: `pgvector` column with an HNSW index, cosine distance, embeddings written at ingestion time.
- **Fusion**: RRF over the two ranked lists, either in one SQL CTE or in application code — both implemented, and the SQL version measured for latency.
- Ablation: lexical-only vs. dense-only vs. hybrid. The instructive result is *where* each wins — dense on paraphrase, lexical on rare identifiers — which is the answer to a very common search-interview question.

### 6.4 Stage 2 — Cross-Encoder Reranking
- Qwen3-Reranker-0.6B scores each (instruction, query, passage) pair; `bge-reranker-v2-m3` benchmarked alongside it.
- **Rerank depth is a first-class experiment**: NDCG@10 versus candidates reranked (10 / 25 / 50 / 100 / 200) against P99 latency, **run separately on each platform profile**. The knee will land in different places on a CUDA desktop and an M-series laptop, and the config carries a per-profile default.
- Throughput work: dynamic batching by sequence length, truncation policy, ONNX INT8 for CPU hosts.
- **Conditional reranking**: skip stage 2 when stage-1 scores are unambiguous — most queries are easy, and this is the cheapest large latency win.

### 6.5 Iterative Retrieval (multi-hop)
- After reranking, a **sufficiency check**: reranker top-score threshold plus an LLM judgement of whether the evidence can answer the question.
- If insufficient, generate a follow-up sub-query for what is missing, retrieve again, accumulate evidence. Hard hop cap; on exhaustion the system **abstains** rather than guessing.
- Evaluated on HotpotQA: answer EM/F1 and supporting-passage recall, single-pass vs. iterative.

### 6.6 Synthesis & Verification
- The LLM generates answers with inline `[doc:chunk]` citations, constrained to reranked passages.
- **Verifier pass** reuses the reranker: each generated sentence is scored against the passage it cites; low scorers are dropped or trigger one revision. Reported as **unsupported-claim rate** and **abstention rate**.
- Position worth defending in interviews: for this customer segment a confident wrong answer costs far more than "I don't know", so abstention is a feature with a target rate.

### 6.7 Domain Adaptation & the Index Study

**LoRA fine-tuning (the differentiating module)**
1. Collect per-passage feedback and search click-through.
2. **Hard-negative mining**: passages retrieved highly but marked irrelevant, plus lexical/dense disagreement cases, which are reliably informative.
3. **LoRA fine-tune** the embedder with InfoNCE over mined hard negatives, and the reranker with a pairwise objective. PEFT runs identically on CUDA and MPS at this model size.
4. **Promotion gate**: held-out domain NDCG@10 must improve **and** BEIR out-of-domain performance must not regress beyond tolerance — an explicit catastrophic-forgetting test.
5. Feasible only because the models are small. That is the project's commercial argument, demonstrated rather than claimed.

**Index and efficiency study** — each a measured trade-off with a documented decision:

| Knob | Experiment | Measured against |
|---|---|---|
| Vector index | pgvector HNSW vs. IVFFlat vs. exact scan; FAISS and usearch in-process as external references | Recall@100, QPS, memory, index build time |
| **Matryoshka dims** | 1024 → 512 → 256 → 128 | Recall loss vs. index size and query speed |
| Quantization | fp16 vs. INT8/ONNX for encoder and reranker | NDCG@10 loss vs. latency and RAM, **on both x86 and arm64** |
| Rerank depth | top-10 … top-200 | NDCG@10 vs. P99, per platform |
| Synthesis model | Llama-3.1-8B Q4 vs. Qwen3-4B Q4 vs. Qwen3-1.7B Q4 | Answer quality vs. tokens/sec vs. RAM |

### 6.8 Deployment Topology — Why Models Run Native

One platform fact drives the topology:

> **Docker on macOS cannot reach the GPU.** Docker Desktop runs a Linux VM and Metal is not passed through. On Windows, containers *can* reach an NVIDIA GPU through WSL2, but only with NVIDIA hardware and extra setup.

Containerizing inference would therefore mean losing acceleration on macOS and adding setup burden on Windows. So the split is by concern:

| Component | Windows | macOS | Linux release |
|---|---|---|---|
| Encoder, reranker, LLM | **native** (uv venv + Ollama app) — CUDA or CPU | **native** — Metal via MPS + Ollama | container (CPU or CUDA) |
| API, worker | native in dev, container in release | same | container |
| Postgres + pgvector, Redis | **container** | **container** | container |

Developer experience is identical on both platforms: `uv sync`, `docker compose up db redis`, `ollama pull <model>`, then one Python entry point. `compose --profile release` runs everything containerized for the Linux target, so the production path stays fully containerized without hurting laptop development.

### 6.9 Memory & Hardware Profiles

| Machine | Encoders | Synthesis model | Notes |
|---|---|---|---|
| **Windows, 16 GB, NVIDIA ≥8 GB VRAM** | CUDA fp16 | Llama-3.1-8B Q4 (~4.9 GB) | Best profile; rerank depth can stay high |
| **Windows, 16 GB, CPU only** | ONNX INT8 (~0.6 GB each) | Qwen3-4B Q4 (~2.5 GB) | Lower rerank depth; conditional reranking on |
| **macOS Apple Silicon, 16 GB** | MPS fp16 (~2.4 GB both) | Llama-3.1-8B Q4 or Qwen3-4B Q4 | Unified memory shared with the OS; leave headroom |
| **macOS / Windows, 8 GB** | load on demand, INT8 | Qwen3-1.7B Q4 (~1.1 GB) | Interactive but constrained; eval loop still fine |
| **Intel Mac** | CPU only | Qwen3-1.7B Q4 | Documented as best-effort, not a supported profile |

Model weights on disk: ~2.5 GB encoders plus 1–5 GB per synthesis model. Demo corpus and Postgres data well under 5 GB.

### 6.10 Operability & CI

- **Three-OS CI matrix** on GitHub Actions: `windows-latest`, `macos-14` (arm64), `ubuntu-latest`. Unit tests, a small end-to-end retrieval test on SciFact, and lint run on all three. *(Note: macOS runner minutes carry a multiplier on private repositories — run the macOS job on pull requests to `main` and nightly rather than on every push.)*
- **Egress test**: run the stack with outbound access blocked — internal-only Docker network on Linux, a network-disabled harness on Windows and macOS — execute a golden query set, and assert both correct answers and zero outbound connection attempts. This test *is* the product guarantee.
- Model weights fetched once into a local cache, **pinned by revision hash**; runtime never contacts Hugging Face or any registry.
- Prometheus metrics: per-stage latency histograms, QPS, cache hit rate, rerank depth, abstention rate, hops per query, tokens/sec. Grafana dashboards for service health and answer quality.
- Locust load tests reported **per platform profile** — the cross-platform table is itself a result worth showing.
- Ops runbook: ingestion, index rebuild, adapter rollback, and what to check when latency regresses.

## 7. Evaluation Plan

| Area | Metrics | Compared against |
|---|---|---|
| Retrieval | NDCG@10, Recall@100, MRR on SciFact / NFCorpus / FiQA / CQADupStack | **Published BEIR baselines**; BGE-m3 and E5 run locally |
| Pipeline ablation | lexical → dense → hybrid → +rerank → +query rewrite → +LoRA | Marginal contribution of each stage |
| Chunking | structure-aware vs. fixed-size | NDCG@10 delta |
| Multi-hop | Answer EM/F1, supporting-passage recall | Single-pass baseline on HotpotQA |
| Answer quality | Citation-grounding rate, unsupported-claim rate, abstention rate | Held-out annotated set (~200 questions) |
| Domain adaptation | Held-out domain NDCG@10 before/after LoRA; BEIR regression check | Pre-tuning checkpoint |
| Efficiency | Recall/QPS/memory per index; latency per quantization and rerank depth | Across x86 and arm64 |
| Portability | Test suite green on three OSes; egress test pass | — |

**Honest-reporting rules**: LLM-as-judge results are labeled as such with their known biases; the annotated set is small and that is stated; no comparison is claimed against an API model the project never ran.

## 8. Milestones (19 hours / 15 sessions)

| Sprint | Sessions | Deliverable |
|---|---|---|
| **S0** | 3–4 | Agile workflow, AI tooling, repo scaffold, `uv` env, compose for Postgres/Redis, Ollama pulls, **CI matrix green on both machines** |
| **S1** | 5–8 | Ingestion + structure-aware chunking; hybrid retrieval in Postgres (tsvector + pgvector); **BEIR baselines + Qwen3 vs. BGE comparison**; `/v1/search` live |
| **S2** | 9–10 | Reranker integration; **rerank depth, quantization, Matryoshka and index studies, per platform**; full ablation table |
| **S3** | 11–12 | Synthesis with citations, verifier pass, iterative retrieval, abstention policy; HotpotQA multi-hop eval; Llama-3.1-8B vs. Qwen3-4B comparison |
| **S4** | 13–14 | Hard-negative mining + LoRA with regression gate; monitoring, load tests per profile, **egress test in CI**; ops runbook |
| — | 15 | Interview sprint |

**Sprint 1 go/no-go**: measure single-passage rerank latency and `/v1/search` P99 at depth 25/50/100 on the actual development machine. If interactive P99 exceeds ~2 s, respond in this order — lower depth, shorten truncation, enable conditional reranking, switch to ONNX INT8. The constraint gets discovered in week 2, not week 9.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reranker latency dominates on CPU | Depth study finds the knee per platform; ONNX INT8; conditional reranking; Redis caching |
| 0.6B encoder underperforms larger models | That gap is the subject, not a flaw — measure it, close it with hybrid + rerank + LoRA, report the remainder honestly |
| Cross-platform drift (works on one machine only) | CI matrix on all three OSes from Sprint 0, so drift fails a build instead of surfacing in week 9 |
| pgvector HNSW slower than FAISS at scale | Corpus is ≤10⁶ vectors where the gap is small; FAISS stays in the benchmark, and the index sits behind an interface if it ever needs swapping |
| LoRA overfits a small feedback set | Held-out domain eval **and** BEIR regression gate; adapter promoted only if both pass |
| macOS CI minutes on a private repo | macOS job restricted to PRs into `main` and a nightly schedule |
| Chunking work quietly eats the schedule | Timeboxed in S1 with a fixed two-strategy comparison |
| "Local-only" invites infra scope creep | Compose plus native models is the ceiling; Kubernetes and multi-tenancy are out of scope |

## 10. Résumé Bullets (experience voice; numbers filled in as sprints land)

- Built a **fully self-hosted retrieval and answering engine** with zero outbound network dependencies, combining Postgres full-text search and pgvector dense retrieval (RRF fusion) with Qwen3-Reranker-0.6B cross-encoder reranking, reaching **NDCG@10 of X on BEIR SciFact/FiQA** — within X points of published baselines from models Xx larger.
- Delivered **one codebase running on Windows, macOS (Apple Silicon), and Linux**, with a single device-resolution layer over CUDA / MPS / CPU-INT8 and a three-OS CI matrix including an **egress-blocked integration test** that verifies the no-network guarantee.
- Ran a **systematic efficiency study** across vector index families (pgvector HNSW / IVFFlat / exact, benchmarked against FAISS and usearch), Matryoshka embedding dimensions (1024→256), quantization (fp16 vs. INT8) on both x86 and arm64, and rerank depth (10–200) — cutting index memory **Xx** and P99 latency **X%** for **X points** of NDCG@10.
- Improved domain retrieval quality **X%** by mining hard negatives from user feedback and **LoRA fine-tuning the 0.6B embedder and reranker locally**, gated on a held-out domain set plus a BEIR out-of-domain regression check to prevent catastrophic forgetting.
- Selected retrieval models on evidence, benchmarking **Qwen3-Embedding/Reranker against BGE-m3 and bge-reranker-v2-m3** across four BEIR datasets, and **Llama-3.1-8B against Qwen3-4B** for grounded synthesis quality and tokens/sec.
- Implemented **grounded synthesis** with inline citations and a reranker-based verifier that rejects unsupported sentences, holding unsupported-claim rate below **X%** at an **X% abstention rate**, plus **iterative multi-hop retrieval** raising HotpotQA supporting-passage recall from **X to X**.

## 11. Why This Reads as Real Company Work

- **The constraint is a customer requirement, not a budget limitation.** "Open weights because the data cannot leave the network" is a product decision with a buyer behind it, and it survives any interview follow-up.
- **Both guarantees are tested.** Zero egress and cross-platform support are CI jobs, not claims. Interviewers notice tests that encode product promises.
- **Cross-platform support is a shipping concern.** Customers run what they already own. Handling Windows and macOS from one repository — and knowing *why* Docker cannot hold the models on a Mac — is the kind of detail that only comes from having actually done it.
- **The trade-off studies are the job.** Choosing rerank depth and embedding dimension against a latency budget is daily applied-ML work, and almost no new-grad project shows it.
- **Model choices are defended with data**, including against the incumbent BGE family, rather than asserted.
- **Domain fine-tuning is a genuine differentiator.** Few candidates have trained a retrieval model on mined hard negatives with a forgetting-regression gate.

## 12. Interview Leverage

- **Search / IR system design**: hybrid recall, cross-encoder reranking, fusion, chunking strategy, index selection — with numbers behind every choice.
- **Efficiency engineering**: quantization, Matryoshka representation learning, ANN trade-offs, batching — the accuracy/latency/memory frontier, measured on two architectures.
- **Retrieval model training**: contrastive learning, hard-negative mining, LoRA, catastrophic-forgetting evaluation. The deepest ML content, and the rarest in a new-grad candidate.
- **LLM engineering**: grounding, verification, abstention policy, iterative retrieval, cost control with no API to lean on.
- **Platform / SDE**: containerized stateful services, native GPU inference, async ingestion jobs, index swap with no downtime, caching, monitoring, and a CI matrix that tests product guarantees.
- **Product judgement**: which market needs on-prem, and why a 0.6B model you can fine-tune beats a frontier model you cannot.
