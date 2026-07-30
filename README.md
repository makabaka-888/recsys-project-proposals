# ML Engineering Project Proposals

Design documents for 1:1 mentoring projects in **recommendation systems, search relevance, and retrieval**. Each doc is a self-contained engineering proposal: architecture, dataset, tech stack, evaluation plan, milestones, risks, and interview leverage.

> **Private repository.** These documents reference a mentee's employment history and job-search strategy. Do not make this repository public.

## Proposals

| # | Project | Domain | Headline technique |
|---|---|---|---|
| [01](01_FeedForge_Design_Doc.md) | **FeedForge** | News / short-video feed | Two-tower retrieval + DeepFM ranking + LLM re-ranking on MIND impression logs |
| [02](02_ShopBrain_Design_Doc.md) | **ShopBrain** | E-commerce | SASRec sequential recs + semantic-ID generative retrieval + LLM shopping assistant |
| [03](03_TuneFlow_Design_Doc.md) | **TuneFlow** | Music streaming | SASRec / BERT4Rec next-track prediction + LLM playlist generation |
| [04](04_SearchCraft_Design_Doc.md) | **SearchCraft** | E-commerce search | Hybrid BM25 + dense retrieval, LambdaMART, contract-first API design |
| [05](05_SnapShop_Design_Doc.md) | **SnapShop** | Visual product discovery | CLIP visual retrieval, FAISS IVF-PQ index benchmarking, LLM agent with critic loop |
| [06](06_AlphaDesk_Design_Doc.md) | **AlphaDesk** | Quantitative research | Quantized document retrieval, two-stage rerank, grounded research agent with backtest gates |
| [07](07_Enclave_Design_Doc.md) | **Enclave** | On-prem enterprise search | **Fully local, open-weight only**: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B, LoRA domain adaptation, egress-blocked CI test |

## Status

- **07 (Enclave)** is the current active proposal: zero-egress, self-hosted, open-weight only — no inference API anywhere in the stack.
- **05 (SnapShop)** and **06 (AlphaDesk)** are prior candidates built around image-retrieval and quantitative-research framings.
- **01–04** are earlier alternatives, retained for reference.

## Shared design principles

- **Public datasets with honest labeling.** Where impression logs do not exist, negative-sampling strategy and exposure bias are documented rather than glossed over.
- **Systems, not scripts.** Every proposal integrates the model into a service: microservices, caching, latency budgets, load tests, monitoring.
- **Laptop-scale by design.** Scoped datasets, small models, precomputed embeddings; no GPU required, LLM API budget under $20.
- **Metrics are deliverables.** Each sprint produces a measured number, benchmarked against a public baseline wherever one exists.
