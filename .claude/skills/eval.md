---
name: eval
description: Run retrieval evals (recall@5, MRR, abstention, same-evidence replay, per-agent breakdown) against the dev set — or the held-out set at a phase gate — and append results to project_memory/eval_log.md. Required after any change to ingest, chunking, embedding, index, fusion, graph expansion or reranking.
---

Apply this skill to measure retrieval quality with numbers, not impressions. The `metrics` skill measures
code quality; this skill measures whether retrieval got better or worse. Definitions come from
`.claude/project_memory/requirements.md` § "Metric definitions" — do not redefine them here or in code.

## When to Run
| Trigger | Set | Required |
|---------|-----|----------|
| Change to `rag_core/ingest*`, `store/`, `ranking/`, `okf/`, `retrieve.py` | dev | yes — before Code Reviewer |
| A/B of a ranking/contextualization variant | dev | yes — both variants, same index_version base |
| Phase exit gate | held-out | yes — once per gate, never for tuning |
| Adapter / MCP tool change (Phase 4+) | dev, replay mode | yes |
| Docs-only or test-only change | — | no |

**Held-out discipline:** never run the held-out set to choose between variants or tune weights. If a held-out
run happens outside a gate, record it in `eval_log.md` with `reason:` and treat the set as contaminated —
flag it to the user.

## Inputs (created in Phase 0)
- `evals/data/dev.jsonl`, `evals/data/heldout.jsonl` — one question per line:
  `{"qid", "question", "type": "single|multihop|unanswerable", "relevant": ["doc_id" | "concept_id", ...], "tenant_id"}`
- `evals/run_retrieval.py` — CLI that calls `RetrievalService.search` directly (not through an agent):
  `python -m evals.run_retrieval --set dev --k 5 --out evals/results/<run_id>.json`
- `evals/replay.py` — replays one recorded `SearchRequest` list through every adapter and the MCP server.

If any input is missing, stop: report exactly which file is missing and block the gate. Never estimate a metric.

## Step 1 — Retrieval Metrics
```bash
python -m evals.run_retrieval --set dev --k 5 --out evals/results/$(date +%Y%m%d-%H%M)-dev.json
```
Compute per question, then average (definitions from requirements.md):
- **Recall@5** — fraction of a question's relevant IDs found in the top 5; chunks are mapped to their `doc_id`
  and a document counts once. Answerable questions only.
- **MRR** — 1 / rank of the first relevant result (0 if none in top k). Answerable questions only.
- **Abstention accuracy** — unanswerable questions: correct when no result clears the abstention threshold
  (or the agent declines, in answer-grading runs). Never folded into recall.
- Report each metric **overall and per type** (single / multihop / unanswerable). Multi-hop is the Phase 3 gate.
- Report `index_version`, embedding model, ranking config hash, and `via="graph"` share of relevant hits.

## Step 2 — Latency (same run)
From the per-stage timings in each `SearchResponse` log record: p50 / p95 for embed, search, fuse, expand,
rerank and total. A p95 total ≥ 1.5 s is a MAJOR finding even if recall improved.

## Step 3 — Same-Evidence Replay (Phase 4+, or any adapter/MCP change)
```bash
python -m evals.replay --set dev --surfaces mcp,claude,openai,gemini
```
For each recorded request: the ordered `result_id` lists and `index_version` must be identical across surfaces.
Any difference is CRITICAL (violates SC-02 / NFR-04). Do not rely on live agents phrasing queries the same way.

## Step 4 — Answer Grading (phase gates and Phase 5 only)
Faithfulness (LLM-graded) and claim-level citation accuracy, per agent family. Before trusting the grader,
confirm it agrees with the 20-question human sample recorded in `evals/data/human_sample.jsonl`; report the
agreement rate alongside the score.

## Step 5 — Compare and Decide
Compare with the most recent `eval_log.md` entry on the same set:
- Recall@5 or MRR down by more than 0.02 (overall or any question type) → **REGRESSION**; MAJOR finding for
  Code Reviewer unless the change was meant to trade it (state the trade explicitly).
- Phase 2 / Phase 3 A/B: keep the variant only if it beats the previous phase on the dev set (Phase 3: on multi-hop).
- Gate run: compare against the targets table in requirements.md; list every missed target.

## Step 6 — Write to eval_log.md
Append to `.claude/project_memory/eval_log.md`:
```markdown
---
## Eval — [YYYY-MM-DD HH:MM] — [dev | held-out (GATE: Phase N)] — [change summary]
index_version: … | embedding: … | ranking_config: … | results: evals/results/<run_id>.json

| Metric | Overall | Single | Multi-hop | Unanswerable | Prev | Δ | Target |
|--------|---------|--------|-----------|--------------|------|---|--------|
| Recall@5 | | | | — | | | ≥0.85 |
| MRR | | | | — | | | ≥0.70 |
| Abstention acc. | — | — | — | | | | — |
| p95 latency (s) | | | | | | | <1.5 |
| Same-evidence mismatches | | | | | | | 0 |
| Faithfulness (gates) | | | | | | | ≥0.90 |
| Citation accuracy (gates) | | | | | | | ≥0.95 |

Verdict: IMPROVED | NO CHANGE | REGRESSION | GATE PASSED | GATE FAILED ([missed targets])
---
```

## Rules
- Numbers come from a run in this session — never copied, estimated or carried forward.
- Tune only on dev. Held-out runs only at gates.
- Eval data is versioned in Git; changing a question or its relevant IDs needs a note in `eval_log.md`.
- Mock nothing in eval runs: evals hit a real (local or dev) index. Unit tests are where mocks belong.
- Cross-tenant: every run includes at least one question asked under a different `tenant_id`; any result from
  another tenant is CRITICAL regardless of other metrics.
