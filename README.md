# Ai-contract-review

Temporal-based AI contract review pipeline: extracts PDFs from S3, summarizes
each contract with an LLM, synthesizes a cross-contract risk report, and pauses
for human approval/feedback before completing.

## Architecture

```
client-app (FastAPI)  ──starts──►  ContractReviewWorkflow (parent)
                                        │
                                        ├─► PDFSummaryWorkflow (child, one per PDF)
                                        │       ├─ extract_pdf  (S3 + PyMuPDF)
                                        │       └─ call_llm     (OpenRouter)
                                        │
                                        ├─ call_llm            (synthesis)
                                        ├─ wait for human review (signal/update)
                                        └─ call_llm            (revision, if requested)
```

- `src/worker.py` — Temporal worker (workflows + activities).
- `src/activities.py` — PDF extraction, LLM calls, LangSmith feedback.
- `src/observability.py` — LangSmith tracing helpers (optional, env-gated).
- `client-app/main.py` — FastAPI endpoints to start/monitor/review a workflow.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in the values
```

## Running

```bash
# terminal 1 — worker
cd src && uv run python worker.py

# terminal 2 — API
cd client-app && uv run uvicorn main:app --port 5000
```

## Observability (LangSmith)

Every LLM call, PDF extraction, and human review decision can be traced to
[LangSmith](https://smith.langchain.com). Tracing is **opt-in** and only
activates when `LANGSMITH_TRACING=true`; without it the app behaves exactly the
same and makes no LangSmith requests.

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ai-contract-review
```

What you get:

| Run | Type | What is captured |
| --- | --- | --- |
| `extract_pdf` | tool | S3 path, page count, chars extracted (contract text is **not** uploaded) |
| `call_llm` | llm | Sanitized prompt (truncated at 20k chars), response, tokens, latency, OpenRouter model |
| `record_review_feedback` | — | Approve → score `1.0`, revise → score `0.0`, plus the reviewer's comment |

Useful metadata on every run:

- `workflow_id` — the review workflow execution id
- `thread_id` — the parent review workflow id, so all PDFs/prompts of one
  review group together as a single thread
- `stage` — `summary` | `synthesis` | `revision`
- `s3_path`, `revision_no`, `contracts`

To follow one review in the UI, filter by
`metadata.workflow_id = "contract-review-<uuid>"` (the id returned by
`POST /contract-review/start`), or open the thread for that id. The report run
id is also exposed by the `get_status` query (`report_run_id`) and receives the
human-review feedback entry.

Notes:

- Traces are buffered and flushed on worker shutdown; a hard kill (`SIGKILL`)
  can drop the last few seconds of runs.
- OpenRouter model names may be absent from LangSmith's price table, so cost
  can show as `$0` even though token counts are correct.
- Tracing lives in activities only. Workflow code must stay deterministic, so
  it passes plain metadata dicts (`trace_metadata`) rather than importing
  LangSmith.
