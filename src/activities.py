import os
import math
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import boto3
import fitz                  # PyMuPDF — page-by-page extraction
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from temporalio import activity

from observability import (
    annotate_run,
    current_run_id,
    record_feedback,
    traced,
    wrap_openai_client,
)

load_dotenv()

# Default LangSmith project when LANGSMITH_PROJECT is not set in .env.
os.environ.setdefault("LANGSMITH_PROJECT", "ai-contract-review")

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ExtractPDFInput:
    s3_path: str
    batch_size: int = 2
    trace_metadata: Optional[dict] = None

@dataclass
class ExtractPDFOutput:
    s3_path: str
    markdown_text: str
    page_count: int

@dataclass
class CallLLMInput:
    prompt: str
    trace_metadata: Optional[dict] = None

@dataclass
class CallLLMOutput:
    content: str
    run_id: Optional[str] = None

@dataclass
class ReviewFeedbackInput:
    run_id: Optional[str]
    decision: str
    feedback: str = ""
    workflow_id: str = ""

# ── S3 helper ────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
        endpoint_url=os.environ["AWS_S3_ENDPOINT_URL"],
    )

def parse_s3_path(s3_path: str):
    s3_path_no_scheme = s3_path.replace("s3://", "")
    bucket, _, key =  s3_path_no_scheme.partition("/")
    return bucket, key

# ── Activity 1: Extract PDF from S3 ──────────────────────────────────────────

async def _extract_pdf(params: ExtractPDFInput) -> ExtractPDFOutput:

    annotate_run(params.trace_metadata)

    activity.logger.info(f"Starting extraction: {params.s3_path}")

    activity.heartbeat({
        "stage": "downloading",
        "s3_path": params.s3_path,
        "pages_done": 0,
        "chars_extracted": 0,
    })

    s3_client = get_s3_client()
    bucket, key = parse_s3_path(params.s3_path)

    filename = Path(key).name
    TEMP_DIR = os.environ["TEMP_DIR"]

    local_path = str(Path(TEMP_DIR) / filename)

    s3_client.download_file(
        bucket,
        key,
        local_path
    )

    doc = fitz.open(local_path)
    total_pages = doc.page_count

    activity.logger.info(f"Downloaded {total_pages}-page PDF: {params.s3_path}")

    all_text_chunks = []
    total_chars_num = 0
    num_batches = math.ceil(total_pages / params.batch_size)

    for batch_idx in range(num_batches):

        start_page = batch_idx * params.batch_size
        end_page =  min(start_page + params.batch_size, total_pages)

        batch_md = pymupdf4llm.to_markdown(
            local_path,
            pages=list(range(start_page, end_page)),
        )

        all_text_chunks.append(batch_md)
        total_chars_num += len(batch_md)

        activity.heartbeat({
            "stage":           "extracting",
            "s3_path":         params.s3_path,
            "pages_done":      end_page,
            "total_pages":     total_pages,
            "batch":           f"{start_page + 1}–{end_page}",
            "chars_extracted": total_chars_num,
            "progress_pct":    round(end_page / total_pages * 100),
        })


    full_md = "\n\n".join(all_text_chunks)

    activity.heartbeat({
        "stage": "done",
        "s3_path": params.s3_path,
        "pages_done": total_pages,
        "chars_extracted": total_chars_num,
    })

    return ExtractPDFOutput(
        s3_path=params.s3_path,
        markdown_text=full_md,
        page_count=total_pages,
    )


# NOTE: `traced` is applied to the implementation, not to `@activity.defn`.
# Temporal rejects the *args/**kwargs signature LangSmith's wrapper produces.
_traced_extract_pdf = traced(
    name="extract_pdf",
    run_type="tool",
    # Never ship the extracted contract text to LangSmith.
    drop_inputs=("trace_metadata",),
    drop_outputs=("markdown_text",),
)(_extract_pdf)

@activity.defn
async def extract_pdf(params: ExtractPDFInput) -> ExtractPDFOutput:
    return await _traced_extract_pdf(params)


# ── Activity 2: Call the LLM via OpenRouter ───────────────────────────────────

async def _call_llm(params: CallLLMInput) -> CallLLMOutput:

    annotate_run(params.trace_metadata)

    activity.logger.info("Calling LLM")
    activity.heartbeat({"stage": "calling_llm",
                         "prompt_chars": len(params.prompt)})


    llm_client = wrap_openai_client(
        OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    )

    response = llm_client.chat.completions.create(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[{"role": "user", "content": params.prompt}],
        max_tokens=8000,
    )

    content = response.choices[0].message.content

    activity.logger.info(f"LLM returned {len(content)} chars")

    return CallLLMOutput(
        content=content,
        # Reference for HITL feedback attached later by the parent workflow.
        run_id=current_run_id(),
    )


_traced_call_llm = traced(
    name="call_llm",
    run_type="llm",
    drop_inputs=("trace_metadata",),
)(_call_llm)

@activity.defn
async def call_llm(params: CallLLMInput) -> CallLLMOutput:
    return await _traced_call_llm(params)


# ── Activity 3: Record the human review decision in LangSmith ────────────────

@activity.defn
async def record_review_feedback(params: ReviewFeedbackInput) -> bool:
    """Best-effort: log approve/revise as feedback on the reviewed report run."""
    activity.logger.info(
        f"Recording review feedback (decision={params.decision}, "
        f"run_id={params.run_id})"
    )

    return record_feedback(
        params.run_id,
        decision=params.decision,
        feedback=params.feedback,
        metadata={"workflow_id": params.workflow_id},
    )
