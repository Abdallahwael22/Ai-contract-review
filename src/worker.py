import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from activities import extract_pdf, call_llm, record_review_feedback
from child_workflow import PDFSummaryWorkflow
from observability import flush_traces
from parent_workflow import ContractReviewWorkflow

load_dotenv()

TEMPORAL_HOST       = os.environ["TEMPORAL_HOST"]
TEMPORAL_NAMESPACE  = os.environ["TEMPORAL_NAMESPACE"]
TEMPORAL_TASK_QUEUE = os.environ["TEMPORAL_CONTRACT_REVIEW_TASK_QUEUE"]

async def main():

    temporal_client = await Client.connect(TEMPORAL_HOST, 
                                           namespace=TEMPORAL_NAMESPACE)
    
    worker = Worker(
        temporal_client,
        task_queue=TEMPORAL_TASK_QUEUE,
        workflows=[ContractReviewWorkflow, PDFSummaryWorkflow],
        activities=[extract_pdf, call_llm, record_review_feedback],
    )

    print(f"Worker running on: '{TEMPORAL_TASK_QUEUE}'")
    try:
        await worker.run()
    finally:
        # Push any buffered LangSmith runs before the process exits.
        flush_traces()


if __name__ == "__main__":
    asyncio.run(main())
