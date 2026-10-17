import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from activities import extract_pdf, call_llm
from child_workflow import PDFSummaryWorkflow
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
        activities=[extract_pdf, call_llm],
    )

    print(f"Worker running on: '{TEMPORAL_TASK_QUEUE}'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())