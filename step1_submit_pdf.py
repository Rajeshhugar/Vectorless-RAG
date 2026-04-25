"""
Step 1 — Submit a PDF to PageIndex and wait for the tree index to be built.

Usage:
    python step1_submit_pdf.py --pdf ./annual_report.pdf
"""

import time
import argparse

from config import get_pageindex_client

def submit_and_wait(pdf_path: str) -> str:
    """Submit PDF and poll until the tree index is ready."""
    pi_client = get_pageindex_client()

    result = pi_client.submit_document(pdf_path)
    doc_id = result["doc_id"]
    print(f"Submitted: {doc_id}")
    print("Waiting for indexing to complete...")

    while True:
        status = pi_client.get_document(doc_id)["status"]
        if status == "completed":
            print(f"Ready: {doc_id}")
            return doc_id
        if status == "failed":
            raise RuntimeError("Processing failed")
        print("  still processing...")
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    args = parser.parse_args()

    doc_id = submit_and_wait(args.pdf)
    print(f"\nSave this doc_id for future queries:\n  {doc_id}")
