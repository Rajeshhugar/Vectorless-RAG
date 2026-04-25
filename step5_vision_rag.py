"""
Step 5 — Vision RAG using LangChain + OpenAI GPT-4o vision.

PageIndex identifies relevant pages via tree reasoning.
PyMuPDF renders them to base64 PNG images.
A LangChain chain with GPT-4o vision analyses charts, tables, and scanned pages.

Usage:
    python step5_vision_rag.py --doc-id <doc_id> --pdf ./annual_report.pdf \
        --query "What does the revenue breakdown chart show for FY2024?"
"""

import re
import base64
import argparse
import fitz  # PyMuPDF

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from config import get_openai_api_key, get_pageindex_client


def find_relevant_pages(doc_id: str, query: str) -> list[int]:
    """Use the PageIndex tree to identify which page numbers are relevant."""
    pi_client = get_pageindex_client()
    tree = pi_client.get_tree(doc_id)["result"]

    def all_pages(nodes: list) -> list[int]:
        nums = []
        for n in nodes:
            if "page_index" in n:
                nums.append(n["page_index"])
            if n.get("nodes"):
                nums.extend(all_pages(n["nodes"]))
        return nums

    resp = pi_client.chat_completions(
        messages=[{
            "role": "user",
            "content": f"Which pages contain content relevant to: {query}\nReturn page numbers only."
        }],
        doc_id=doc_id
    )
    text = resp["choices"][0]["message"]["content"]

    # Filter returned numbers against valid pages in the tree
    pool = set(all_pages(tree))
    found = [int(n) for n in re.findall(r"\b(\d+)\b", text) if int(n) in pool]

    # Fall back to first 3 pages if nothing found
    return sorted(set(found))[:5] or sorted(pool)[:3]


def render_pages(pdf_path: str, pages: list[int]) -> list[dict]:
    """
    Render PDF pages to base64-encoded PNG images at 150 DPI.

    Note: PageIndex page_index is 1-based; PyMuPDF load_page() is 0-based.
    The `pg - 1` conversion handles this.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    result = []

    for pg in pages:
        pix = doc.load_page(pg - 1).get_pixmap(matrix=mat, alpha=False)
        result.append({
            "page": pg,
            "b64": base64.standard_b64encode(pix.tobytes("png")).decode()
        })

    doc.close()
    return result


def build_vision_message(query: str, page_images: list[dict]) -> list:
    """
    Build a LangChain HumanMessage with interleaved text and image content
    for GPT-4o vision.
    """
    content = []

    for img in page_images:
        content.append({
            "type": "text",
            "text": f"\n### Page {img['page']}\n"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['b64']}",
                "detail": "high"
            }
        })

    content.append({"type": "text", "text": f"\n## Question\n\n{query}"})
    return content


def vision_answer(query: str, page_images: list[dict]) -> str:
    """Send rendered page images to GPT-4o vision via LangChain."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=2048,
        api_key=get_openai_api_key()
    )

    system = SystemMessage(content=(
        "Read all charts, tables, and diagrams carefully. "
        "Answer precisely with specific values. Cite page numbers."
    ))
    human = HumanMessage(content=build_vision_message(query, page_images))

    response = llm.invoke([system, human])
    return StrOutputParser().invoke(response)


def run_vision_pipeline(doc_id: str, pdf_path: str, query: str) -> str:
    """Full Vision RAG pipeline."""
    print(f"Query: {query}\n")
    print("Finding relevant pages via PageIndex tree...")
    pages = find_relevant_pages(doc_id, query)
    print(f"Relevant pages: {pages}")

    print("Rendering pages to images...")
    images = render_pages(pdf_path, pages)

    print("Sending to GPT-4o vision via LangChain...\n")
    return vision_answer(query, images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True, help="PageIndex document ID")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF file")
    parser.add_argument("--query", required=True, help="Question about charts or visuals")
    args = parser.parse_args()

    answer = run_vision_pipeline(args.doc_id, args.pdf, args.query)
    print("Answer:\n" + "-" * 40)
    print(answer)
