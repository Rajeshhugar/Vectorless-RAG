"""
Step 4 — Multi-document cross-synthesis using LangChain + OpenAI.

PageIndex retrieves tagged sections from all documents simultaneously.
A LangChain chain with GPT-4o synthesises a comparative answer.

Usage:
    python step4_multi_doc.py --doc-ids pi-doc-2023 pi-doc-2024 \
        --query "How did operating margins change year over year?"
"""

import os
import argparse
from dotenv import load_dotenv
from pageindex import PageIndexClient

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pi_client = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])


def retrieve_multi_context(doc_ids: list[str], query: str) -> str:
    """Retrieve tagged sections from multiple documents via PageIndex."""
    retrieval = pi_client.chat_completions(
        messages=[{
            "role": "user",
            "content": f"Find and tag sections from ALL documents relevant to: {query}"
        }],
        doc_id=doc_ids  # list, not string
    )
    return retrieval["choices"][0]["message"]["content"]


def build_multi_doc_chain() -> object:
    """Build a LangChain chain for cross-document analysis with GPT-4o."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a cross-document analyst. "
            "Compare information across the provided sources. "
            "Note agreements and contradictions. "
            "Always cite document names and page numbers."
        ),
        (
            "human",
            "## Multi-document context\n\n{context}\n\n## Question\n\n{query}"
        )
    ])

    return prompt | llm | StrOutputParser()


def multi_doc_query(doc_ids: list[str], query: str) -> str:
    """Full multi-document retrieve + synthesise pipeline."""
    print(f"Retrieving from {len(doc_ids)} documents: {doc_ids}")
    context = retrieve_multi_context(doc_ids, query)

    print("Generating cross-document answer with GPT-4o via LangChain...")
    chain = build_multi_doc_chain()
    return chain.invoke({"context": context, "query": query})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-ids", nargs="+", required=True, help="Two or more PageIndex document IDs")
    parser.add_argument("--query", required=True, help="Question to answer across documents")
    args = parser.parse_args()

    if len(args.doc_ids) < 2:
        print("Please provide at least two doc IDs for multi-document synthesis.")
        exit(1)

    answer = multi_doc_query(args.doc_ids, args.query)
    print("\nAnswer:\n" + "-" * 40)
    print(answer)
