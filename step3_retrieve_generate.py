"""
Step 3 — Core RAG pipeline using LangChain + OpenAI.

PageIndex performs LLM-guided tree search to retrieve relevant sections.
LangChain's ChatOpenAI and PromptTemplate handle answer generation.

Usage:
    python step3_retrieve_generate.py --doc-id <doc_id> --query "What was Q3 net revenue?"
"""

import argparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_openai_api_key, get_pageindex_client


def retrieve_context(doc_id: str, query: str) -> str:
    """PageIndex tree search — returns relevant sections with page references."""
    pi_client = get_pageindex_client()
    response = pi_client.chat_completions(
        messages=[{
            "role": "user",
            "content": (
                f"Find sections relevant to: {query}\n"
                f"Return only the retrieved text with page references."
            )
        }],
        doc_id=doc_id
    )
    return response["choices"][0]["message"]["content"]


def build_rag_chain() -> object:
    """Build a LangChain RAG chain with GPT-4o."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=get_openai_api_key()
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise document analyst. "
            "Answer using ONLY the provided context. "
            "Always cite page numbers in your answer."
        ),
        (
            "human",
            "## Context\n\n{context}\n\n## Question\n\n{query}"
        )
    ])

    return prompt | llm | StrOutputParser()


def run_pipeline(doc_id: str, query: str) -> str:
    """Full retrieve + generate pipeline."""
    print(f"Query: {query}\n")
    print("Retrieving context from PageIndex...")
    context = retrieve_context(doc_id, query)

    print("Generating answer with GPT-4o via LangChain...\n")
    chain = build_rag_chain()
    return chain.invoke({"context": context, "query": query})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True, help="PageIndex document ID")
    parser.add_argument("--query", required=True, help="Question to answer")
    args = parser.parse_args()

    answer = run_pipeline(args.doc_id, args.query)
    print("Answer:\n" + "-" * 40)
    print(answer)
