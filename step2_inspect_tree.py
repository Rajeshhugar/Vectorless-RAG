"""
Step 2 — Fetch and print the hierarchical tree index built by PageIndex.

Usage:
    python step2_inspect_tree.py --doc-id <doc_id>
"""

import argparse

from config import get_pageindex_client


def format_pages(page_index: object) -> str:
    """Format a page index value into a readable page label."""
    if isinstance(page_index, list):
        return ", ".join(str(page) for page in page_index)
    return str(page_index)


def print_tree(nodes: list, depth: int = 0) -> None:
    """Recursively print the document tree with indentation."""
    for node in nodes:
        indent = "  " * depth
        pages = f"pp.{format_pages(node['page_index'])}"
        print(f"{indent}• {node['title']} ({pages})")
        if node.get("nodes"):
            print_tree(node["nodes"], depth + 1)


def inspect_tree(doc_id: str) -> list:
    """Fetch and display the document tree."""
    pi_client = get_pageindex_client()

    tree = pi_client.get_tree(doc_id)["result"]
    print(f"\nDocument tree for: {doc_id}\n" + "-" * 40)
    print_tree(tree)
    return tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True, help="PageIndex document ID")
    args = parser.parse_args()

    inspect_tree(args.doc_id)
