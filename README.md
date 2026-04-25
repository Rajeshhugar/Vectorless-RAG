# Vectorless RAG — LangChain + OpenAI

Reasoning-based document retrieval using [PageIndex](https://pageindex.ai) + [LangChain](https://langchain.com) + [OpenAI](https://openai.com) — no vector database required.

## Overview

Instead of chunking documents and searching by approximate vector similarity, Vectorless RAG builds a **hierarchical tree index** over the document and uses an LLM to *reason* its way to the right section — like a human expert navigating a table of contents.

**Benchmark: 98.7% accuracy on FinanceBench**, outperforming vector-based RAG on complex financial questions.

## Stack

- **PageIndex** — hierarchical tree indexing + LLM-guided retrieval
- **LangChain** — chain orchestration, prompt templates, output parsers
- **OpenAI GPT-4o** — answer generation and vision analysis

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create .env file and set API keys

Create a `.env` file in the project root and add your API keys:

```bash
# .env
PAGEINDEX_API_KEY=your_pageindex_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Get your API keys:**
- **PageIndex API Key**: Sign up at [pageindex.ai](https://pageindex.ai) and generate an API key
- **OpenAI API Key**: Get your key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## Usage

### Step 1 — Index a PDF

```bash
python step1_submit_pdf.py --pdf ./docs/annual_report.pdf
```

### Step 2 — Inspect the document tree

```bash
python step2_inspect_tree.py --doc-id <doc_id>
```

### Step 3 — Ask a question (core pipeline)

```bash
python step3_retrieve_generate.py --doc-id <doc_id> --query "What was  revenue for 2025?"
```

### Step 4 — Query across multiple documents

```bash
python step4_multi_doc.py --doc-ids pi-doc-2023 pi-doc-2024 --query "How did margins change YoY?"
```

### Step 5 — Vision RAG (charts and scanned pages)

```bash
python step5_vision_rag.py --doc-id <doc_id> --pdf ./docs/annual_report.pdf --query "What does the revenue chart show?"
```

## Project structure

```
vectorless-rag-langchain/
├── README.md
├── requirements.txt
├── .env
├── step1_submit_pdf.py
├── step2_inspect_tree.py
├── step3_retrieve_generate.py
├── step4_multi_doc.py
└── step5_vision_rag.py
```

## When to use Vectorless RAG

**Use it when:**
- Precision matters (financial, legal, medical documents)
- Documents have natural structure (reports, contracts, manuals)
- Your corpus changes frequently (re-index in seconds)
- You need explainability (page citations, not cosine scores)
- Documents contain charts, tables, or scanned pages

**Stick with vector RAG when:**
- Semantic paraphrase matching is the core challenge
- You need sub-100ms retrieval at massive scale
- Documents have no structure (transcripts, chat logs)
