# DataByte Task 2 — Build a Healthcare RAG Application

## Overview

This repository contains a working **Retrieval-Augmented Generation (RAG) engine**, not a finished application.

The supplied code handles the AI-specific pipeline:

- extracting text from healthcare documents;
- splitting the text into chunks;
- generating embeddings;
- storing and retrieving chunks with ChromaDB;
- generating grounded answers through a locally hosted Ollama model.

Your task is to design and build the product around that pipeline. You must implement:

- the API contract and FastAPI application;
- the rest of the application backend;
- the complete frontend;
- validation and error handling;
- loading, empty, success, and failure states;
- documentation for your engineering decisions.

Endpoint paths, HTTP methods, request schemas, response schemas, project structure, frontend technology, and persistence strategy are intentionally not provided. These decisions are part of the task.

> This is an educational project, not a medical device. Do not present generated responses as professional medical advice.

---

## Product goal

Build a healthcare document assistant in which a user can:

1. upload a PDF, DOCX, or TXT document;
2. see which documents have been indexed;
3. ask questions about the uploaded material;
4. receive an answer grounded in retrieved document chunks;
5. inspect the sources returned with the answer;
6. understand when the application is loading, unavailable, or has failed;
7. use the application comfortably on desktop and mobile.

The final result should behave like a coherent application, not a page containing disconnected forms or API buttons.

---

## What is provided

### 1. Document ingestion

`backend/rag/ingestion.py` contains the document-processing pipeline. It provides the functionality required to:

- parse PDF, DOCX, and TXT documents;
- clean extracted text;
- divide text into overlapping chunks;
- generate embeddings using Sentence Transformers;
- store chunks and metadata in ChromaDB;
- inspect collection statistics.

The main functions you will build around are:

```python
ingest_document(file_path, filename)
get_collection_stats()
```

**Why this is provided:** implementing document parsing, chunking, embeddings, and vector storage from scratch would turn the assignment into an ML infrastructure task. The goal here is to test whether you can build a reliable application around an existing AI subsystem.

### 2. Semantic retrieval

`backend/rag/retriever.py` embeds a question and retrieves the closest stored chunks from ChromaDB.

```python
retrieve(query, top_k)
```

**Why this is provided:** semantic search is the core RAG primitive. You are expected to integrate and expose it correctly, not spend the entire task tuning a vector-search implementation.

### 3. Answer generation

`backend/rag/generator.py` formats retrieved chunks as context and sends them to the configured Ollama model.

```python
generate_answer(question, retrieved_chunks)
check_ollama_health()
```

**Why this is provided:** every participant can run the same local baseline without needing a paid model API. Submissions can therefore be evaluated mainly on application engineering and product quality.

### 4. Configuration

`backend/config.py` contains the model names, ChromaDB path, chunking settings, retrieval count, and system prompt.

**Why this is provided:** the supplied RAG modules need one consistent configuration source. You may read these values from your application, but avoid unnecessarily rewriting the RAG internals.

### 5. Sample documents and seed utility

`sample_docs/` contains small healthcare documents for development and testing.

`scripts/seed_sample_docs.py` can index those files before you finish the document-upload workflow.

```bash
python scripts/seed_sample_docs.py
```

**Why this is provided:** it lets you test retrieval and answer rendering independently from file upload. It is optional and is not part of the final product flow.

---

## What you must build

## Task 1 — Design the API

Create the FastAPI application in `backend/main.py` and decide which endpoints your product requires.

At minimum, your API must support these capabilities:

- checking whether the service and its dependencies are usable;
- uploading and ingesting a supported document;
- viewing indexed-document information;
- submitting a question;
- returning a generated answer together with useful source information.

You must decide and document:

- endpoint paths;
- HTTP methods;
- request formats;
- response formats;
- file-upload handling;
- validation rules;
- status codes;
- error-response structure;
- how source chunks are represented;
- how the frontend determines whether an operation succeeded.

Do not merely recreate endpoint code from another project. Your contract should be consistent and justified by the product you build.

### Why this task exists

The frontend and RAG pipeline need a clear integration boundary. API design tests whether you can translate product actions into HTTP operations, validate untrusted input, communicate failures properly, and separate client responsibilities from server responsibilities.

---

## Task 2 — Build the application backend

Build the backend around the supplied RAG modules. Keep HTTP and product logic separate from the RAG implementation.

Your backend should handle:

- FastAPI application setup;
- routing and schemas;
- configuration and CORS for your frontend setup;
- file-type, filename, and empty-file validation;
- temporary-file creation and cleanup;
- calls into ingestion, retrieval, and generation;
- conversion of internal results into stable API responses;
- useful error handling when parsing, retrieval, ChromaDB, or Ollama fails;
- any application state required by your product.

You may add document metadata, conversations, users, authentication, a relational database, or other services when they support your design. These are optional extensions unless your team is instructed otherwise.

### Why this task exists

The supplied RAG modules solve only the AI-specific portion of the system. A usable backend must still coordinate files, state, validation, external dependencies, and failures. This task tests whether you can build around an existing subsystem without coupling every responsibility together.

---

## Task 3 — Build the complete frontend

No frontend implementation is supplied. You may use vanilla HTML/CSS/JavaScript or a framework of your choice.

The interface must include:

- document selection and upload;
- clear supported-file guidance;
- indexed-document or collection information;
- a question input and submission flow;
- generated-answer rendering;
- readable source rendering;
- loading and disabled states;
- useful empty, success, and error states;
- a responsive desktop and mobile layout.

### Why this task exists

A feature is not complete merely because its endpoint works. Users need feedback, understandable states, and a clear journey through the application. This task tests API integration, asynchronous state management, interface design, accessibility, and error communication.

---

## Task 4 — Handle failure cases deliberately

Your application should behave sensibly when:

- the backend is unavailable;
- Ollama is not running;
- the configured model has not been pulled;
- a file is unsupported or empty;
- document extraction fails;
- no documents have been indexed;
- the question is empty;
- retrieval returns no useful context;
- ingestion or generation takes noticeable time;
- the server returns an unexpected error.

Do not show infrastructure errors as successful-looking AI answers.

### Why this task exists

AI and document-processing operations are slower and more failure-prone than ordinary in-memory operations. Explicit error handling makes the product understandable and prevents users from confusing a broken dependency with a valid response.

---

## Task 5 — Document your decisions

Update this README or add project documentation explaining:

- your architecture;
- your API contract;
- your frontend and backend technologies;
- setup and run commands;
- important design decisions and trade-offs;
- assumptions and limitations;
- known issues;
- optional features you implemented.

### Why this task exists

Engineering work must be understandable by people who did not write it. Good documentation makes the implementation reproducible and demonstrates that architectural choices were intentional.

---

## Expected system boundary

A typical request flow is:

```text
Browser / frontend
        |
        |  API contract designed by you
        v
Application backend
        |
        |  Python function calls
        v
Provided RAG pipeline
        |
        +--> document parsing and chunking
        +--> Sentence Transformer embeddings
        +--> ChromaDB storage and retrieval
        +--> Ollama answer generation
```

This is a responsibility boundary, not a required folder structure. You may organize your project differently as long as the design remains understandable.

---

## Starter repository structure

```text
databyte-task-2/
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py              # Placeholder: design and build the FastAPI app
│   ├── config.py            # Supplied RAG configuration
│   └── rag/
│       ├── __init__.py
│       ├── ingestion.py     # Supplied parsing, chunking, embedding, storage
│       ├── retriever.py     # Supplied semantic retrieval
│       └── generator.py     # Supplied grounded generation
├── sample_docs/             # Sample development documents
└── scripts/
    └── seed_sample_docs.py  # Optional sample-data loader
```

Create any frontend, backend modules, schemas, database code, or tests that your design requires.

---

## Local setup

### Requirements

- Python 3.10 or newer
- Ollama
- at least 8 GB of RAM; larger models may require more

### Install dependencies

```bash
git clone https://github.com/scienstien/databyte-task-2.git
cd databyte-task-2
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

### Prepare Ollama

```bash
ollama pull mistral
ollama serve
```

The default model can be changed in `backend/config.py`.

### Optional: load sample documents

```bash
python scripts/seed_sample_docs.py
```

### Run your application

`backend/main.py` intentionally contains no working FastAPI application or endpoints. Implement them first, then document the command used to start your backend.

A common command after creating an `app` object is:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

---

## Minimum completion criteria

A submission is complete when:

- the project can be set up from its documentation;
- the participant designed and documented an API contract;
- a supported document can be uploaded and indexed through that API;
- indexed-document information is visible in the interface;
- a question can be submitted from the frontend;
- the answer and its sources are displayed clearly;
- loading, empty, and error states are visible;
- temporary files and failed requests are handled safely;
- the layout works on desktop and mobile;
- model answers and document results are not hard-coded.

---

## Evaluation focus

Submissions should be evaluated primarily on:

1. correctness of the end-to-end workflow;
2. quality and consistency of the API design;
3. backend structure and error handling;
4. frontend usability and responsiveness;
5. integration with the supplied RAG modules;
6. code clarity and documentation;
7. handling of edge cases.

Visual polish is valuable, but a stable and understandable product is more important than decorative complexity.

---

## Optional extensions

After completing the required workflow, you may explore:

- authentication and per-user collections;
- persistent chat history;
- document deletion and re-indexing;
- duplicate-document detection;
- streaming answers;
- configurable retrieval settings;
- improved citation metadata such as page numbers;
- automated tests;
- Docker-based setup;
- deployment.

Optional extensions should not replace the required end-to-end application.

---

## Restrictions

- Do not hard-code model answers.
- Do not bypass the supplied RAG pipeline with a separate hosted chatbot API.
- Do not present the application as a source of medical diagnosis or professional advice.
- Avoid modifying the supplied RAG internals unless you are fixing a clearly documented issue or implementing an optional extension.

The purpose of the task is to design and build a reliable application around an existing AI subsystem.
