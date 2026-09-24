# Support Assistant

## Overview

This module implements a policy-based support assistant using retrieval, embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

The assistant uses a collection of Zepto-related policy text and retrieves relevant information when a user asks a policy question.

The implementation also contains a mock LLM mode for deterministic execution.

---

# Architecture

The current implementation follows this general flow:

```text
Policy Documents
       ↓
Text Processing
       ↓
Sentence Transformer Embeddings
       ↓
ChromaDB
       ↓
User Query
       ↓
Intent Classification
       ↓
 ┌──────────────────────┐
 │                      │
Policy Question    General Question
 │                      │
 ↓                      ↓
Retrieval           Direct Answer
 │
 ↓
Retrieved Context
 │
 ↓
Answer Generation
 │
 ↓
Pydantic Response
 │
 ↓
FastAPI
```

---

# Technologies

The implementation uses:

* Python
* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* Pydantic
* FastAPI
* Uvicorn

---

# Policy Corpus

The current implementation stores the policy information in:

```text
Documents.txt
```

The file contains the policy content used by the assistant.

The policies cover topics including:

* Delivery
* Returns and refunds
* Membership
* Order tracking
* Order cancellation
* Damaged or missing items
* Gift cards
* Customer support

---

# Embeddings

The application uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The policy text is converted into vector embeddings.

These embeddings are then stored in a ChromaDB collection.

---

# ChromaDB

ChromaDB is used as the vector database.

The collection stores the document text together with the corresponding embeddings and metadata.

When a user submits a policy-related query, the query is embedded and similarity search is performed against the stored documents.

The implementation retrieves the most relevant documents/chunks for the query.

---

# Prompt Structure

The implementation contains a structured prompt for answer generation.

The prompt includes sections corresponding to:

```text
ROLE
CONTEXT
TASK
FORMAT
LENGTH
```

The prompt also contains:

* A negative constraint instructing the model to stay grounded in the supplied context.
* A few-shot example demonstrating the expected response behavior.

---

# LangGraph

The application uses LangGraph to represent the assistant workflow.

The graph contains three main processing nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The intent classification determines which path the query follows.

The workflow is approximately:

```text
                  classify_intent
                  /             \
                 /               \
      policy_question       general_question
              ↓                     ↓
    retrieve_and_answer       direct_answer
```

---

# Intent Classification

The current implementation includes keyword-based intent classification.

Policy-related keywords include terms such as:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Queries containing these types of terms are routed toward the policy retrieval path.

Other queries are routed to the general-question path.

---

# Mock LLM

The implementation supports a `MOCK_LLM` configuration.

When mock mode is enabled, the assistant uses deterministic keyword-based routing instead of relying on an external LLM for the basic intent classification flow.

The mock implementation provides a deterministic response path for general questions and uses retrieved policy context for policy-related questions.

This allows the application to be demonstrated without requiring an external LLM API.

---

# Pydantic Response

The FastAPI response is represented using a Pydantic model.

The response contains:

```text
answer
sources
confidence
```

The structured response is returned as JSON.

Example structure:

```json
{
  "answer": "...",
  "sources": [],
  "confidence": 1.0
}
```

---

# FastAPI

The application exposes a POST endpoint:

```text
POST /ask
```

The request accepts a JSON object containing a query.

Example:

```json
{
  "query": "How much is delivery?"
}
```

The application processes the query through the LangGraph workflow and returns a structured JSON response.

---

# Running the Application

The Python application can be started using Uvicorn.

From the `support-assistant` directory:

```bash
uvicorn support_assistant:app --host 0.0.0.0 --port 7860
```

The service listens on port:

```text
7860
```

The main endpoint is:

```text
POST /ask
```

---

# Example Policy Question

Example request:

```json
{
  "query": "How much is delivery for an order below INR 149?"
}
```

The query is identified as a policy-related question and routed through the retrieval path.

The retrieved policy context is then used to construct the response.

---

# Example General Question

Example request:

```json
{
  "query": "What is the capital of India?"
}
```

This query does not match the configured policy keywords and is therefore routed through the general-question path.

In mock mode, the application returns the configured general-question response.

---

# Current Implementation Files

The current module contains the support-assistant implementation and its policy corpus.

The main components are:

```text
support-assistant/
│
├── support_assistant.py
└── Documents.txt
```

The notebook version of the implementation is also included in the project submission.

---

# Processing Summary

The current support assistant operates through the following sequence:

```text
1. Load policy text
       ↓
2. Create embeddings using all-MiniLM-L6-v2
       ↓
3. Store embeddings in ChromaDB
       ↓
4. Receive user query
       ↓
5. Classify query intent
       ↓
6. Retrieve relevant context for policy questions
       ↓
7. Generate the response
       ↓
8. Validate/structure the response using Pydantic
       ↓
9. Return the result through FastAPI
```

The implementation is designed to provide policy-grounded responses while keeping the default demonstration independent of an external LLM API.
