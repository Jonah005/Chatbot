# RAG Question Answering System

A customer-facing question-answering system that uses **semantic vector search with Qdrant** and an **LLM** to generate answers grounded in approved business knowledge.

The system is designed for situations where customers ask questions in natural language, while the business already has a curated set of questions and answers prepared by its sales or support team.

## How It Works

The business knowledge base is prepared in advance as question-and-answer pairs and stored as vector embeddings in **Qdrant**.

When a customer asks a question:

1. The question is converted into a vector representation.
2. Qdrant performs semantic similarity search.
3. The system retrieves the **Top-K most relevant question-and-answer pairs**.
4. The retrieved context is passed to the LLM.
5. The LLM uses that context to generate a natural-language response for the customer.

```text
Customer Question
       ↓
Vector Embedding
       ↓
Qdrant Semantic Search
       ↓
Top-K Relevant Q&A Results
       ↓
LLM + Retrieved Context
       ↓
Grounded Customer Answer
```

## Why This Approach

Traditional keyword search can fail when a customer phrases a question differently from the wording stored in the knowledge base.

Semantic vector search allows the system to retrieve information based on **meaning**, not only exact words.

For example, questions such as:

> "When will my order arrive?"

and:

> "How long does delivery usually take?"

can still retrieve similar knowledge if they are semantically related.

The LLM then turns the retrieved information into a more natural response.

## Key Features

- Natural-language customer questions
- Semantic search using vector embeddings
- Qdrant vector database
- Top-K context retrieval
- LLM-based answer generation
- Curated business question-and-answer knowledge base
- Customer-facing chat interface
- Grounded responses based on retrieved business information

## Knowledge Base

The system uses question-and-answer pairs prepared by the business team.

Example structure:

```text
Question:
What is your return policy?

Answer:
Customers may return eligible products within the configured return period.
```

These entries are embedded and stored in Qdrant before the chatbot is used.

This allows the business to control the information available to the assistant while still allowing customers to ask questions in their own words.

## Project Structure

```text
rag-question-answering-system/
├── main.py
├── intents.json
├── scripts/
├── templates/
│   └── chatbot.html
├── static/
│   └── style.css
├── data/
├── models/
├── README.md
└── requirements.txt
```

The exact contents may vary depending on the local training and ingestion files used by the project.

## Tech Stack

- Python
- Qdrant
- Vector Embeddings
- Large Language Models
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Flask
- HTML / CSS

## Use Case

The system is intended for customer-facing business Q&A where:

- the business already knows the approved answers it wants customers to receive
- customers may phrase the same question in many different ways
- semantic retrieval is more useful than simple keyword matching
- an LLM is used to produce a more conversational final response

## Example Flow

```text
Customer:
"Do you deliver on weekends?"

        ↓

Qdrant retrieves the closest matching business Q&A entries.

        ↓

The Top-K results are sent to the LLM as context.

        ↓

Assistant:
Generates a response using the retrieved business information.
```

## Goal

The goal of the project is to make business knowledge easier for customers to access by combining the reliability of a curated Q&A knowledge base with the flexibility of semantic search and LLM-generated responses.
