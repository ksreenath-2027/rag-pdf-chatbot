# PDF RAG Chatbot with Citations

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from PDF documents and receive AI-generated answers with source citations.

## Features

- PDF ingestion and processing
- Document chunking
- Semantic search using embeddings
- Vector storage with Chroma DB
- AI-powered question answering using Groq + LLM
- Source citations with page references
- Conversational chat support

## Tech Stack

- Python
- LangChain
- Chroma Vector Database
- Hugging Face Embeddings
- Groq API
- FastAPI (coming next)
- Docker & Kubernetes (planned)

## Workflow

PDF → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer + Citations

## Example

Question:

> What is Kubernetes Service?

Answer:

> A Kubernetes Service exposes applications running in Pods and provides stable networking endpoints.

