# Portfolio AI Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered portfolio chatbot that answers questions about your projects, skills, and experience using RAG with Pinecone vector search and Ollama local inference. Embeds your portfolio data into a vector store and retrieves relevant context for conversational responses via Streamlit.

## Features

- **RAG-powered Q&A** over your portfolio data using Pinecone + Ollama
- **Session management** with Redis-backed conversation history
- **Streamlit chat interface** with streaming responses
- **Customizable portfolio data** — edit `portfolio_data.txt` with your info

## Quick Start

```bash
git clone https://github.com/rchhabra13/portfolio-ai.git
cd portfolio-ai
pip install -r requirements.txt
cp .env.example .env  # Configure Pinecone + Ollama settings
streamlit run portfolio_chatbot.py
```

## Configuration

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone vector database API key |
| `PINECONE_INDEX` | Index name for portfolio embeddings |
| `OLLAMA_BASE_URL` | Ollama server URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | Model to use (default: llama3.2) |

Edit `portfolio_data.txt` to customize the chatbot with your own projects, skills, and experience.

## Tech Stack

Python, Streamlit, Pinecone, Ollama, LangChain, Redis

## License

MIT
