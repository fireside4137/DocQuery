# DocQuery

A from-scratch Retrieval-Augmented Generation (RAG) pipeline in Python using LangChain, SentenceTransformers, ChromaDB, and multiple LLM backends (OpenAI, Groq, Anthropic Claude).

## Pipeline Overview

1. **Ingestion (`RAG_pipeline.ipynb`)**: Loads PDFs from `data/pdfs/` with `PyPDFLoader` and splits them into 500-character overlapping chunks using `RecursiveCharacterTextSplitter`.
2. **Embeddings**: Generates 384-dimensional embeddings locally with `all-MiniLM-L6-v2` (`SentenceTransformer`).
3. **Vector Store**: Persists embeddings, documents, and metadata in ChromaDB (`data/vector_store/`) using deterministic MD5 chunk IDs to avoid duplicates.
4. **Retrieval**: Embeds user queries, retrieves top-k candidates from ChromaDB, and scores them with `sklearn.metrics.pairwise.cosine_similarity`.
5. **Generation**: Feeds retrieved context into `generate_output()` with swappable LLM backends (`ChatOpenAI`, `ChatGroq`, `ChatAnthropic`).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   GROQ_API_KEY=your_groq_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```
3. Open and run `RAG_pipeline.ipynb`.
