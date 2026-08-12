# Medical-GraphRAG

A medical GraphRAG system that combines patient-specific evidence, PubMed literature, and UMLS terminology into a triple-layer knowledge graph for grounded medical question answering.

## What it Does
Medical-GraphRAG is an experimental/research prototype designed to improve the grounding of large language models (LLMs) in the healthcare domain. It organizes medical data into a structured knowledge graph, linking patient symptoms and queries to established medical literature and standardized vocabularies.

## Key Features
- **Triple-Layer Semantic Graph**: Combines active patient notes (Layer 1), PubMed evidence (Layer 2), and UMLS dictionary terms (Layer 3).
- **Vectorized Cross-Layer Linking**: Uses sentence-transformers to link entities semantically across layers.
- **Top-Down & Bottom-Up Retrieval**: Implements a graph-based retrieval strategy (U-Retrieval) that narrows down to specific nodes and synthesizes broader contextual answers.
- **Database-Native Operations**: Leverages Neo4j for efficient graph traversal and persistence, reducing memory overhead for large ontologies.

## Architecture & Retrieval Pipeline
The system uses a U-Retrieval process:
1. **Top-Down Retrieval**: The LLM embeds the question and traverses a hierarchical tag tree to locate relevant Layer 1 subgraphs.
2. **Triple-Neighbour Expansion**: Neo4j Cypher traversals gather localized context spanning all three layers (Patient Data + PubMed References + UMLS Definitions).
3. **Bottom-Up Refinement**: The system iterates over the tag tree, synthesizing specific answers back up into a global, refined summary.


## Data Sources
- **PubMed**: Ingested via NCBI E-utilities for Layer 2. Uses MeSH terms for structured metadata.
- **UMLS**: Unified Medical Language System provides the core definitions and relationships for Layer 3. (TODO: Specify UMLS release/version).

## Technology Stack
- **Database**: Neo4j
- **Application**: Python, Streamlit
- **LLM**: LangChain (e.g., `gpt-4o-mini`, `Gemma 31B`)
- **Embeddings**: Sentence-Transformers (e.g., `SapBERT-from-PubMedBERT-fulltext`)

## Repository Structure
```text
Medical-GraphRAG/
├── app/               # Streamlit application UI
├── medgraphrag/       # Core GraphRAG pipeline, models, and LLM clients
├── ingestion/         # Scripts for PubMed and UMLS data ingestion

├── screenshots/       # (TODO: Add screenshots of the Streamlit UI)
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Hari027/Medical-GraphRAG.git
cd Medical-GraphRAG
```

2. **Create a virtual environment**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Variables**
Copy the example environment file and fill in your keys:
```bash
cp .env.example .env
```
Ensure you provide `OPENAI_API_KEY`, `UMLS_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, and configure your `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` appropriately.

5. **Neo4j Setup**
Ensure you have a Neo4j instance running (e.g., Neo4j Desktop or Docker) matching the credentials in your `.env`.

6. **PubMed & UMLS Ingestion**
To populate Layer 2 and Layer 3, run the ingestion scripts:
*(Note: UMLS ingestion requires the raw UMLS `.RRF` data files to be downloaded and placed locally in the repository root folder before running).*
```bash
python ingestion/pubmed.py
python ingestion/umls.py
```

## Running the Application
```bash
streamlit run app/streamlit_app.py
```

## Evaluation / Benchmarking
Initial local testing with a 31B open-weights model using the MedQA-USMLE dataset demonstrated an 89.3% accuracy rate using this architecture.

## Limitations and Safety
- **Not Clinically Validated**: This is an educational/research prototype. It is not a clinically validated decision-support system and must not be used for actual medical diagnosis or treatment.
- **Hallucinations**: While designed to improve grounding, generated answers can still be incorrect.
- **Retrieval Dependency**: Quality depends heavily on entity extraction accuracy and cosine similarity thresholds.
- **Privacy**: Do not enter actual Protected Health Information (PHI) or patient-identifiable data into this prototype.

## References
This implementation is inspired by and adapts concepts from recent literature on Medical Graph RAG (e.g., "Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation").
- Embeddings use models such as [SapBERT](https://github.com/cambridgeltl/sapbert).
- Data sourced from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) and [UMLS](https://www.nlm.nih.gov/research/umls/index.html).

## License
MIT License

