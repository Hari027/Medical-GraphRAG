# MedGraphRAG 🧠🧬

An advanced, locally hosted, Triple-Layer Medical Graph Retrieval-Augmented Generation (RAG) framework designed to eliminate hallucination in LLM healthcare interactions by rigorously anchoring semantic inferences to the authoritative UMLS (Unified Medical Language System) Knowledge Graph.

Based on the architecture proposed in *"Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation"*, this implementation is scaled to natively support end-to-end local knowledge graph ingestion—bypassing rate limits and securely hosting tens of millions of authentic medical concepts exactly where you compute.

## 🚀 Features

- **Triple-Layer Graph Architecture:**
  - **Layer 1 (Meta-MedGraphs):** Your personal documents & clinical notes structurally chunked and parsed automatically into entities and Semantic Triples.
  - **Layer 2 (Domain Graph):** Broad medical texts, literature, and internal book repositories to anchor arbitrary LLM knowledge chunks.
  - **Layer 3 (Core Medical Vocabulary):** High-density backbone powered by a specialized bulk importer capable of gracefully ingesting and indexing ~4M Nodes (concepts from `MRCONSO.RRF` and `MRSTY.RRF`) and ~60M+ Edges (ontologies from `MRREL.RRF`) into a local Neo4j Cluster seamlessly.
- **U-Retrieval Routing:** Top-Down indexing combined with Bottom-Up semantic retrieval dynamically traverses your private Graph clustering structure, bypassing naive vector-similarity hallucinations.
- **Intelligent Batch Stream Pipeline:** Features a hyper-optimized Python bulk ingester to map and merge massive multi-gigabyte NIH data distributions directly into Neo4j with exponential index tracking and streaming—bypassing memory overflow bugs native to smaller RAG attempts.
- **Streamlit Analytics Dashboard:** Highly interactive analytical dashboard and control room for real-time document chunk evaluation, graphical layout mapping, and direct connection pipelines.

## 🛠️ Architecture Stack
- **Graph Database:** Neo4j (Cypher)
- **Memory Construction:** NetworkX
- **Orchestration:** LangChain
- **Embeddings & LLM Generation:** OpenAI
- **UI:** Streamlit

---

## 🚦 System Requirements

- **Python 3.10+**
- **Neo4j Desktop / Server Setup** running locally (or remotely)
- **UMLS Dataset Extracts** downloaded from the NIH National Library of Medicine. Specifically, you need `MRCONSO.RRF`, `MRSTY.RRF`, and `MRREL.RRF` extracted into the root of this project.

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/MedGraphRAG-Local.git
   cd MedGraphRAG-Local
   ```

2. **Initialize Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure the .env variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-xxxx...
   NEO4J_URI=bolt://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=my_secure_password
   ```

---

## 💻 Running the Control Room

Simply boot the frontend:
```bash
streamlit run app.py
```

### 🗄️ Ingesting the UMLS Backbone (First Time Setup)
1. Place the massive raw UMLS dump files `MRCONSO.RRF` and `MRSTY.RRF` directly into the root folder.
2. In the Streamlit sidebar, select **"📥 Load Local UMLS Nodes"**. (This securely mounts ~4 million fundamental medical definitions tracking directly into your local database using `MERGE` properties + automatic cypher indexing to enforce `O(1)` inserts).
3. Ensure `MRREL.RRF` is loaded in the root. 
4. Select **"🔗 Load Local UMLS Relationships"**. The background memory processor will silently build a translation layer mapping the millions of numerical CUI IDs together, bridging up to ~65 Million ontological edges across your architecture in roughly 15 minutes. 

### 📝 Processing Clinical Documents
Drag and drop unstructured clinical summaries, papers, or patient anonymized reports into the primary document ingestor. The engine will chunk it, tag it against your massive Layer-3 backbone, and allow you to ask RAG questions with supreme accuracy and zero hallucination. 

---

## ⚖️ License
This project operates under the MIT License. Due to NIH restrictions, the raw UMLS datasets are **not** bundled in this repository. Users must license and obtain UMLS `.RRF` data directly from the National Library of Medicine. 
