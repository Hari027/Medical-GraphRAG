# MedGraphRAG 🧠🧬

**High-Performance database-resident Medical Graph Retrieval-Augmented Generation.**

An advanced, Triple-Layer Medical Graph RAG framework designed to eliminate hallucinations in healthcare AI. By anchoring LLM semantic inferences to a **3.5 Million node UMLS (Unified Medical Language System)** backbone stored directly in Neo4j, this system ensures clinical responses are grounded in authoritative ground truth without crashing your machine's RAM.

---

## ⚡ Native Database-First Architecture
This version of MedGraphRAG has been refactored for **Production-Grade Scale**. It offloads all "Million-Node" complexity to Neo4j, allowing the application to run smoothly on standard hardware.

- **🚀 Database-Native U-Retrieval**: Performs hierarchical tag traversals directly inside Neo4j. Top-Down search and Bottom-Up refinement deliver evidence-based answers in ~45s.
- **🔋 Zero-RAM Vocabulary**: The 3.5 Million+ UMLS concepts (Layer 3) reside entirely in Neo4j. Python only handles the active document, keeping RAM usage extremely low.
- **📦 Persistent Embeddings**: Layer 2 (PubMed) and Layer 3 entities store their vector embeddings directly in the graph. Startup is instant; ingestion doesn't require recalculating known data.
- **🔗 Vectorized Triple Linking**: High-speed NumPy-based cross-layer linking that bridges patient documents (L1) to medical literature (L2) and vocabulary (L3).

---

## 🏗️ Triple-Layer Architecture

1.  **Layer 1 (Live Evidence):** Your uploaded patient documents and clinical notes, parsed into Semantic Triples.
2.  **Layer 2 (Medical Literature):** A repository of 14,000+ PubMed articles and reference papers providing clinical context.
3.  **Layer 3 (Ground Truth):** The medical backbone. 3.5 Million UMLS concepts and 60M+ relationships forming the "dictionary" for all medical reasoning.

---

## 💻 Tech Stack
- **Graph Engine**: Neo4j (Cypher) - Primary store and traversal engine.
- **LLM**: GPT-4o-mini (via LangChain) for extraction and refinement.
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) hosted locally.
- **UI**: Streamlit with live progress logging for both Ingestion and Retrieval.

---

## 📦 Getting Started

### 1. Installation
```bash
git clone https://github.com/Hari027/Medical-GraphRAG.git
cd Medical-GraphRAG

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root:
```env
OPENAI_API_KEY=sk-xxxx...
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## 📥 Usage Flow

### High-Speed Ingestion
1. Start the app: `streamlit run app.py`.
2. Paste your patient document or clinical text.
3. Click **"Ingest Patient Document"**.
4. The system will extract entities, link them to the 3.5M node backbone, and build the tag hierarchy in seconds.

### Professional U-Retrieval
1. Go to the **"Query"** tab.
2. Ask a complex medical question.
3. Watch the **live progress** as the system performs:
   - **Top-Down Search**: Finding the most relevant graph cluster.
   - **Cross-Layer Expansion**: Pulling clinical neighbors from all 3 layers.
   - **Bottom-Up Refinement**: Cleaning and verifying the answer using hierarchical tags.

---

## 🔬 Safety & Performance
- **Deterministic Verification**: No "lucky guesses." Every LLM answer is cross-checked against the Neo4j ontology.
- **Scalability**: Designed to handle 3.5M+ nodes without performance degradation.
- **Privacy**: Patient data is processed into a graph format; raw text is never stored in the permanent knowledge base.
