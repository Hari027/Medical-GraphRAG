# 🧠 MedGraphRAG: High-Performance Medical Graph Retrieval-Augmented Generation

An enterprise-grade, Triple-Layer Medical Graph RAG framework engineered to eliminate "hallucinations" in healthcare AI. By anchoring LLM inferences to a **3.5 Million node UMLS (Unified Medical Language System)** ontological backbone stored directly in Neo4j, this system ensures that every clinical response is derived from and grounded within authoritative medical ground truth.

---

## 🏗️ Architecture: The Triple-Layer Semantic Graph
Traditional RAG models struggle in medical domains because they cannot connect disparate symptoms to underlying diseases across different documents. MedGraphRAG solves this by strictly organizing data into three distinct, interconnected layers:

### Layer 1: Live Patient Evidence (The "What")
- **Source**: Raw clinical notes, patient histories, or real-time diagnostic inputs.
- **Backend Process**: The application chunks the text and uses an LLM (e.g., `gpt-4o-mini`) to perform **Entity-to-Entity Relationship Extraction**. It doesn't just find words; it extracts semantic triples (e.g., `[Aspirin] --(treats)--> [Headache]`). These extracted networks form the **Active Graph**.

### Layer 2: Medical Literature (The "Context")
- **Source**: A persistent repository of 14,000+ PubMed articles, reference papers, and clinical trials.
- **Backend Process**: Instead of embedding these on the fly, Layer 2 embeddings are pre-computed in high-speed batches and persisted in Neo4j. This prevents RAM overload and provides the clinical "context" linking patient symptoms to documented medical research.

### Layer 3: Ground Truth Ontology (The "Dictionary")
- **Source**: The Unified Medical Language System (UMLS).
- **Backend Process**: A massive, read-only graph consisting of **3,480,704 concepts** and over **60 million relationships**. This layer lives *entirely* in the Neo4j database (Zero-RAM footprint). It is the authoritative dictionary that prevents the LLM from making medical assumptions.

---

## ⚙️ Engine Mechanics: Deep Dive into the Backend

### 1. Vectorized Cross-Layer Linking
Patient symptoms (Layer 1) do not exist in a vacuum. During ingestion, the system performs high-speed cosine similarity matching (using localized `HuggingFace` sentence-transformers) to automatically draw `the_reference_of` edges from Layer 1 to Layer 2, and `the_definition_of` edges from Layer 2 to Layer 3.

### 2. Database-Native Persistence & Batching
Earlier architectures loaded embeddings into Python RAM via lists, which aggressively bottlenecked ingestion (8+ minute hangs). This system was heavily refactored for **Production-Grade Scale**:
- Embeddings are generated in vectorized batches of 256.
- The results are physically synced to Neo4j.
- On application restart, loading 14,000 entities takes **<3 seconds**, and re-ingestion of known documents takes exactly **0 seconds**.

---

## 🔍 The Magic: How "U-Retrieval" Works
The core algorithmic achievement of this project is the **U-Retrieval Engine**. It operates like a funnel (Top-Down Search) followed by a refining filter (Bottom-Up Synthesis).

#### ⬇️ The "Top-Down" Search (Finding the Evidence)
Instead of executing full-text database scans, the system builds a hierarchical **Tag Tree** during ingestion. When a user queries the graph:
1. The LLM embeds the question.
2. It traverses from the broad "Root" branches of the Tag Tree (e.g., *Cardiology*) down to the specific semantic "Leaves" (e.g., *Arrhythmia symptoms in Chunk 4*).
3. This allows the system to instantly locate the exact subgraph in Layer 1 containing the relevant data without scanning the entire document.

#### 🔀 Triple-Neighbour Extraction
Once the target Layer 1 subgraph is found, the system performs a Neo4j Cypher traversal across **all three layers**. It gathers up to 50 localized nodes (Patient Data + PubMed References + UMLS Definitions) to create a highly focused context window.

#### ⬆️ The "Bottom-Up" Refinement (The Expert Review)
A standard AI stops at generating a ground-level answer. MedGraphRAG performs an iterative, multi-layer LLM synthesis:
1. **Initial Answer**: The LLM generates a response based *only* on the specific leaf nodes.
2. **Reviewing the Tree**: The system ascends the Tag Tree. It passes the *Initial Answer* back into the LLM, alongside the summarized meta-data of the parent tag (e.g., moving from the *Arrhythmia* tag to the broader *Patient Cardiac History* tag).
3. **Synthesis**: The LLM refines and updates the answer with this broader context. This repeats until the root of the tree is reached, ensuring the final answer accounts for both specific, localized symptoms and broad, long-term patient histories.

*(In the Streamlit UI, this process is visible via real-time progress callbacks: `⬆️ Bottom-Up Refinement (Level 2) with LLM...`)*

---

## 📈 Engineering Achievements & Scale
- **Zero-RAM Ontological Lookups**: Handled 3.5M nodes completely via Neo4j indexing, side-stepping standard `MemoryError` limitations found in naive RAG applications.
- **Asynchronous Database Indexing**: Boot times were reduced to ~1 second by offloading Cypher index constraints to background threading.
- **Deterministic AI Safety**: By relying on graph-traversal logic rather than vector-only similarity, the system prioritizes established medical relationships over probabilistic word guessing.

---

## 💻 Tech Stack
- **Database / Graph Engine**: Neo4j (Cypher)
- **Application Logic**: Python 3.12 
- **LLM Orchestration**: LangChain (`gpt-4o-mini`)
- **Embedding Models**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Frontend / Telemetry**: Streamlit

---

## 🚀 Quickstart / Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/Hari027/Medical-GraphRAG.git
cd Medical-GraphRAG

# 2. Setup your isolation environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory. *(Ensure your Neo4j Desktop server is running locally).*
```env
OPENAI_API_KEY=sk-your-openai-key
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
HUGGINGFACEHUB_API_TOKEN=hf_your_key
```

### Running the App
```bash
streamlit run app.py
```
* **Step 1**: Use the UI to ingest clinical text. The system extracts relationships and builds the Triple-Graph.
* **Step 2**: Navigate to the Query Tab, ask a medical question, and watch the live logs as the U-Retrieval engine performs Top-Down/Bottom-Up synthesis in real-time.
