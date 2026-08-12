# System Context: Optimized MedGraphRAG Architecture

You are an AI assistant helping with the development and refinement of a specialized medical AI project. Below is the complete context of the project architecture, the engineering decisions made, and the current state of the codebase. Please use this context to answer any future questions or assist with coding tasks.

## 1. Project Overview
The project is a privacy-first, lightweight implementation of **Medical Graph Retrieval-Augmented Generation (MedGraphRAG)**. The goal is to provide expert-level clinical reasoning (capable of passing the USMLE) on consumer-grade hardware without sending any Protected Health Information (PHI) to cloud providers like OpenAI or Google. 

## 2. Core Architecture: The Three-Layer Triple Graph
The knowledge graph is stored locally in **Neo4j** and consists of three distinct layers:
- **Layer 1 (Private Context)**: Meta-MedGraphs built from the user's private documents or queries.
- **Layer 2 (Evidentiary Grounding)**: Real-world medical evidence built by pulling relevant abstracts from the PubMed Entrez API.
- **Layer 3 (Standardized Definitions)**: Core medical concepts and terminology seeded directly from the UMLS Metathesaurus via its REST API.

## 3. Technology Stack (100% Local)
- **LLM Reasoning Engine**: Gemma 31B (served locally via Ollama with 4-bit quantisation to fit within local VRAM).
- **Embedding Model**: SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`). This replaces OpenAI embeddings, offering superior biomedical entity normalization while maintaining offline privacy.
- **Graph Database**: Neo4j Community Edition.
- **Cross-Layer Linking**: PyTorch with CUDA for offline GPU batched matrix multiplication.

## 4. Key Engineering Innovations & Differentiators
Unlike standard, computationally unbounded GraphRAG models, this project introduces specific optimizations to make it viable on local hardware:
- **Argmax Edge Constraint (L2 -> L3)**: Standard implementations use flat cosine similarity thresholds (e.g., 0.45) which cause an exponential explosion of edges (bloating the database and flooding the LLM context window with noise). We replaced this with a strict `argmax` constraint, forcing a 1:1 mapping. Every piece of PubMed evidence is locked to its single most semantically relevant UMLS definition. 
- **Sparse Graphing (L1 -> L2)**: Increased the similarity threshold ($\delta_r$) to 0.55 to heavily prune peripheral noise, optimizing the signal-to-noise ratio.
- **Zero Cloud APIs**: Replaced all proprietary API reliance with an offline GPU tensor linker.

## 5. The Retrieval Pipeline (U-Retrieval)
Instead of flat vector search, we use a structured **U-Retrieval** method:
- **Top-Down Traversal**: A hop-constrained traversal down a hierarchical tag tree. It starts at a semantically verified subgraph and filters down to prune bad candidates before calculating entity-level cosine scores.
- **Bottom-Up Refinement**: Re-grounds highly specific entity answers back into broader global context summaries, preventing the LLM from getting "tunnel vision" on single facts.

## 6. Current Benchmarks & Results
- **Dataset**: Evaluated zero-shot on MedQA-USMLE (1,273 questions).
- **Evaluation Protocol**: Automated zero-shot grading pipeline extracting the first A/B/C/D/E choice.
- **Baseline (Gemma 31B, No Retrieval)**: 83.0% accuracy.
- **Ours (Gemma 31B + MedGraphRAG)**: **89.3%** accuracy (+6.3% absolute gain).

## 7. Known Failure Modes (Error Analysis)
1. **Retrieval Noise (Semantic Drift)**: SapBERT struggles with biostatistics/epidemiology questions (e.g., measurement bias) because it clusters loosely around words like "trial," pulling up random clinical trials instead of methodological definitions.
2. **Negative-Constraint Questions**: The LLM frequently ignores "EXCEPT" qualifiers. Providing lists of graph-retrieved disease associations actually *hurts* accuracy here because it tempts the LLM into picking a true association.
3. **Visual/Image Bottleneck**: The pipeline is purely text-based. Any MedQA question referencing an exhibit image results in a random guess.

## 8. Future Roadmap
- **BM25 Hybrid Retrieval**: Adding sparse keyword retrieval alongside dense SapBERT embeddings to improve recall on exact genetic markers or statistical terms.
- **FAISS (ANN) Indexing**: Optimizing the PyTorch GPU linker with Approximate Nearest Neighbors to scale to the millions of nodes in the full UMLS vocabulary.
- **Dynamic U-Retrieval Router**: Implementing an LLM router to dynamically determine traversal depth based on the complexity of the query.
- **Methodological Subgraphs**: Building specialized graphs for biostats, ethics, and epidemiology from textbooks, as recent PubMed abstracts are heavily biased toward new treatments.
