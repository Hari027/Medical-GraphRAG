"""
GPU-Accelerated Cross-Layer Cosine Similarity Linker
Computes cosine similarity between all Layer 2 and Layer 3 entities using
PyTorch matmul on GPU, and pushes `the_definition_of` edges to Neo4j.
"""

import os
import sys
import gc
import numpy as np
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


def link_layers_gpu(uri: str, user: str, password: str,
                    threshold: float = 0.45,
                    embed_batch_size: int = 512,
                    sim_l2_batch: int = 250,      # Reduced from 1000 for stability
                    sim_l3_chunk: int = 100_000,  # Reduced from 500k for stability
                    progress_callback=None):
    """
    1. Loads all L2 and L3 entity names+context from Neo4j
    2. Embeds them (or reuses existing .dat file)
    3. Computes similarity in small, safe chunks with verbose logging
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CrossLayerLinker] Using device: {device}")
    
    # ---------------------------------------------------------------
    # Step 1: Load entities from Neo4j
    # ---------------------------------------------------------------
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    l2_names = []
    l2_texts = []
    l3_names = []
    l3_texts = []
    
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) WHERE n.layer = 2 RETURN n.name AS name, n.type AS type, n.context AS context")
        for record in result:
            name, etype, ctx = record["name"] or "", record["type"] or "", record["context"] or ""
            l2_names.append(name)
            l2_texts.append(f"name: {name}; type: {etype}; context: {ctx[:200]}")
            
        result = session.run("MATCH (n:Entity) WHERE n.layer = 3 RETURN n.name AS name, n.type AS type, n.context AS context")
        for record in result:
            name, etype, ctx = record["name"] or "", record["type"] or "", record["context"] or ""
            l3_names.append(name)
            l3_texts.append(f"name: {name}; type: {etype}; context: {ctx[:200]}")
    
    l3_total = len(l3_names)
    print(f"[CrossLayerLinker] Loaded L2: {len(l2_names)}, L3: {l3_total}")
    
    if not l2_names or not l3_names:
        driver.close(); return 0

    # ---------------------------------------------------------------
    # Step 2: Embed (or Resume from Disk)
    # ---------------------------------------------------------------
    mmap_file = "l3_embeddings.dat"
    embedding_dim = 384
    expected_size = l3_total * embedding_dim * 4 # float32
    
    resume_available = False
    if os.path.exists(mmap_file):
        actual_size = os.path.getsize(mmap_file)
        if actual_size == expected_size:
            print(f"[CrossLayerLinker] Found existing embeddings ({actual_size} bytes). RESUMING...")
            resume_available = True
    
    # Always embed L2 (it's small)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    l2_embeddings = model.encode(l2_texts, batch_size=embed_batch_size, normalize_embeddings=True, convert_to_numpy=True)
    l2_tensor_full = torch.from_numpy(l2_embeddings).to(torch.float32)
    del l2_embeddings

    if resume_available:
        l3_embeddings_mmap = np.memmap(mmap_file, dtype='float32', mode='r', shape=(l3_total, embedding_dim))
    else:
        print("[CrossLayerLinker] No valid embedding file found. Starting Stage 2...")
        l3_embeddings_mmap = np.memmap(mmap_file, dtype='float32', mode='w+', shape=(l3_total, embedding_dim))
        for start in range(0, l3_total, embed_batch_size * 20):
            end = min(start + embed_batch_size * 20, l3_total)
            chunk_emb = model.encode(l3_texts[start:end], batch_size=embed_batch_size, normalize_embeddings=True, convert_to_numpy=True)
            l3_embeddings_mmap[start:end] = chunk_emb
            l3_embeddings_mmap.flush()
            del chunk_emb
            gc.collect()
            print(f"  Embedded {end}/{l3_total}...")

    # Free model
    del model
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Chunked GPU Top-1 Similarity Search
    # ---------------------------------------------------------------
    num_l2 = l2_tensor_full.shape[0]
    total_l3_chunks = (l3_total + sim_l3_chunk - 1) // sim_l3_chunk
    total_iterations = total_l3_chunks # Since we do L2 in one block if small enough
    
    # Global buffers to track the absolute best match for each L2 article
    best_scores = np.zeros(num_l2, dtype='float32')
    best_target_names = [""] * num_l2
    
    # Optimization: L2 is usually small enough (~14k) to stay on GPU during the whole cycle
    l2_gpu_full = l2_tensor_full.to(device)
    
    print(f"[CrossLayerLinker] Finding Top-1 Best Definition for {num_l2} articles across {total_l3_chunks} UMLS chunks...")

    for l3_chunk_idx in range(total_l3_chunks):
        l3_start_idx = l3_chunk_idx * sim_l3_chunk
        l3_end_idx = min(l3_start_idx + sim_l3_chunk, l3_total)
        
        # Load and move L3 chunk to GPU
        l3_np_slice = l3_embeddings_mmap[l3_start_idx : l3_end_idx]
        l3_chunk_full = torch.from_numpy(l3_np_slice).to(torch.float32)
        l3_gpu_slice = l3_chunk_full.to(device)
        chunk_l3_names = l3_names[l3_start_idx : l3_end_idx]
        
        # Compute dot product
        # (num_l2, emb_dim) @ (emb_dim, chunk_size) -> (num_l2, chunk_size)
        sim_matrix = torch.mm(l2_gpu_full, l3_gpu_slice.t())
        
        # Find best match within THIS chunk
        max_vals, max_indices = torch.max(sim_matrix, dim=1)
        
        # Move results to CPU for global buffer comparison
        max_vals_cpu = max_vals.cpu().numpy()
        max_indices_cpu = max_indices.cpu().numpy()
        
        # Update global bests
        for i in range(num_l2):
            if max_vals_cpu[i] > best_scores[i]:
                best_scores[i] = max_vals_cpu[i]
                best_target_names[i] = chunk_l3_names[int(max_indices_cpu[i])]
        
        # Progress report
        pct = 0.65 + 0.30 * ((l3_chunk_idx + 1) / total_l3_chunks)
        if progress_callback:
            progress_callback(min(pct, 0.95), f"Scanning UMLS Chunk {l3_chunk_idx+1}/{total_l3_chunks}...")
        print(f"  Chunk {l3_chunk_idx+1}/{total_l3_chunks} scanned.")
        
        # Cleanup
        del l3_gpu_slice, l3_chunk_full, sim_matrix
        if device == "cuda":
            torch.cuda.empty_cache()
            
    # ---------------------------------------------------------------
    # Step 4: Finalize and Push to Neo4j
    # ---------------------------------------------------------------
    # Only link if score > manual threshold (e.g. 0.60) to avoid linking garbage
    final_threshold = 0.60
    edge_batch = []
    links_created = 0
    
    print(f"[CrossLayerLinker] Scanning complete. Filtering the {num_l2} best matches...")
    
    for i in range(num_l2):
        if best_scores[i] >= final_threshold:
            edge_batch.append({
                "source": l2_names[i],
                "target": best_target_names[i],
                "similarity": float(best_scores[i])
            })
            links_created += 1
            
            if len(edge_batch) >= 2000:
                _push_edges(driver, edge_batch)
                edge_batch = []

    if edge_batch:
        _push_edges(driver, edge_batch)
    
    driver.close()
    
    # Cleanup memmap
    del l3_embeddings_mmap
    try: os.remove(mmap_file)
    except: pass
    
    print(f"[CrossLayerLinker] Complete! Created {links_created} high-quality 'the_definition_of' relationships.")
    if progress_callback:
        progress_callback(1.0, f"Done! Created {links_created} L2->L3 definition links.")
    
    return links_created


def _push_edges(driver, batch):
    """Push a batch of the_definition_of edges to Neo4j."""
    with driver.session() as session:
        query = (
            "UNWIND $batch AS item "
            "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
            "MERGE (s)-[rel:LINK {type: 'the_definition_of'}]->(t) "
            "SET rel.similarity = item.similarity"
        )
        session.run(query, batch=batch)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    edges = link_layers_gpu(uri, user, password, threshold=0.45)
    print(f"Total edges created: {edges}")
