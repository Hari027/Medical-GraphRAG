"""
MedGraphRAG — Triple Graph Construction + U-Retrieval
Based on: "Medical Graph RAG: Towards Safe Medical Large Language Model
           via Graph Retrieval-Augmented Generation" (Wu et al., 2024)
"""

from __future__ import annotations
from typing import Optional
import networkx as nx
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_models import Entity, Relationship, MetaMedGraph, MEDICAL_TAGS, BUILTIN_VOCAB
from api_clients import UMLSClient, Neo4jClient, PubMedClient
from llm_helpers import (
    EmbeddingStore, 
    _call_llm_json, 
    _extract_entities, 
    _extract_relationships, 
    _tag_graph, 
    _generate_answer, 
    _refine_answer
)

class MedGraphRAG:
    """
    Implements the three-layer graph + U-Retrieval from the MedGraphRAG paper.

    Graph layers
    ------------
    Layer 1 : Meta-MedGraphs  (one per semantic chunk of user documents)
    Layer 2 : Repository graph – Med Papers/Books entities
    Layer 3 : Repository graph – Medical Vocabulary / UMLS-style controlled vocab

    Cross-layer links are built by cosine-similarity thresholding.
    """

    SIMILARITY_THRESHOLD = 0.45   # δr  — cross-layer linking threshold
    TAG_MERGE_THRESHOLD  = 0.60   # δt  — hierarchical tag clustering threshold
    TOP_N_ENTITIES       = 8      # Nu  — entities retrieved per query
    TOP_K_NEIGHBOURS     = 2      # ku  — triple-neighbour hops
    MAX_TRIPLE_NEIGHBORS = 50     # safety cap to avoid OpenAI 429 token limits
    MAX_TAG_LAYERS       = 6      # max U-Retrieval layers

    def __init__(
        self, 
        llm: ChatOpenAI, 
        embedder: OpenAIEmbeddings, 
        umls_api_key: Optional[str] = None,
        neo4j_creds: Optional[dict] = None,
    ):
        self.llm = llm
        self.emb = EmbeddingStore(embedder)
        self.umls = UMLSClient(api_key=umls_api_key)
        self.pubmed = PubMedClient()
        
        # Neo4j setup
        self.neo4j = None
        if neo4j_creds:
            self.neo4j = Neo4jClient(
                uri=neo4j_creds.get("uri"),
                user=neo4j_creds.get("user"),
                password=neo4j_creds.get("password")
            )

        # Layer 1 – user RAG graphs
        self.meta_graphs: list[MetaMedGraph] = []

        # Layer 2 – repository (paper) entities — restore from Neo4j for persistence
        if self.neo4j:
            self.repo_entities_l2: list[Entity] = self.neo4j.load_layer2_entities()
            print(f"[MedGraphRAG] Restored {len(self.repo_entities_l2)} Layer-2 entities from Neo4j.")
        else:
            self.repo_entities_l2: list[Entity] = []

        # Layer 3 – vocabulary entities (UMLS)
        # WE NO LONGER load 3.5Million nodes into Python RAM.
        # Layer 3 resides persistently in Neo4j only.
        self.repo_entities_l3: list[Entity] = [] 
 
        # Hierarchical tag tree  [{graph_id: str, tags: dict, children: list}]
        self.tag_tree: list[dict] = []

        # NetworkX graph for full triple structure
        self.nx_graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Layer 3 – Vocabulary (UMLS-style) constant layer initialization
    # ------------------------------------------------------------------

    def _build_vocab_layer(self) -> list[Entity]:
        entities = []
        for v in BUILTIN_VOCAB:
            e = Entity(
                name=v["name"],
                entity_type=v["type"],
                context=v["definition"],
                definition=v["definition"],
                layer=3,
            )
            entities.append(e)
        return entities

    def bulk_seed_vocabulary(self, terms: list[str], progress_callback=None):
        """
        Seed Layer 3 with a specific list of medical terms.
        Falls back to BUILTIN_VOCAB if UMLS fails.
        """
        total = len(terms)
        for i, term in enumerate(terms):
            # Duplicate check
            if any(v.name.lower() == term.lower() for v in self.repo_entities_l3):
                if progress_callback:
                    progress_callback(i + 1, total, f"⏩ Skipping (exists): {term}")
                continue

            details = None
            if self.umls.api_key:
                if progress_callback:
                    progress_callback(i + 1, total, f"🔍 UMLS Seeding: {term}")
                details = self.umls.get_term_details(term)
            
            if details:
                u_ent = Entity(
                    name=details["name"],
                    entity_type=details["type"],
                    context=details["definition"] or f"UMLS Concept: {details['name']}",
                    definition=details["definition"],
                    layer=3,
                )
                self.repo_entities_l3.append(u_ent)
                if self.neo4j:
                    self.neo4j.sync_entities([u_ent])
            else:
                # Fallback to BUILTIN_VOCAB
                for vocab in BUILTIN_VOCAB:
                    if vocab["name"].lower() == term.lower():
                        if progress_callback:
                            progress_callback(i + 1, total, f"🔍 BUILTIN fallback: {term}")
                        u_ent = Entity(
                            name=vocab["name"],
                            entity_type=vocab["type"],
                            context=vocab["definition"],
                            definition=vocab["definition"],
                            layer=3,
                        )
                        self.repo_entities_l3.append(u_ent)
                        if self.neo4j:
                            self.neo4j.sync_entities([u_ent])
                        break

    # ------------------------------------------------------------------
    # Step 1 – Semantic document chunking
    # ------------------------------------------------------------------

    def _semantic_chunks(self, text: str, chunk_size: int = 1200) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Step 2+4 – Entity extraction + relationship linking → MetaMedGraph
    # ------------------------------------------------------------------

    def _build_meta_graph(self, chunk_text: str, graph_id: str) -> MetaMedGraph:
        g = MetaMedGraph(graph_id=graph_id, source_text=chunk_text)
        g.entities = _extract_entities(self.llm, chunk_text)
        g.relationships = _extract_relationships(self.llm, chunk_text, g.entities)

        # Embed entities
        for e in g.entities:
            e.embedding = self.emb.embed(e.content_text)

        return g

    # ------------------------------------------------------------------
    # Step 3 – Triple Linking  (L1 → L2 → L3)
    # ------------------------------------------------------------------

    def _embed_all_layers(self, skip_l3: bool = False, progress_callback=None):
        """Batch-embed all L2 entities that are missing embeddings, then persist."""
        targets = self.repo_entities_l2
        if not skip_l3:
            targets = targets + self.repo_entities_l3

        # Find entities missing embeddings
        missing = [e for e in targets if e.embedding is None]
        if not missing:
            if progress_callback:
                progress_callback(f"✅  All {len(targets)} L2 entities already have embeddings.")
            return

        if progress_callback:
            progress_callback(f"⚡  Batch-embedding {len(missing)}/{len(targets)} L2 entities (one-time backfill)…")

        # Batch embed for speed (uses embed_documents internally)
        texts = [e.content_text for e in missing]
        batch_size = 256
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_vecs = self.emb.embed_batch(batch_texts)
            for j, vec in enumerate(batch_vecs):
                missing[batch_start + j].embedding = vec
            if progress_callback:
                progress_callback(f"⚡  Embedded {batch_end}/{len(missing)} L2 entities…")

        # Persist embeddings back to Neo4j so this NEVER runs again
        if self.neo4j and missing:
            if progress_callback:
                progress_callback(f"💾  Persisting {len(missing)} embeddings to Neo4j (one-time)…")
            self.neo4j.sync_entities(missing, progress_callback=progress_callback)
            if progress_callback:
                progress_callback("✅  Embeddings persisted. Future runs will be instant.")

    def _link_layers(self, skip_l3_embed: bool = False, progress_callback=None):
        """
        For every Layer-1 entity, find sufficiently similar Layer-2 entities
        and add 'the_reference_of' edges. Vectorized for instant performance.
        """
        self._embed_all_layers(skip_l3=skip_l3_embed, progress_callback=progress_callback)
        
        if not self.repo_entities_l2:
            return
            
        l2_embs = []
        valid_l2 = []
        for e2 in self.repo_entities_l2:
            if e2.embedding is not None:
                l2_embs.append(e2.embedding)
                valid_l2.append(e2)
        
        if not valid_l2:
            return
            
        l2_matrix = np.vstack(l2_embs)
        l2_norms = np.linalg.norm(l2_matrix, axis=1)
        
        # Batch collect edges to avoid repetitive DB calls
        new_edges = []
        
        for mg in self.meta_graphs:
            for e1 in mg.entities:
                if e1.embedding is None:
                    continue
                
                e1_norm = np.linalg.norm(e1.embedding)
                if e1_norm == 0: continue
                
                # Fast vectorized cosine similarity
                sims = np.dot(l2_matrix, e1.embedding) / (l2_norms * e1_norm + 1e-9)
                matches = np.where(sims >= self.SIMILARITY_THRESHOLD)[0]
                
                for idx in matches:
                    e2 = valid_l2[idx]
                    sim = float(sims[idx])
                    
                    if not self.nx_graph.has_node(e2.name):
                        self.nx_graph.add_node(e2.name, entity=e2)
                    self.nx_graph.add_edge(
                        e1.name, e2.name,
                        relation="the_reference_of",
                        similarity=sim,
                    )
                    new_edges.append((e1.name, e2.name, "the_reference_of", sim))
        
        if self.neo4j and new_edges:
            self.neo4j.batch_add_cross_layer_edges(new_edges)

    # ------------------------------------------------------------------
    # Step 5 – Tag the graphs  (hierarchical clustering)
    # ------------------------------------------------------------------

    def _tag_all_graphs(self, progress_callback=None):
        import concurrent.futures
        total = len(self.meta_graphs)
        if total == 0: return

        # Progress tracking for thread pool
        finished_count = 0
        def tag_one(mg):
            mg.tag_summary = _tag_graph(self.llm, mg)
            return mg

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_mg = {executor.submit(tag_one, mg): mg for mg in self.meta_graphs}
            for future in concurrent.futures.as_completed(future_to_mg):
                finished_count += 1
                if progress_callback:
                    progress_callback(f"🏷️  Tagged graph {finished_count}/{total}…")

    def _build_tag_tree(self, progress_callback=None):
        """
        Agglomerative hierarchical clustering over tag embeddings.
        Vectorized using NumPy for high performance on many chunks.
        """
        import numpy as np
        
        if not self.meta_graphs:
            self.tag_tree = []
            return

        # 1. Pre-calculate 'centroid' embeddings for each graph's tags
        nodes = []
        for mg in self.meta_graphs:
            tag_embs = []
            for t_val in mg.tag_summary.values():
                tag_embs.append(self.emb.embed(t_val))
            
            centroid = np.mean(tag_embs, axis=0) if tag_embs else np.zeros(384)
            nodes.append({
                "ids": [mg.graph_id],
                "tags": mg.tag_summary,
                "embedding": centroid,
                "children": []
            })

        def _get_node_sim(n1, n2):
            # Fast cosine between centroids as a proxy for Average Linkage
            norm1 = np.linalg.norm(n1["embedding"])
            norm2 = np.linalg.norm(n2["embedding"])
            if norm1 == 0 or norm2 == 0: return 0.0
            return float(np.dot(n1["embedding"], n2["embedding"]) / (norm1 * norm2))

        # 2. Iterate and Merge
        total_merges = len(nodes) - 1
        for merge_idx in range(total_merges):
            if len(nodes) <= 1: break
            if progress_callback:
                progress_callback(f"🌳  Merging Tag Branches: {merge_idx+1}/{total_merges}…")

            best_sim, best_i, best_j = -1.0, 0, 1
            # Current simplified O(N^2) search. For N=34, this is <0.1s.
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    sim = _get_node_sim(nodes[i], nodes[j])
                    if sim > best_sim:
                        best_sim, best_i, best_j = sim, i, j

            if best_sim < self.TAG_MERGE_THRESHOLD:
                break

            # Create merged node
            merged_tags = self._merge_tags(nodes[best_i]["tags"], nodes[best_j]["tags"])
            # Re-calculate centroid for the branch
            combined_ids = nodes[best_i]["ids"] + nodes[best_j]["ids"]
            # Weighting by branch size for accuracy
            w1, w2 = len(nodes[best_i]["ids"]), len(nodes[best_j]["ids"])
            new_centroid = (nodes[best_i]["embedding"] * w1 + nodes[best_j]["embedding"] * w2) / (w1 + w2)
            
            merged = {
                "ids": combined_ids,
                "tags": merged_tags,
                "embedding": new_centroid,
                "children": [nodes[best_i], nodes[best_j]],
            }
            
            # Update nodes list
            new_list = [n for k, n in enumerate(nodes) if k not in (best_i, best_j)]
            new_list.append(merged)
            nodes = new_list

        self.tag_tree = nodes

    def _tag_similarity(self, tags_a: dict[str, str], tags_b: dict[str, str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        sims = []
        for ta_val in tags_a.values():
            for tb_val in tags_b.values():
                sims.append(self.emb.similarity(ta_val, tb_val))
        return float(np.mean(sims)) if sims else 0.0

    def _merge_tags(self, tags_a: dict[str, str], tags_b: dict[str, str]) -> dict[str, str]:
        merged = {}
        all_keys = set(tags_a) | set(tags_b)
        for k in all_keys:
            parts = []
            if k in tags_a:
                parts.append(tags_a[k])
            if k in tags_b:
                parts.append(tags_b[k])
            merged[k] = "; ".join(parts)
        return merged

    # ------------------------------------------------------------------
    # Step 6 – U-Retrieval
    # ------------------------------------------------------------------

    def _get_node_sim(self, query_obj, node_obj):
        """Helper to compute similarity between query and tag tree node."""
        if "embedding" not in node_obj:
            # If no centroid, fall back to string comparison or re-embed (cached)
            return self._tag_similarity(query_obj.get("tags", {}), node_obj.get("tags", {}))
        
        q_emb = query_obj["embedding"]
        n_emb = node_obj["embedding"]
        return np.dot(q_emb, n_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(n_emb) + 1e-9)

    def _top_down_retrieve(self, q_tags: dict[str, str]) -> Optional[MetaMedGraph]:
        """
        Database-native top-down retrieval. Traverses the TagNode hierarchy 
        in Neo4j to find the most relevant chunk.
        """
        if not self.neo4j:
            return self.meta_graphs[0] if self.meta_graphs else None

        # Fallback to local search if no TagNodes found in DB
        if not self.tag_tree:
             return self.meta_graphs[0] if self.meta_graphs else None
        
        # We start with the roots and move down
        current_level_nodes = [n for n in self.tag_tree]
        target_mg = None
        
        for _ in range(self.MAX_TAG_LAYERS):
            if not current_level_nodes: break
            best_sim, best_node = -1.0, None
            for node in current_level_nodes:
                sim = self._get_node_sim({"embedding": self.emb.embed(str(q_tags))}, node)
                if sim > best_sim:
                    best_sim, best_node = sim, node
            
            if not best_node: break
            if not best_node["children"]:
                # Found leaf
                target_id = best_node["ids"][0]
                for mg in self.meta_graphs:
                    if mg.graph_id == target_id:
                        return mg
                break
            else:
                current_level_nodes = best_node["children"]

        return self.meta_graphs[0] if self.meta_graphs else None

    def _get_triple_neighbours(
        self, entity_name: str, k: int
    ) -> list[Entity]:
        """
        Return all entities within k hops across all three graph layers
        (following any edge type, including cross-layer links).
        Queries directly against the persistent Neo4j Graph.
        """
        if self.neo4j:
            # High-performance DB search across all millions of L2 and L3 nodes
            return self.neo4j.get_k_hop_neighbors(entity_name, k)

        # Fallback to local nx_graph if Neo4j is not connected
        neighbours: list[Entity] = []
        try:
            reachable = nx.single_source_shortest_path_length(
                self.nx_graph.to_undirected(), entity_name, cutoff=k
            )
            for name, dist in reachable.items():
                if name == entity_name or dist == 0:
                    continue
                node_data = self.nx_graph.nodes.get(name, {})
                ent = node_data.get("entity")
                if ent:
                    neighbours.append(ent)
        except Exception:
            pass
        return neighbours

    def query(self, question: str, progress_callback=None) -> dict:
        """
        Full U-Retrieval pipeline.
        Returns a dict with keys: answer, target_graph, top_entities,
        triple_neighbours, refinement_log.
        """
        if not self.meta_graphs:
            return {"answer": "No documents loaded.", "target_graph": None,
                    "top_entities": [], "triple_neighbours": [], "refinement_log": []}

        if progress_callback: progress_callback("🔍 Running Top-Down Retrieval...")
        # Top-down retrieval
        target_mg = self._top_down_retrieve(question)

        # Embed query and retrieve top-N entities from target graph
        q_emb = self.emb.embed(question)
        scored = []
        for e in target_mg.entities:
            if e.embedding is not None:
                sim = self.emb.similarity(q_emb, e.embedding)
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_entities = [e for _, e in scored[: self.TOP_N_ENTITIES]]

        if progress_callback: progress_callback(f"🔗 Gathering Triple Neighbors for {len(top_entities)} entities...")
        # Gather triple neighbours (cross-layer)
        triple_neighbours: list[Entity] = []
        seen = {e.name for e in top_entities}
        for e in top_entities:
            for nb in self._get_triple_neighbours(e.name, self.TOP_K_NEIGHBOURS):
                if len(triple_neighbours) >= self.MAX_TRIPLE_NEIGHBORS:
                    break
                if nb.name not in seen:
                    seen.add(nb.name)
                    triple_neighbours.append(nb)
            if len(triple_neighbours) >= self.MAX_TRIPLE_NEIGHBORS:
                break

        if progress_callback: progress_callback("🧠 Generating initial answer using Ground-Level graph...")
        # Initial bottom-level answer
        answer = _generate_answer(
            self.llm, question, target_mg, top_entities, triple_neighbours
        )

        refinement_log = [{"level": 0, "answer": answer}]

        # Bottom-up refinement through higher tag layers
        def collect_ancestor_summaries(nodes, target_id, depth=0) -> list[tuple[int, dict]]:
            results = []
            for node in nodes:
                if target_id in node["ids"]:
                    results.append((depth, node["tags"]))
                    if node["children"]:
                        results += collect_ancestor_summaries(
                            node["children"], target_id, depth + 1
                        )
            return results

        ancestors = collect_ancestor_summaries(self.tag_tree, target_mg.graph_id)
        # Sort by depth ascending (higher-level first for bottom-up)
        ancestors.sort(key=lambda x: x[0])

        for level, summary in ancestors[1:]:  # skip level 0 (the graph itself)
            if progress_callback: progress_callback(f"⬆️ Bottom-Up Refinement (Level {level}) with LLM...")
            answer = _refine_answer(self.llm, question, answer, summary)
            refinement_log.append({"level": level, "answer": answer})

        return {
            "answer": answer,
            "target_graph": target_mg,
            "top_entities": top_entities,
            "triple_neighbours": triple_neighbours,
            "refinement_log": refinement_log,
        }

    # ------------------------------------------------------------------
    # Public API: load documents
    # ------------------------------------------------------------------

    def load_documents(
        self,
        user_text: str,
        paper_texts: list[str] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Build the full triple-graph from user documents (Layer 1) and
        optional medical paper texts (Layer 2).

        Returns a summary of what was built.
        """
        # --- Selective clearing ---
        self.meta_graphs.clear()
        self.tag_tree.clear()

        # Remove only L1 nodes from NetworkX (keep L2 and L3)
        nodes_to_remove = [
            n for n, d in self.nx_graph.nodes(data=True)
            if d.get("entity") and d.get("entity").layer == 1
        ]
        self.nx_graph.remove_nodes_from(nodes_to_remove)

        # We NO LONGER rebuild L3 here; it persists.
        # But we ensure L1 and L2 nodes are in nx_graph for linking
        # (L3 is handled via Neo4j directly during U-Retrieval)
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # --- Layer 1: User documents ---
        _progress("⚙️  Chunking user documents…")
        chunks = self._semantic_chunks(user_text)
        
        import concurrent.futures
        
        all_entities_batch = []
        all_rels_batch = []
        
        def process_chunk(idx, chunk):
            gid = f"chunk_{idx}"
            return self._build_meta_graph(chunk, gid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    mg = future.result()
                    self.meta_graphs.append(mg)
                    all_entities_batch.extend(mg.entities)
                    all_rels_batch.extend(mg.relationships)
                    
                    # Safely apply to NetworkX on the main thread to avoid thread deadlocks
                    for e in mg.entities:
                        self.nx_graph.add_node(e.name, entity=e)
                    for r in mg.relationships:
                        self.nx_graph.add_edge(r.source, r.target, relation=r.relation)
                        
                    _progress(f"⚙️  Finished Layer-1 graph for chunk {idx+1}/{len(chunks)}…")
                except Exception as exc:
                    _progress(f"❌  Chunk {idx+1}/{len(chunks)} generated an exception: {exc}")

        # Batch Sync L1 to Neo4j
        if self.neo4j and all_entities_batch:
            _progress("⚙️  Syncing Layer-1 Entities to graph database…")
            self.neo4j.sync_entities(all_entities_batch, progress_callback=_progress)
            _progress("⚙️  Syncing Layer-1 Relationships to graph database…")
            self.neo4j.sync_relationships(all_rels_batch, progress_callback=_progress)
        # --- Layer 2: Medical paper entities (if provided) ---
        if paper_texts:
            for j, paper in enumerate(paper_texts):
                _progress(f"⚙️  Extracting Layer-2 entities from paper {j+1}/{len(paper_texts)}…")
                paper_entities = _extract_entities(self.llm, paper[:3000])
                for e in paper_entities:
                    e.layer = 2
                    e.embedding = self.emb.embed(e.content_text)
                self.repo_entities_l2.extend(paper_entities)
                
                # Sync L2 to Neo4j
                if self.neo4j:
                    self.neo4j.sync_entities(paper_entities)
                


        # --- Layer 3 is persistent and large; we only embed on-demand during linking ---
        # No longer scanning 500k+ nodes here every time.

        # --- Triple Linking ---
        _progress("🔗  Triple linking across graph layers (Vectorized)…")
        self._link_layers(skip_l3_embed=True, progress_callback=_progress)

        # --- Tag graphs ---
        _progress("🏷️  Tagging graphs in parallel…")
        self._tag_all_graphs(progress_callback=_progress)

        # --- Build tag hierarchy ---
        _progress("🌲  Building hierarchical tag tree (Vectorized)…")
        self._build_tag_tree(progress_callback=_progress)
        
        # Sync tag tree to Neo4j for persistent U-Retrieval
        if self.neo4j:
            _progress("⚙️  Syncing Tag Hierarchy to database…")
            self.neo4j.sync_tag_tree(self.tag_tree)

        _progress("✅  Triple graph construction complete.")

        total_l1_entities = sum(len(mg.entities) for mg in self.meta_graphs)
        total_l1_rels = sum(len(mg.relationships) for mg in self.meta_graphs)
        cross_layer_edges = sum(
            1 for _, _, d in self.nx_graph.edges(data=True)
            if d.get("relation") in ("the_reference_of", "the_definition_of")
        )

        return {
            "chunks": len(chunks),
            "meta_graphs": len(self.meta_graphs),
            "l1_entities": total_l1_entities,
            "l1_relationships": total_l1_rels,
            "l2_entities": len(self.repo_entities_l2),
            "l3_entities": self.neo4j.count_layer3() if self.neo4j else len(self.repo_entities_l3),
            "cross_layer_edges": cross_layer_edges,
            "tag_tree_roots": len(self.tag_tree),
            "total_graph_nodes": self.nx_graph.number_of_nodes(),
            "total_graph_edges": self.nx_graph.number_of_edges(),
        }

    def get_graph_stats(self) -> dict:
        return {
            "meta_graphs": len(self.meta_graphs),
            "l1_entities": sum(len(mg.entities) for mg in self.meta_graphs),
            "l2_entities": len(self.repo_entities_l2),
            "l3_entities": self.neo4j.count_layer3() if self.neo4j else len(self.repo_entities_l3),
            "total_nodes": self.nx_graph.number_of_nodes(),
            "total_edges": self.nx_graph.number_of_edges(),
        }

    def clear_all(self):
        """
        Clears all graph data globally from memory and Neo4j.
        """
        self.meta_graphs.clear()
        self.repo_entities_l2.clear()
        self.repo_entities_l3.clear()
        self.tag_tree.clear()
        self.nx_graph.clear()
        if self.neo4j:
            self.neo4j.clear_all_db()

    def clear_layer1(self):
        """
        Clears ONLY Layer 1 (user documents) from memory and Neo4j, preserving Layer 2 and Layer 3.
        """
        self.meta_graphs.clear()
        self.tag_tree.clear()
        
        # Remove only L1 nodes from NetworkX
        nodes_to_remove = [
            n for n, d in self.nx_graph.nodes(data=True)
            if d.get("entity") and d.get("entity").layer == 1
        ]
        self.nx_graph.remove_nodes_from(nodes_to_remove)
        
        if self.neo4j:
            self.neo4j.clear_layer1_db()

    def clear_all_relationships(self):
        """
        Clears all relationships (edges) from memory and Neo4j, but retains nodes.
        """
        self.nx_graph.clear_edges()
        if self.neo4j:
            self.neo4j.clear_all_relationships()

    def simulate_massive_vocab(self, num_nodes=500000, batch_size=10000, progress_callback=None):
        """
        Injects a massive set of dummy nodes into Layer 3 to simulate high capacity.
        Processed in batches to avoid OOM in python list and Neo4j.
        Computes Hugging Face embeddings in batches if configured.
        """
        self.clear_all()
        
        total_batches = (num_nodes // batch_size) + (1 if num_nodes % batch_size else 0)
        
        for batch_i in range(total_batches):
            if progress_callback:
                progress_callback(batch_i + 1, total_batches, f"Injecting batch {batch_i + 1}/{total_batches} ({num_nodes} nodes total) into L3")
            
            start_i = len(self.repo_entities_l3)
            end_i = min(start_i + batch_size, num_nodes)
            
            if start_i >= num_nodes:
                break
                
            batch_entities = []
            for i in range(start_i, end_i):
                e = Entity(
                    name=f"SimNode_{i}",
                    entity_type="Simulated",
                    context=f"Simulated entity {i} for stress test",
                    definition=f"Definition of simulated entity {i}",
                    layer=3
                )
                batch_entities.append(e)
            
            # Embed batch
            texts = [e.content_text for e in batch_entities]
            try:
                embeddings = self.emb.embed_batch(texts)
                for e, vec in zip(batch_entities, embeddings):
                    e.embedding = vec
            except Exception as e:
                print(f"Embedding batch failed: {e}")
            
            self.repo_entities_l3.extend(batch_entities)
            
            if self.neo4j:
                self.neo4j.sync_entities(batch_entities)
                # Ensure we add mock relationships sequentially so the graph structure exists.
                # E.g., SimNode_i -> SimNode_i+1
                rels = []
                for i in range(len(batch_entities) - 1):
                    rels.append(Relationship(
                        source=batch_entities[i].name,
                        target=batch_entities[i+1].name,
                        relation="SIMULATED_LINK"
                    ))
                self.neo4j.sync_relationships(rels)

    def seed_pubmed_literature(self, query: str, max_results: int = 5, progress_callback=None):
        """
        Fetches PubMed abstracts for a query, extracts entities and relationships
        via the LLM, and persists them as Layer 2 (L2_Entity) nodes in Neo4j.
        """
        if progress_callback:
            progress_callback(0.05, f"🔎 Searching PubMed for: {query}")

        abstracts = self.pubmed.fetch_abstracts(query, max_results=max_results)
        if not abstracts:
            if progress_callback:
                progress_callback(1.0, "⚠️ No PubMed results found.")
            return {"articles": 0, "entities": 0, "relationships": 0}

        total_entities = 0
        total_rels = 0

        for idx, abstract in enumerate(abstracts):
            pct = (idx + 1) / len(abstracts)
            if progress_callback:
                progress_callback(pct * 0.9, f"📄 Processing article {idx+1}/{len(abstracts)}...")

            # Extract entities and relationships from abstract text via LLM
            entities = _extract_entities(self.llm, abstract)
            relationships = _extract_relationships(self.llm, abstract, entities)

            # Mark all as Layer 2 and embed
            for e in entities:
                e.layer = 2
                e.embedding = self.emb.embed(e.content_text)
                self.repo_entities_l2.append(e)
                self.nx_graph.add_node(e.name, entity=e)

            for r in relationships:
                self.nx_graph.add_edge(r.source, r.target, relation=r.relation)

            # Sync to Neo4j with L2_Entity label
            if self.neo4j and entities:
                self.neo4j.sync_entities(entities)
            if self.neo4j and relationships:
                self.neo4j.sync_relationships(relationships)

            total_entities += len(entities)
            total_rels += len(relationships)

        if progress_callback:
            progress_callback(1.0, f"✅ Done! {total_entities} entities, {total_rels} relationships from {len(abstracts)} articles.")

        return {"articles": len(abstracts), "entities": total_entities, "relationships": total_rels}

    def import_local_umls_relationships_dump(self, mrconso_path: str, mrrel_path: str, progress_callback=None):
        from umls_importer import load_umls_relationships_to_neo4j
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required to import UMLS dump.")
            
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return load_umls_relationships_to_neo4j(mrconso_path, mrrel_path, uri, user, pwd, progress_callback)

    def import_local_umls_dump(self, mrconso_path: str, mrsty_path: str, progress_callback=None):
        from umls_importer import load_umls_to_neo4j
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required to import UMLS dump.")
            
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return load_umls_to_neo4j(mrconso_path, mrsty_path, uri, user, pwd, progress_callback)

    def bulk_import_pubmed(self, keywords=None, max_per_keyword=50, progress_callback=None):
        from pubmed_importer import bulk_fetch_pubmed, PUBMED_KEYWORDS
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required.")
        
        if keywords is None:
            keywords = PUBMED_KEYWORDS
            
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return bulk_fetch_pubmed(keywords, max_per_keyword, uri, user, pwd, progress_callback)

    def link_cross_layers_gpu(self, threshold=0.45, progress_callback=None):
        from cross_layer_linker import link_layers_gpu
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required.")
        
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return link_layers_gpu(uri, user, pwd, threshold=threshold, progress_callback=progress_callback)
