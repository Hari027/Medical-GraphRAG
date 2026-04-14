"""
MedGraphRAG — Triple Graph Construction + U-Retrieval
Based on: "Medical Graph RAG: Towards Safe Medical Large Language Model
           via Graph Retrieval-Augmented Generation" (Wu et al., 2024)

Architecture
------------
Layer 1  (RAG Graph / Meta-MedGraph Gm)
  Raw user documents → semantic chunks → entity extraction → relationship linking
  Each entity:  {name, type (UMLS semantic type), context}

Layer 2  (Repository Graph – Med Papers/Books)
  Medical reference texts → same entity+relationship extraction
  Linked to Layer-1 entities via cosine-similarity ("the_reference_of")

Layer 3  (Repository Graph – Medical Vocabularies / UMLS-style)
  Hardcoded controlled-vocabulary entries (UMLS concepts)
  Linked to Layer-2 entities via cosine-similarity ("the_definition_of")
  Each entity also carries a formal definition string

Retrieval: U-Retrieval
  Top-down: generate tags for query → match tags layer-by-layer to find target
            Meta-MedGraph Gmt
  Bottom-up: fetch top-N entities + triple-neighbours from Gmt, generate
             initial response, then refine upward through higher tag layers
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.spatial.distance import cosine


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

UMLS_SEMANTIC_TYPES = [
    "Disease", "Finding", "Pharmacologic Substance", "Clinical Drug",
    "Therapeutic or Preventive Procedure", "Diagnostic Procedure",
    "Body Part, Organ, or Organ Component", "Pathologic Function",
    "Sign or Symptom", "Organism", "Anatomy", "Hormone",
    "Biologically Active Substance", "Cell", "Gene or Genome",
    "Injury or Poisoning", "Mental or Behavioral Dysfunction",
    "Neoplastic Process", "Virus", "Bacterium", "Other",
]

MEDICAL_TAGS = [
    "SYMPTOMS", "PATIENT_HISTORY", "BODY_FUNCTION", "MEDICATION",
    "DIAGNOSIS", "PROCEDURE", "LAB_RESULTS", "PROGNOSIS",
    "RISK_FACTORS", "TREATMENT_PLAN",
]

# Minimal built-in vocabulary (stands in for UMLS graph, Layer 3)
BUILTIN_VOCAB: list[dict] = [
    {"name": "Hypertension", "type": "Disease",
     "definition": "Persistently elevated arterial blood pressure (≥130/80 mmHg)."},
    {"name": "Myocardial Infarction", "type": "Disease",
     "definition": "Irreversible necrosis of heart muscle secondary to ischaemia."},
    {"name": "Beta-blocker", "type": "Pharmacologic Substance",
     "definition": "Drug class that blocks β-adrenergic receptors, reducing heart rate and BP."},
    {"name": "ACE Inhibitor", "type": "Pharmacologic Substance",
     "definition": "Inhibits angiotensin-converting enzyme, used for HTN and heart failure."},
    {"name": "Metformin", "type": "Clinical Drug",
     "definition": "First-line oral biguanide for type-2 diabetes; reduces hepatic glucose output."},
    {"name": "Type 2 Diabetes Mellitus", "type": "Disease",
     "definition": "Chronic metabolic disorder characterised by insulin resistance and hyperglycaemia."},
    {"name": "COPD", "type": "Disease",
     "definition": "Chronic obstructive pulmonary disease; persistent airflow limitation."},
    {"name": "Bronchodilator", "type": "Pharmacologic Substance",
     "definition": "Agent that relaxes bronchial smooth muscle to widen airways."},
    {"name": "Heart Failure", "type": "Disease",
     "definition": "Inability of the heart to pump sufficient blood to meet the body's needs."},
    {"name": "Atrial Fibrillation", "type": "Disease",
     "definition": "Irregular rapid atrial rhythm causing ineffective atrial contraction."},
    {"name": "Statin", "type": "Pharmacologic Substance",
     "definition": "HMG-CoA reductase inhibitor that reduces LDL cholesterol levels."},
    {"name": "Sepsis", "type": "Disease",
     "definition": "Life-threatening organ dysfunction caused by dysregulated host response to infection."},
    {"name": "Pneumonia", "type": "Disease",
     "definition": "Infection causing inflammation of alveoli, often with consolidation."},
    {"name": "Echocardiography", "type": "Diagnostic Procedure",
     "definition": "Ultrasound imaging of cardiac structure and function."},
    {"name": "Creatinine", "type": "Biologically Active Substance",
     "definition": "Serum marker of renal filtration; elevated in kidney disease."},
]


@dataclass
class Entity:
    name: str
    entity_type: str      # UMLS semantic type
    context: str          # A few sentences of contextual description
    definition: str = ""  # Used in Layer-3 only
    layer: int = 1        # 1 = RAG, 2 = Med Papers, 3 = Vocabulary
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def content_text(self) -> str:
        return f"name: {self.name}; type: {self.entity_type}; context: {self.context}"


@dataclass
class Relationship:
    source: str   # entity name
    relation: str
    target: str   # entity name


@dataclass
class MetaMedGraph:
    """A graph built from one semantic chunk (or a merged cluster of chunks)."""
    graph_id: str
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    tag_summary: dict[str, str] = field(default_factory=dict)   # tag → description
    source_text: str = ""

    def entity_names(self) -> list[str]:
        return [e.name for e in self.entities]


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

class EmbeddingStore:
    def __init__(self, embedder: OpenAIEmbeddings):
        self._emb = embedder
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self._emb.embed_query(text)
            self._cache[text] = np.array(vec, dtype=np.float32)
        return self._cache[text]

    def similarity(self, a: str | np.ndarray, b: str | np.ndarray) -> float:
        va = a if isinstance(a, np.ndarray) else self.embed(a)
        vb = b if isinstance(b, np.ndarray) else self.embed(b)
        if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
            return 0.0
        return float(1.0 - cosine(va, vb))


# ---------------------------------------------------------------------------
# LLM-powered helpers
# ---------------------------------------------------------------------------

def _call_llm_json(llm: ChatOpenAI, prompt: str) -> dict | list:
    """Call LLM and parse JSON from the response."""
    resp = llm.invoke(prompt)
    raw = resp.content.strip()
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # try to extract first JSON object/array
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        return {}


def _extract_entities(llm: ChatOpenAI, chunk_text: str) -> list[Entity]:
    semantic_types_str = ", ".join(UMLS_SEMANTIC_TYPES)
    prompt = textwrap.dedent(f"""
        You are a biomedical NLP expert. Extract all medically relevant entities from the text below.

        For each entity return a JSON object with keys:
          - "name": the entity name (string)
          - "type": one of [{semantic_types_str}]
          - "context": 1-2 sentence contextual description based on the text

        Return ONLY a JSON array of these objects, nothing else. No markdown, no explanation.

        TEXT:
        {chunk_text}
    """).strip()
    result = _call_llm_json(llm, prompt)
    entities = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "name" in item:
                entities.append(Entity(
                    name=item.get("name", "Unknown"),
                    entity_type=item.get("type", "Other"),
                    context=item.get("context", ""),
                    layer=1,
                ))
    return entities


def _extract_relationships(
    llm: ChatOpenAI, chunk_text: str, entities: list[Entity]
) -> list[Relationship]:
    entity_names = [e.name for e in entities]
    if len(entity_names) < 2:
        return []
    prompt = textwrap.dedent(f"""
        You are a biomedical knowledge graph expert.
        Given the entities: {entity_names}
        And the source text below, identify meaningful relationships BETWEEN those entities.

        Return ONLY a JSON array where each element has:
          - "source": name of source entity (must be from the list above)
          - "relation": a short relation phrase (e.g. "treats", "causes", "is_symptom_of")
          - "target": name of target entity (must be from the list above)

        Only include relationships explicitly or strongly implied by the text.
        Return ONLY the JSON array, no markdown, no explanation.

        TEXT:
        {chunk_text}
    """).strip()
    result = _call_llm_json(llm, prompt)
    rels = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "source" in item and "target" in item:
                rels.append(Relationship(
                    source=item.get("source", ""),
                    relation=item.get("relation", "related_to"),
                    target=item.get("target", ""),
                ))
    return rels


def _tag_graph(llm: ChatOpenAI, graph: MetaMedGraph) -> dict[str, str]:
    entity_texts = "\n".join(
        f"- {e.name} ({e.entity_type}): {e.context}" for e in graph.entities
    )
    tags_str = ", ".join(MEDICAL_TAGS)
    prompt = textwrap.dedent(f"""
        You are a medical text summarizer. Summarize the following medical entities using
        these structured tag categories: {tags_str}

        For each relevant tag, provide a short phrase describing what is present.
        Return ONLY a JSON object where keys are tag names and values are short descriptions.
        Omit tags that are not relevant. No markdown, no explanation.

        ENTITIES:
        {entity_texts}
    """).strip()
    result = _call_llm_json(llm, prompt)
    if isinstance(result, dict):
        return {k: str(v) for k, v in result.items()}
    return {}


def _generate_answer(
    llm: ChatOpenAI,
    question: str,
    graph: MetaMedGraph,
    top_entities: list[Entity],
    top_k_neighbors: list[Entity],
) -> str:
    graph_text = ""
    all_ents = {e.name: e for e in top_entities + top_k_neighbors}
    for rel in graph.relationships:
        if rel.source in all_ents or rel.target in all_ents:
            src_e = all_ents.get(rel.source)
            tgt_e = all_ents.get(rel.target)
            src_ctx = f" [{src_e.context[:80]}]" if src_e else ""
            tgt_ctx = f" [{tgt_e.context[:80]}]" if tgt_e else ""
            # Include definitions and sources from layers 2 & 3
            src_def = ""
            tgt_def = ""
            if src_e and src_e.layer == 3:
                src_def = f" (Definition: {src_e.definition})"
            if tgt_e and tgt_e.layer == 3:
                tgt_def = f" (Definition: {tgt_e.definition})"
            graph_text += (
                f"{rel.source}{src_ctx}{src_def} "
                f"--[{rel.relation}]--> "
                f"{rel.target}{tgt_ctx}{tgt_def}\n"
            )

    entity_detail = "\n".join(
        f"• {e.name} ({e.entity_type}, Layer {e.layer}): {e.context}"
        + (f"\n  Source/Definition: {e.definition}" if e.definition else "")
        for e in (top_entities + top_k_neighbors)
    )

    prompt = textwrap.dedent(f"""
        You are a medical expert assistant generating evidence-based responses.

        QUESTION: {question}

        RELEVANT ENTITIES (with source and definition references):
        {entity_detail}

        GRAPH RELATIONSHIPS:
        {graph_text if graph_text else "(no direct relationships found)"}

        Using the entities and graph above, answer the question in detail.
        Cite specific entities by name. If definitions are provided use them to
        clarify terminology. Be precise and evidence-based.
    """).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip()


def _refine_answer(
    llm: ChatOpenAI, question: str, prev_response: str, summary: dict[str, str]
) -> str:
    summary_text = "\n".join(f"  {k}: {v}" for k, v in summary.items())
    prompt = textwrap.dedent(f"""
        You are a medical expert assistant. Refine the response below using the
        higher-level summary context provided.

        QUESTION: {question}

        PREVIOUS RESPONSE:
        {prev_response}

        ADDITIONAL CONTEXT (higher-level summary):
        {summary_text}

        Adjust and improve the response, ensuring completeness and accuracy.
        Preserve all cited evidence from the previous response and add any new
        relevant information from the additional context.
    """).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip()


# ---------------------------------------------------------------------------
# Core MedGraphRAG system
# ---------------------------------------------------------------------------

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
    MAX_TAG_LAYERS       = 6      # max U-Retrieval layers

    def __init__(self, llm: ChatOpenAI, embedder: OpenAIEmbeddings):
        self.llm = llm
        self.emb = EmbeddingStore(embedder)

        # Layer 1 – user RAG graphs
        self.meta_graphs: list[MetaMedGraph] = []

        # Layer 2 – repository (paper) entities
        self.repo_entities_l2: list[Entity] = []

        # Layer 3 – vocabulary entities  (pre-seeded from BUILTIN_VOCAB)
        self.repo_entities_l3: list[Entity] = self._build_vocab_layer()

        # Hierarchical tag tree  [{graph_id: str, tags: dict, children: list}]
        self.tag_tree: list[dict] = []

        # NetworkX graph for full triple structure
        self.nx_graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Layer 3 – Vocabulary (UMLS-style), built once
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

        # Add to NetworkX
        for e in g.entities:
            self.nx_graph.add_node(e.name, entity=e)
        for r in g.relationships:
            self.nx_graph.add_edge(r.source, r.target, relation=r.relation)

        return g

    # ------------------------------------------------------------------
    # Step 3 – Triple Linking  (L1 → L2 → L3)
    # ------------------------------------------------------------------

    def _embed_all_layers(self):
        """Ensure all layer-2 and layer-3 entities are embedded."""
        for e in self.repo_entities_l2 + self.repo_entities_l3:
            if e.embedding is None:
                e.embedding = self.emb.embed(e.content_text)

    def _link_layers(self):
        """
        For every Layer-1 entity, find sufficiently similar Layer-2 entities
        and add 'the_reference_of' edges.  Then for each Layer-2 entity find
        Layer-3 entities and add 'the_definition_of' edges.
        """
        self._embed_all_layers()

        # L1 → L2
        for mg in self.meta_graphs:
            for e1 in mg.entities:
                if e1.embedding is None:
                    continue
                for e2 in self.repo_entities_l2:
                    sim = self.emb.similarity(e1.embedding, e2.embedding)
                    if sim >= self.SIMILARITY_THRESHOLD:
                        if not self.nx_graph.has_node(e2.name):
                            self.nx_graph.add_node(e2.name, entity=e2)
                        self.nx_graph.add_edge(
                            e1.name, e2.name,
                            relation="the_reference_of",
                            similarity=sim,
                        )

        # L2 → L3
        for e2 in self.repo_entities_l2:
            if e2.embedding is None:
                continue
            for e3 in self.repo_entities_l3:
                sim = self.emb.similarity(e2.embedding, e3.embedding)
                if sim >= self.SIMILARITY_THRESHOLD:
                    if not self.nx_graph.has_node(e3.name):
                        self.nx_graph.add_node(e3.name, entity=e3)
                    self.nx_graph.add_edge(
                        e2.name, e3.name,
                        relation="the_definition_of",
                        similarity=sim,
                    )

        # Also directly link L1 → L3 when there is no L2 (fallback)
        for mg in self.meta_graphs:
            for e1 in mg.entities:
                if e1.embedding is None:
                    continue
                for e3 in self.repo_entities_l3:
                    if self.nx_graph.has_edge(e1.name, e3.name):
                        continue
                    sim = self.emb.similarity(e1.embedding, e3.embedding)
                    if sim >= self.SIMILARITY_THRESHOLD:
                        if not self.nx_graph.has_node(e3.name):
                            self.nx_graph.add_node(e3.name, entity=e3)
                        self.nx_graph.add_edge(
                            e1.name, e3.name,
                            relation="the_definition_of",
                            similarity=sim,
                        )

    # ------------------------------------------------------------------
    # Step 5 – Tag the graphs  (hierarchical clustering)
    # ------------------------------------------------------------------

    def _tag_all_graphs(self):
        for mg in self.meta_graphs:
            mg.tag_summary = _tag_graph(self.llm, mg)

    def _build_tag_tree(self):
        """
        Agglomerative hierarchical clustering over tag embeddings.
        Produces a tree list used for top-down retrieval.
        """
        # Start: each graph is its own leaf node
        nodes = [
            {"ids": [mg.graph_id], "tags": mg.tag_summary, "children": []}
            for mg in self.meta_graphs
        ]

        for _layer in range(self.MAX_TAG_LAYERS):
            if len(nodes) <= 1:
                break

            # Compute pairwise tag similarities
            best_sim, best_i, best_j = -1.0, 0, 1
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    sim = self._tag_similarity(nodes[i]["tags"], nodes[j]["tags"])
                    if sim > best_sim:
                        best_sim, best_i, best_j = sim, i, j

            if best_sim < self.TAG_MERGE_THRESHOLD:
                break  # nothing more to merge

            # Merge nodes[best_i] and nodes[best_j]
            merged_tags = self._merge_tags(nodes[best_i]["tags"], nodes[best_j]["tags"])
            merged = {
                "ids": nodes[best_i]["ids"] + nodes[best_j]["ids"],
                "tags": merged_tags,
                "children": [nodes[best_i], nodes[best_j]],
            }
            new_nodes = [n for k, n in enumerate(nodes) if k not in (best_i, best_j)]
            new_nodes.append(merged)
            nodes = new_nodes

        self.tag_tree = nodes  # list of root-level nodes

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

    def _top_down_retrieve(self, query: str) -> MetaMedGraph | None:
        """
        Generate query tags, then traverse the tag tree top-down to find
        the most relevant Meta-MedGraph.
        """
        q_tags = _tag_graph(self.llm, MetaMedGraph(
            graph_id="query",
            entities=[Entity(name="query", entity_type="Other", context=query)],
        ))

        if not self.tag_tree:
            return self.meta_graphs[0] if self.meta_graphs else None

        # Traverse from root nodes
        current_nodes = self.tag_tree
        target_graph_id = None

        for _ in range(self.MAX_TAG_LAYERS):
            best_sim, best_node = -1.0, None
            for node in current_nodes:
                sim = self._tag_similarity(q_tags, node["tags"])
                if sim > best_sim:
                    best_sim, best_node = sim, node

            if best_node is None:
                break

            if not best_node["children"]:
                # Leaf: pick the graph
                target_graph_id = best_node["ids"][0]
                break
            else:
                current_nodes = best_node["children"]

        if target_graph_id is None and self.meta_graphs:
            target_graph_id = self.meta_graphs[0].graph_id

        for mg in self.meta_graphs:
            if mg.graph_id == target_graph_id:
                return mg
        return self.meta_graphs[0] if self.meta_graphs else None

    def _get_triple_neighbours(
        self, entity_name: str, k: int
    ) -> list[Entity]:
        """
        Return all entities within k hops across all three graph layers
        (following any edge type, including cross-layer links).
        """
        neighbours: list[Entity] = []
        try:
            # BFS up to k hops in both directions
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

    def query(self, question: str) -> dict:
        """
        Full U-Retrieval pipeline.
        Returns a dict with keys: answer, target_graph, top_entities,
        triple_neighbours, refinement_log.
        """
        if not self.meta_graphs:
            return {"answer": "No documents loaded.", "target_graph": None,
                    "top_entities": [], "triple_neighbours": [], "refinement_log": []}

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

        # Gather triple neighbours (cross-layer)
        triple_neighbours: list[Entity] = []
        seen = {e.name for e in top_entities}
        for e in top_entities:
            for nb in self._get_triple_neighbours(e.name, self.TOP_K_NEIGHBOURS):
                if nb.name not in seen:
                    seen.add(nb.name)
                    triple_neighbours.append(nb)

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
        self.meta_graphs.clear()
        self.repo_entities_l2.clear()
        self.nx_graph.clear()
        self.tag_tree.clear()
        # Rebuild L3 (always fresh from vocab)
        self.repo_entities_l3 = self._build_vocab_layer()

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # --- Layer 1: User documents ---
        _progress("⚙️  Chunking user documents…")
        chunks = self._semantic_chunks(user_text)

        for i, chunk in enumerate(chunks):
            _progress(f"⚙️  Building Layer-1 graph for chunk {i+1}/{len(chunks)}…")
            gid = f"chunk_{i}"
            mg = self._build_meta_graph(chunk, gid)
            self.meta_graphs.append(mg)

        # --- Layer 2: Medical paper entities (if provided) ---
        if paper_texts:
            for j, paper in enumerate(paper_texts):
                _progress(f"⚙️  Extracting Layer-2 entities from paper {j+1}/{len(paper_texts)}…")
                paper_entities = _extract_entities(self.llm, paper[:3000])
                for e in paper_entities:
                    e.layer = 2
                    e.embedding = self.emb.embed(e.content_text)
                self.repo_entities_l2.extend(paper_entities)

        # --- Layer 3 already built (BUILTIN_VOCAB) ---
        _progress("⚙️  Embedding Layer-3 vocabulary entities…")
        for e in self.repo_entities_l3:
            e.embedding = self.emb.embed(e.content_text)

        # --- Triple Linking ---
        _progress("🔗  Triple linking across graph layers…")
        self._link_layers()

        # --- Tag graphs ---
        _progress("🏷️  Tagging graphs…")
        self._tag_all_graphs()

        # --- Build tag hierarchy ---
        _progress("🌲  Building hierarchical tag tree…")
        self._build_tag_tree()

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
            "l3_entities": len(self.repo_entities_l3),
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
            "l3_entities": len(self.repo_entities_l3),
            "total_nodes": self.nx_graph.number_of_nodes(),
            "total_edges": self.nx_graph.number_of_edges(),
        }
