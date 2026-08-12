import json
import re
import textwrap
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from scipy.spatial.distance import cosine

from medgraphrag.models import Entity, Relationship, MetaMedGraph, UMLS_SEMANTIC_TYPES, MEDICAL_TAGS

from typing import Any

class EmbeddingStore:
    def __init__(self, embedder: Any):
        self._emb = embedder
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self._emb.embed_query(text)
            self._cache[text] = np.array(vec, dtype=np.float32)
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        uncached = [t for t in set(texts) if t and t not in self._cache]
        if uncached:
            if hasattr(self._emb, 'embed_documents'):
                vecs = self._emb.embed_documents(uncached)
                for t, v in zip(uncached, vecs):
                    self._cache[t] = np.array(v, dtype=np.float32)
            else:
                for t in uncached:
                    self.embed(t)
        return [self._cache[t] if t else None for t in texts]


    def similarity(self, a: str | np.ndarray, b: str | np.ndarray) -> float:
        va = a if isinstance(a, np.ndarray) else self.embed(a)
        vb = b if isinstance(b, np.ndarray) else self.embed(b)
        if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
            return 0.0
        return float(1.0 - cosine(va, vb))


def _call_llm_json(llm: ChatOpenAI, prompt: str) -> dict | list:
    """Call LLM and parse JSON from the response."""
    print(f"\n[LLM Debug] Invoking LLM with prompt ({len(prompt)} chars)...")
    resp = llm.invoke(prompt)
    raw = resp.content.strip()
    print(f"[LLM Debug] Raw LLM Response:\n{raw}\n{'-'*40}")
    
    # Try to find a JSON block inside markdown fences first
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        block = fence_match.group(1).strip()
        try:
            parsed = json.loads(block)
            print(f"[LLM Debug] Successfully parsed from markdown fence: {type(parsed)}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[LLM Debug] Fence JSON decode error: {e}")
            raw = block

    # Strip any leading/trailing fences if still present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        print(f"[LLM Debug] Successfully parsed full raw string: {type(parsed)}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"[LLM Debug] Full string JSON decode error: {e}. Attempting robust substring extraction...")
        # Robustly extract the outermost JSON array or object
        # Look for the first occurrence of '[' or '{'
        arr_idx = raw.find('[')
        obj_idx = raw.find('{')
        
        if arr_idx != -1 and (obj_idx == -1 or arr_idx < obj_idx):
            # Array comes first or is the only container
            m = re.search(r"(\[.*\])", raw, re.DOTALL)
            if m:
                try: 
                    parsed = json.loads(m.group(1))
                    print(f"[LLM Debug] Successfully extracted array: {type(parsed)}")
                    return parsed
                except Exception as ex: 
                    print(f"[LLM Debug] Substring array parse failed: {ex}")
        elif obj_idx != -1:
            # Object comes first
            m = re.search(r"(\{.*\})", raw, re.DOTALL)
            if m:
                try: 
                    parsed = json.loads(m.group(1))
                    print(f"[LLM Debug] Successfully extracted object: {type(parsed)}")
                    return parsed
                except Exception as ex:
                    print(f"[LLM Debug] Substring object parse failed: {ex}")
        print("[LLM Debug] Failed to extract any valid JSON structure. Returning empty dict.")
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
    if isinstance(result, dict):
        # Unwrap if LLM returned {"entities": [...]} or similar
        for v in result.values():
            if isinstance(v, list):
                result = v
                break
        else:
            if "name" in result:
                result = [result]

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
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                result = v
                break
        else:
            if "source" in result and "target" in result:
                result = [result]

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
    
    # Restrict to top 20 relationships to prevent context window bloat
    for rel in graph.relationships[:20]:
        if rel.source in all_ents or rel.target in all_ents:
            src_e = all_ents.get(rel.source)
            tgt_e = all_ents.get(rel.target)
            src_ctx = f" [{src_e.context[:80]}]" if src_e else ""
            tgt_ctx = f" [{tgt_e.context[:80]}]" if tgt_e else ""
            
            src_def = ""
            tgt_def = ""
            if src_e and src_e.layer == 3 and src_e.definition:
                src_def = f" (Definition: {src_e.definition[:100]}...)"
            if tgt_e and tgt_e.layer == 3 and tgt_e.definition:
                tgt_def = f" (Definition: {tgt_e.definition[:100]}...)"
            graph_text += (
                f"{rel.source}{src_ctx}{src_def} "
                f"--[{rel.relation}]--> "
                f"{rel.target}{tgt_ctx}{tgt_def}\n"
            )

    entity_detail_parts = []
    for e in (top_entities + top_k_neighbors):
        ctx_raw = e.context or ""
        ctx_trun = (ctx_raw[:300] + "...") if len(ctx_raw) > 300 else ctx_raw
        base = f"• {e.name} ({e.entity_type}, Layer {e.layer}): {ctx_trun}"
        if e.definition:
            def_trun = (e.definition[:300] + "...") if len(e.definition) > 300 else e.definition
            base += f"\n  Source/Definition: {def_trun}"
        entity_detail_parts.append(base)
    
    entity_detail = "\n".join(entity_detail_parts)

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
