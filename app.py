"""
MedGraphRAG – Streamlit Interface
Implements the triple-layer graph from Wu et al. (2024)
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from med_graph_rag import MedGraphRAG, MEDICAL_TAGS

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="MedGraphRAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #58a6ff !important;
}

.layer-card {
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    background: #161b22;
}

.layer-1 { border-left: 4px solid #58a6ff; }
.layer-2 { border-left: 4px solid #3fb950; }
.layer-3 { border-left: 4px solid #f78166; }

.entity-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}

.chip-l1 { background: #1f3a5f; color: #58a6ff; }
.chip-l2 { background: #1a3d2b; color: #3fb950; }
.chip-l3 { background: #3d1f1a; color: #f78166; }

.answer-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px;
    font-size: 0.95em;
    line-height: 1.7;
}

.metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}

.metric-num {
    font-size: 1.8em;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
}

.metric-label {
    font-size: 0.75em;
    color: #8b949e;
    margin-top: 2px;
}

.tag-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    background: #21262d;
    border: 1px solid #30363d;
    font-size: 0.72em;
    font-family: 'IBM Plex Mono', monospace;
    color: #79c0ff;
    margin: 2px;
}

.relation-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7em;
    color: #d2a8ff;
    background: #2d1f4e;
    padding: 1px 6px;
    border-radius: 3px;
}

.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: #388bfd !important;
}

.sidebar-section {
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px;
    margin: 8px 0;
    background: #0d1117;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Init
# -------------------------------------------------------------------------

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️  Set OPENAI_API_KEY in your .env file.")
    st.stop()

if "med_rag" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    st.session_state.med_rag = MedGraphRAG(llm=llm, embedder=embedder)
    st.session_state.build_stats = None
    st.session_state.last_result = None
    st.session_state.build_log = []

rag: MedGraphRAG = st.session_state.med_rag

# -------------------------------------------------------------------------
# Sidebar – document loading
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧬 MedGraphRAG")
    st.caption("Triple Graph Construction + U-Retrieval")

    st.markdown("---")
    st.markdown("### 📄 Layer 1 – User Documents")

    input_mode = st.radio("Input mode", ["URL", "Paste text"], label_visibility="collapsed")

    user_text = ""
    if input_mode == "URL":
        url = st.text_input("Document URL", placeholder="https://...")
        if url and st.button("Fetch URL"):
            with st.spinner("Loading…"):
                try:
                    docs = WebBaseLoader(url).load()
                    user_text = "\n\n".join(d.page_content for d in docs)
                    st.session_state["fetched_text"] = user_text
                    st.success(f"Fetched {len(docs)} page(s).")
                except Exception as exc:
                    st.error(str(exc))
        user_text = st.session_state.get("fetched_text", "")
    else:
        user_text = st.text_area(
            "Paste clinical / medical text",
            height=160,
            placeholder="Paste EHR notes, discharge summaries, case studies…",
        )

    st.markdown("### 📚 Layer 2 – Medical Reference Texts *(optional)*")
    paper_text = st.text_area(
        "Paste reference paper/book excerpt(s)",
        height=100,
        placeholder="Paste one or more relevant paper abstracts or textbook passages…",
    )
    paper_texts = [paper_text.strip()] if paper_text.strip() else []

    st.markdown("### 📖 Layer 3 – Vocabulary")
    st.caption("Built-in UMLS-style controlled vocabulary (always active)")
    with st.expander("View vocabulary entries"):
        from med_graph_rag import BUILTIN_VOCAB
        for v in BUILTIN_VOCAB:
            st.markdown(
                f'<span class="entity-chip chip-l3">{v["name"]}</span> '
                f'<small style="color:#8b949e">{v["type"]}</small>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    build_btn = st.button("🔨 Build Triple Graph", use_container_width=True)

    if build_btn:
        if not user_text.strip():
            st.warning("Please provide Layer-1 document text.")
        else:
            log_placeholder = st.empty()
            st.session_state.build_log = []

            def log(msg: str):
                st.session_state.build_log.append(msg)
                log_placeholder.markdown(
                    "\n\n".join(st.session_state.build_log[-5:])
                )

            with st.spinner("Building triple graph…"):
                stats = rag.load_documents(
                    user_text=user_text,
                    paper_texts=paper_texts if paper_texts else None,
                    progress_callback=log,
                )
            st.session_state.build_stats = stats
            st.session_state.last_result = None
            st.rerun()

    if st.session_state.build_stats:
        s = st.session_state.build_stats
        st.success("Graph built ✓")
        st.markdown(f"""
<div class="sidebar-section">
<small>
🔷 <b>L1</b> {s['l1_entities']} entities · {s['l1_relationships']} relations · {s['meta_graphs']} subgraphs<br>
🟢 <b>L2</b> {s['l2_entities']} reference entities<br>
🔴 <b>L3</b> {s['l3_entities']} vocab entries<br>
🔗 {s['cross_layer_edges']} cross-layer links<br>
📊 {s['total_graph_nodes']} nodes · {s['total_graph_edges']} edges total
</small>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Main area
# -------------------------------------------------------------------------

st.markdown("# MedGraphRAG")
st.markdown(
    "Evidence-based medical QA via **Triple Graph Construction** "
    "(RAG data → Med Papers → UMLS Vocab) and **U-Retrieval** "
    "(top-down tag indexing + bottom-up refinement)."
)

# Tabs
tab_query, tab_graph, tab_architecture = st.tabs(["🔍 Query", "🕸️ Graph Explorer", "📐 Architecture"])

# ==============================
# Tab 1 – Query
# ==============================

with tab_query:
    if not st.session_state.build_stats:
        st.info("👈  Load documents and build the triple graph first.")
    else:
        question = st.text_area(
            "Medical question",
            height=80,
            placeholder="e.g. What medication adjustments should be made for a patient with COPD and heart failure?",
        )
        col_btn, col_lvl = st.columns([2, 3])
        with col_btn:
            run_query = st.button("🔍 Run U-Retrieval", use_container_width=True)

        if run_query and question.strip():
            with st.spinner("Running U-Retrieval…"):
                result = rag.query(question)
                st.session_state.last_result = result

        if st.session_state.last_result:
            res = st.session_state.last_result
            tg = res["target_graph"]

            # Answer
            st.markdown("### 💬 Answer")
            st.markdown(
                f'<div class="answer-box">{res["answer"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            col_left, col_right = st.columns(2)

            # Left – retrieved entities
            with col_left:
                st.markdown("#### 🔷 Layer-1 Entities Retrieved")
                for e in res["top_entities"]:
                    st.markdown(
                        f'<div class="layer-card layer-1">'
                        f'<b>{e.name}</b> '
                        f'<span class="entity-chip chip-l1">{e.entity_type}</span><br>'
                        f'<small style="color:#8b949e">{e.context[:120]}…</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 🟢🔴 Triple Neighbours (L2 + L3)")
                if res["triple_neighbours"]:
                    for e in res["triple_neighbours"]:
                        cls = "chip-l2" if e.layer == 2 else "chip-l3"
                        layer_cls = "layer-2" if e.layer == 2 else "layer-3"
                        label = "L2 Ref" if e.layer == 2 else "L3 Def"
                        defn = (
                            f'<br><small style="color:#f78166">Definition: {e.definition[:120]}</small>'
                            if e.definition else ""
                        )
                        st.markdown(
                            f'<div class="layer-card {layer_cls}">'
                            f'<b>{e.name}</b> '
                            f'<span class="entity-chip {cls}">{label} · {e.entity_type}</span><br>'
                            f'<small style="color:#8b949e">{e.context[:120]}</small>'
                            f'{defn}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No cross-layer neighbours found for these entities.")

            # Right – target graph + refinement
            with col_right:
                if tg:
                    st.markdown(f"#### 🎯 Target Meta-MedGraph: `{tg.graph_id}`")

                    # Tags
                    if tg.tag_summary:
                        st.markdown("**Tag Summary:**")
                        tag_html = "".join(
                            f'<span class="tag-pill">{k}: {v[:40]}</span>'
                            for k, v in tg.tag_summary.items()
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)

                    # Relationships
                    st.markdown("**Graph Relationships:**")
                    if tg.relationships:
                        for r in tg.relationships[:12]:
                            st.markdown(
                                f'`{r.source}` '
                                f'<span class="relation-badge">{r.relation}</span>'
                                f' `{r.target}`',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No relationships in this subgraph.")

                # Refinement log
                if len(res["refinement_log"]) > 1:
                    st.markdown("#### 🔄 Bottom-up Refinement")
                    st.caption(f"{len(res['refinement_log'])} refinement pass(es)")
                    with st.expander("View refinement steps"):
                        for step in res["refinement_log"]:
                            st.markdown(f"**Level {step['level']}**")
                            st.markdown(
                                f'<div class="answer-box" style="font-size:0.85em">'
                                f'{step["answer"][:600]}…</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("---")

# ==============================
# Tab 2 – Graph Explorer
# ==============================

with tab_graph:
    if not st.session_state.build_stats:
        st.info("Build the graph first.")
    else:
        stats = rag.get_graph_stats()

        # Metrics row
        cols = st.columns(6)
        metrics = [
            ("Subgraphs", stats["meta_graphs"]),
            ("L1 Entities", stats["l1_entities"]),
            ("L2 Refs", stats["l2_entities"]),
            ("L3 Vocab", stats["l3_entities"]),
            ("Total Nodes", stats["total_nodes"]),
            ("Total Edges", stats["total_edges"]),
        ]
        for col, (label, val) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-num">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Show each MetaMedGraph
        for mg in rag.meta_graphs:
            with st.expander(f"📊 {mg.graph_id}  ({len(mg.entities)} entities, {len(mg.relationships)} rels)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Entities (Layer 1)**")
                    for e in mg.entities:
                        st.markdown(
                            f'<span class="entity-chip chip-l1">{e.name}</span>'
                            f'<small style="color:#8b949e"> {e.entity_type}</small>',
                            unsafe_allow_html=True,
                        )
                    # Cross-layer neighbours
                    nx_g = rag.nx_graph
                    l2_links = [
                        (v, d) for _, v, d in nx_g.out_edges(
                            [e.name for e in mg.entities], data=True
                        )
                        if d.get("relation") == "the_reference_of"
                    ]
                    l3_links = [
                        (v, d) for _, v, d in nx_g.out_edges(
                            [e.name for e in mg.entities], data=True
                        )
                        if d.get("relation") == "the_definition_of"
                    ]
                    if l2_links:
                        st.markdown("**→ Layer-2 References**")
                        for tgt, d in l2_links[:6]:
                            st.markdown(
                                f'<span class="entity-chip chip-l2">{tgt}</span>'
                                f'<small style="color:#8b949e"> sim={d.get("similarity", 0):.2f}</small>',
                                unsafe_allow_html=True,
                            )
                    if l3_links:
                        st.markdown("**→ Layer-3 Definitions**")
                        for tgt, d in l3_links[:6]:
                            st.markdown(
                                f'<span class="entity-chip chip-l3">{tgt}</span>'
                                f'<small style="color:#8b949e"> sim={d.get("similarity", 0):.2f}</small>',
                                unsafe_allow_html=True,
                            )

                with c2:
                    st.markdown("**Relationships**")
                    for r in mg.relationships:
                        st.markdown(
                            f'`{r.source}` '
                            f'<span class="relation-badge">{r.relation}</span> '
                            f'`{r.target}`',
                            unsafe_allow_html=True,
                        )
                    st.markdown("**Tag Summary**")
                    if mg.tag_summary:
                        for k, v in mg.tag_summary.items():
                            st.markdown(
                                f'<span class="tag-pill">{k}</span> {v}',
                                unsafe_allow_html=True,
                            )

        st.markdown("---")
        st.markdown("### 🌲 Tag Tree (Hierarchical Clusters)")

        def render_tree(nodes, depth=0):
            for node in nodes:
                indent = "&nbsp;" * (depth * 4)
                ids_str = ", ".join(node["ids"])
                tags_str = " ".join(
                    f'<span class="tag-pill">{k}</span>'
                    for k in node["tags"].keys()
                )
                st.markdown(
                    f'{indent}📁 <b>{ids_str}</b> {tags_str}',
                    unsafe_allow_html=True,
                )
                if node["children"]:
                    render_tree(node["children"], depth + 1)

        if rag.tag_tree:
            render_tree(rag.tag_tree)
        else:
            st.caption("No tag tree yet.")

# ==============================
# Tab 3 – Architecture
# ==============================

with tab_architecture:
    st.markdown("### MedGraphRAG – Triple Graph Construction")
    st.markdown("""
This implementation follows the paper architecture exactly:

**Graph Construction (6 steps)**

| Step | What happens |
|------|-------------|
| 1. Semantic Chunking | Documents split by topic using `RecursiveCharacterTextSplitter` with overlap |
| 2. Entity Extraction | LLM extracts entities with `{name, type (UMLS semantic type), context}` per chunk |
| 3. Triple Linking | Layer-1 entities linked to Layer-2 (papers) and Layer-3 (vocab) via cosine similarity ≥ δᵣ |
| 4. Relationship Linking | LLM generates directed relationships between entities in each chunk |
| 5. Tag Graphs | Each Meta-MedGraph tagged with predefined medical categories |
| 6. Hierarchical Clustering | Agglomerative clustering over tag embeddings builds retrieval tree |

**U-Retrieval**
- **Top-down**: Query → generate tags → traverse tag tree layer-by-layer → find target Meta-MedGraph Gmt
- **Bottom-up**: Fetch top-N entities + triple-neighbours from Gmt → initial answer → refine upward through ancestor tag summaries

**Three Graph Layers**
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div class="layer-card layer-1">
<b style="color:#58a6ff">Layer 1 – RAG Graph</b><br>
<small>User documents (EHR, clinical notes, discharge summaries)</small><br><br>
• Semantic chunking<br>
• Entity extraction (name, UMLS type, context)<br>
• Intra-chunk relationship linking<br>
• One Meta-MedGraph per chunk<br>
• Tagged with medical categories
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="layer-card layer-2">
<b style="color:#3fb950">Layer 2 – Repository Graph</b><br>
<small>Medical papers, textbooks (e.g. MedC-K corpus)</small><br><br>
• Same entity extraction as Layer 1<br>
• Linked to Layer 1 via cosine sim<br>
• Edge type: <code>the_reference_of</code><br>
• Provides source citations<br>
• Supports evidence-based responses
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="layer-card layer-3">
<b style="color:#f78166">Layer 3 – Vocabulary Graph</b><br>
<small>Controlled vocabulary (UMLS, medical dictionaries)</small><br><br>
• Pre-built from authoritative vocab<br>
• Linked to Layer 2 via cosine sim<br>
• Edge type: <code>the_definition_of</code><br>
• Provides formal definitions<br>
• Terminological clarification
</div>
""", unsafe_allow_html=True)

    st.markdown("""
```
User document ──▶ [Chunk₁ Graph] ──the_reference_of──▶ [Med Paper Entity] ──the_definition_of──▶ [UMLS Vocab]
                  [Chunk₂ Graph] ──the_reference_of──▶ [Med Paper Entity] ──the_definition_of──▶ [UMLS Vocab]
                       ▲
                  Triple: [RAG entity, source, definition]
```
    """)
    st.markdown("""
> **Key insight from the paper**: Unlike standard GraphRAG, entities in MedGraphRAG are directly linked 
> to their references and definitions in separate graph layers. This allows precise evidence retrieval 
> without mixing user data and repository data in the same layer.
""")
