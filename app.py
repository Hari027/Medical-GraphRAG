import os
from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "MedGraphRAG/1.0"

import json
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from medical_terms import get_medical_terms
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

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
.neo4j-status {
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 0.8em;
    margin-bottom: 10px;
}
.status-online { background: #1a7f37; color: white; }
.status-offline { background: #cf222e; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Init
# -------------------------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY")
umls_api_key = os.environ.get("UMLS_API_KEY")
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_pass = os.environ.get("NEO4J_PASSWORD", "password")

if not api_key:
    st.error("⚠️  Set OPENAI_API_KEY in your .env file.")
    st.stop()

if "med_rag" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    neo4j_creds = {
        "uri": neo4j_uri,
        "user": neo4j_user,
        "password": neo4j_pass
    }
    st.session_state.med_rag = MedGraphRAG(
        llm=llm, 
        embedder=embedder, 
        umls_api_key=umls_api_key,
        neo4j_creds=neo4j_creds
    )
    st.session_state.build_stats = None
    st.session_state.last_result = None
    st.session_state.build_log = []

rag: MedGraphRAG = st.session_state.med_rag

# -------------------------------------------------------------------------
# Sidebar – document loading
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧬 MedGraphRAG")
    
    # Neo4j Status
    if st.session_state.med_rag.neo4j and st.session_state.med_rag.neo4j.driver:
        st.markdown('<div class="neo4j-status status-online">🟢 Neo4j Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="neo4j-status status-offline">🔴 Neo4j Disconnected</div>', unsafe_allow_html=True)
        st.caption("Check .env or start Neo4j server.")

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
    paper_input = st.text_area(
        "Paste reference paper/book excerpt(s)",
        height=100,
        placeholder="Paste one or more relevant paper abstracts or textbook passages…",
    )
    paper_texts = [p.strip() for p in paper_input.split("\n") if p.strip()]

    st.markdown("---")
    st.markdown(f"### 🗂️ Global Knowledge")
    st.metric("Medical Vocabulary (L3)", len(st.session_state.med_rag.repo_entities_l3))
    st.caption("This knowledge base persists across documents.")

    seed_btn = st.button("🚀 Bulk Seed (Top 1000)", help="Fetches ~1000 medical concepts from UMLS (takes a few mins)")
    if seed_btn:
        terms = get_medical_terms()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, msg):
            progress_bar.progress(current / total)
            status_text.caption(msg)
            
        st.session_state.med_rag.bulk_seed_vocabulary(terms, progress_callback=update_progress)
        st.success("✅ Seeding complete!")
        st.rerun()

    import_btn = st.button("📥 Load Local UMLS Nodes", help="Imports MRCONSO.RRF and MRSTY.RRF to build vocabulary nodes")
    if import_btn:
        import os
        if not os.path.exists("MRCONSO.RRF") or not os.path.exists("MRSTY.RRF"):
            st.error("❌ Could not find MRCONSO.RRF or MRSTY.RRF in the root directory.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_import_progress(val, msg):
                progress_bar.progress(val)
                status_text.caption(msg)
                
            try:
                st.session_state.med_rag.import_local_umls_dump(
                    "MRCONSO.RRF", 
                    "MRSTY.RRF", 
                    progress_callback=update_import_progress
                )
                st.success("✅ Massive UMLS nodes injected!")
            except Exception as e:
                st.error(f"Import failed: {e}")
            st.rerun()

    rel_btn = st.button("🔗 Load Local UMLS Relationships", help="Imports MRREL.RRF to build ontology edges between existing nodes")
    if rel_btn:
        import os
        if not os.path.exists("MRCONSO.RRF") or not os.path.exists("MRREL.RRF"):
            st.error("❌ Could not find MRCONSO.RRF or MRREL.RRF in the root directory.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_rel_progress(val, msg):
                progress_bar.progress(val)
                status_text.caption(msg)
                
            try:
                st.session_state.med_rag.import_local_umls_relationships_dump(
                    "MRCONSO.RRF", 
                    "MRREL.RRF", 
                    progress_callback=update_rel_progress
                )
                st.success("✅ Massive UMLS relationships injected!")
            except Exception as e:
                st.error(f"Import failed: {e}")
            st.rerun()
            st.rerun()
        
    clear_btn = st.button("🗑️ Clear Graph", help="Wipes all memory and detaches all Neo4j nodes")
    if clear_btn:
        st.session_state.med_rag.clear_all()
        st.session_state.build_stats = None
        st.success("✅ Graph completely cleared")
        st.rerun()

    clear_rel_btn = st.button("🗑️ Clear Relationships Only", help="Wipes all relationship edges but keeps the nodes intact")
    if clear_rel_btn:
        st.session_state.med_rag.clear_all_relationships()
        st.success("✅ All relationships successfully cleared")
        st.rerun()

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
                        
                        # Show UMLS source if available
                        label = "L2 Ref" if e.layer == 2 else "L3 Def"
                        if e.layer == 3 and hasattr(e, 'definition') and e.definition:
                            # A simple check for UMLS source (could be improved)
                            label = "L3 UMLS" if len(e.definition) > 0 else "L3 Def"

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

def render_interactive_graph(nx_graph, layers_to_show):
    net = Network(height="600px", width="100%", bgcolor="#0d1117", font_color="#e6edf3")
    
    # Node styles per layer
    colors = {1: "#58a6ff", 2: "#3fb950", 3: "#f78166"}
    
    # Pre-select nodes to avoid showing orphaned L3 nodes
    all_nodes = list(nx_graph.nodes(data=True))
    relevant_l3 = set()
    if 3 in layers_to_show:
        # A Layer 3 node is relevant if it has an edge to/from a Layer 1 or Layer 2 node
        for s, t, d in nx_graph.edges(data=True):
            s_ent = nx_graph.nodes[s].get("entity")
            t_ent = nx_graph.nodes[t].get("entity")
            if s_ent and t_ent:
                if s_ent.layer == 3 and t_ent.layer in [1, 2]: relevant_l3.add(s)
                if t_ent.layer == 3 and s_ent.layer in [1, 2]: relevant_l3.add(t)

    for n, data in all_nodes:
        ent = data.get("entity")
        if not ent: continue
        
        should_add = False
        if ent.layer in [1, 2] and ent.layer in layers_to_show:
            should_add = True
        elif ent.layer == 3 and 3 in layers_to_show and n in relevant_l3:
            should_add = True
            
        if should_add:
            net.add_node(
                n, 
                label=n, 
                title=f"{n} ({ent.entity_type})\n{ent.context[:100]}...",
                color=colors.get(ent.layer, "#8b949e")
            )
            
    for s, t, data in nx_graph.edges(data=True):
        if s in net.get_nodes() and t in net.get_nodes():
            net.add_edge(s, t, title=data.get("relation", ""))

    # Physics for a nice feel
    net.force_atlas_2based()
    
    # Save and read
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
    return html

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

        # View Mode Toggle
        view_mode = st.radio("View Mode", ["Network Graph (Interactive)", "Entity Table"], horizontal=True)

        if view_mode == "Network Graph (Interactive)":
            st.markdown("### 🕸️ Interactive Triple-Layer Graph")
            st.caption("Drag nodes to explore, hover for details. Colors: 🔵 L1 🟢 L2 🔴 L3")
            
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1: show_l1 = st.checkbox("Show Layer 1 (User Docs)", value=True)
            with col_f2: show_l2 = st.checkbox("Show Layer 2 (Med Papers)", value=True)
            with col_f3: show_l3 = st.checkbox("Show Layer 3 (UMLS Vocab)", value=True)
            
            layers = []
            if show_l1: layers.append(1)
            if show_l2: layers.append(2)
            if show_l3: layers.append(3)
            
            if layers:
                html_graph = render_interactive_graph(rag.nx_graph, layers)
                components.html(html_graph, height=620)
            else:
                st.info("Select at least one layer to view.")
        
        else:
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
