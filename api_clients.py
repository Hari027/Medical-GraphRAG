from typing import Optional
import requests
import numpy as np
from neo4j import GraphDatabase
from data_models import Entity, Relationship

class UMLSClient:
    """
    Client for UMLS UTS REST API.
    Handles semantic search for terms and retrieval of definitions.
    """
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Simple in-memory cache: term -> {name, definition, type}
        self._cache: dict[str, dict] = {}

    def get_term_details(self, term: str) -> Optional[dict]:
        """
        Search for a term, find its CUI, and fetch definitions.
        Returns a dict with name, definition, and semantic type if found.
        """
        if not self.api_key:
            return None
        
        term_lower = term.lower()
        if term_lower in self._cache:
            return self._cache[term_lower]

        try:
            # 1. Search for Concept (CUI)
            search_url = f"{self.BASE_URL}/search/current"
            params = {
                "string": term,
                "apiKey": self.api_key,
                "searchType": "exact"  # better for specific medical terms
            }
            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("result", {}).get("results", [])

            if not results:
                # Try 'words' search if exact fails
                params["searchType"] = "words"
                resp = requests.get(search_url, params=params, timeout=10)
                results = resp.json().get("result", {}).get("results", [])

            if not results:
                return None

            # Pick the first result
            best_match = results[0]
            cui = best_match.get("ui")
            name = best_match.get("name")

            # 2. Get Definitions
            def_url = f"{self.BASE_URL}/content/current/CUI/{cui}/definitions"
            resp = requests.get(def_url, params={"apiKey": self.api_key}, timeout=10)
            
            definition = ""
            if resp.status_code == 200:
                defs = resp.json().get("result", [])
                if defs:
                    # Prefer NCI or MSH sources if available
                    for d in defs:
                        if d.get("rootSource") in ["NCI", "MSH"]:
                            definition = d.get("value")
                            break
                    if not definition:
                        definition = defs[0].get("value")

            # 3. Get Semantic Types (optional, but good for Entity object)
            # In UMLS API, we can get this from concept details
            concept_url = f"{self.BASE_URL}/content/current/CUI/{cui}"
            resp = requests.get(concept_url, params={"apiKey": self.api_key}, timeout=10)
            semantic_type = "Other"
            if resp.status_code == 200:
                stys = resp.json().get("result", {}).get("semanticTypes", [])
                if stys:
                    semantic_type = stys[0].get("name", "Other")

            if name and (definition or semantic_type):
                res = {
                    "name": name,
                    "definition": definition,
                    "type": semantic_type,
                    "source": "UMLS"
                }
                self._cache[term_lower] = res
                return res

        except Exception as e:
            print(f"UMLS API Error: {e}")
            return None

        return None


class Neo4jClient:
    """
    Client for syncing the knowledge graph to Neo4j.
    """
    def __init__(self, uri: Optional[str], user: Optional[str], password: Optional[str]):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        if uri and user and password:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                print("[Neo4j] Connected successfully.")
                # Create index in background thread so it doesn't block startup
                import threading
                def _bg_index():
                    try:
                        self.ensure_indexes()
                        print("[Neo4j] Background index creation complete.")
                    except Exception as ex:
                        print(f"[Neo4j] Background index error (non-fatal): {ex}")
                threading.Thread(target=_bg_index, daemon=True).start()
            except Exception as e:
                print(f"Neo4j Connection Error: {e}")
                self.driver = None

    def ensure_indexes(self):
        """Creates a unique constraint on Entity name if it doesn't exist."""
        if not self.driver: return
        with self.driver.session() as session:
            try:
                # Use newer Neo4j 5.x syntax for constraints
                session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            except Exception as e:
                print(f"Neo4j Index Creation (Constraint) Error: {e}")
                # Fallback to older syntax if needed or just skip
                try:
                    session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)")
                except:
                    pass

    def sync_tag_tree(self, tag_tree: list[dict]):
        """Persists the hierarchical tag tree to Neo4j for database-native U-Retrieval."""
        if not self.driver or not tag_tree: return
        
        with self.driver.session() as session:
            # Helper to recursively sync nodes
            def _sync_node(node, parent_id=None):
                node_id = f"tag_{hash(str(node['tags']))}"
                # Create the TagNode
                query = (
                    "MERGE (t:TagNode {id: $tid}) "
                    "SET t.tags_json = $tags_json, t.is_root = $is_root "
                    "RETURN t"
                )
                session.run(query, tid=node_id, tags_json=str(node['tags']), is_root=(parent_id is None))
                
                # Link to parent
                if parent_id:
                    session.run(
                        "MATCH (p:TagNode {id: $pid}), (c:TagNode {id: $cid}) "
                        "MERGE (p)-[:HAS_CHILD]->(c)",
                        pid=parent_id, cid=node_id
                    )
                
                # Link to leaf chunks if children is empty
                if not node['children'] and node['ids']:
                    for cid in node['ids']:
                        # Assuming chunks are also in Neo4j with a Chunk label or identified by prefix
                        session.run(
                            "MATCH (t:TagNode {id: $tid}), (e:Entity {name: $ename}) "
                            "MERGE (t)-[:INDEXES]->(e)",
                            tid=node_id, ename=node['ids'][0] # simplify to first entity
                        )

                for child in node['children']:
                    _sync_node(child, node_id)

            for root in tag_tree:
                _sync_node(root)

    def count_layer3(self) -> int:
        """Fast count of Layer 3 nodes in Neo4j (no RAM loading)."""
        if not self.driver: return 0
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) WHERE n.layer = 3 RETURN count(n) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_db(self, progress_callback=None):
        if not self.driver: return
        if progress_callback: progress_callback("🧹 Clearing Layer 1 & 2 from Neo4j…")
        with self.driver.session() as session:
            # Delete L1 and L2 entities, and all TagNodes
            session.run("MATCH (n:Entity) WHERE n.layer IN [1, 2] DETACH DELETE n")
            session.run("MATCH (t:TagNode) DETACH DELETE t")

    def clear_layer1_db(self):
        """Specifically clears only Layer 1 (user document) nodes from Neo4j."""
        if not self.driver: return
        with self.driver.session() as session:
            session.run("MATCH (n:Entity) WHERE n.layer = 1 DETACH DELETE n")

    def clear_all_db(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Delete every entity across all layers
            session.run("MATCH (n:Entity) DETACH DELETE n")

    def clear_all_relationships(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Batch delete relationships to prevent memory crash on 27M edges
            while True:
                res = session.run("MATCH ()-[r]->() WITH r LIMIT 50000 DELETE r RETURN count(r) AS deleted_count")
                record = res.single()
                if not record or record["deleted_count"] == 0:
                    break

    def load_layer2_entities(self) -> list[Entity]:
        """Load all Layer 2 entities back from Neo4j into Python Entity objects."""
        if not self.driver: return []
        entities = []
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) WHERE n.layer = 2 "
                "RETURN n.name AS name, n.type AS type, n.context AS context, "
                "n.definition AS definition, n.embedding AS embedding"
            )
            for record in result:
                e = Entity(
                    name=record["name"],
                    entity_type=record["type"] or "Medical Concept",
                    context=record["context"] or "",
                    definition=record["definition"] or "",
                    layer=2,
                )
                if record["embedding"]:
                    e.embedding = np.array(record["embedding"])
                entities.append(e)
        return entities

    def sync_entities(self, entities: list[Entity], progress_callback=None):
        if not self.driver or not entities: return
        
        total = len(entities)
        chunk_size = 1000
        
        for i in range(0, total, chunk_size):
            chunk = entities[i : i + chunk_size]
            if progress_callback:
                progress_callback(f"📦 Syncing Entities: {i}/{total}…")
                
            with self.driver.session() as session:
                query_simple = (
                    "UNWIND $batch AS item "
                    "MERGE (n:Entity {name: item.name}) "
                    "SET n.type = item.type, n.context = item.context, "
                    "    n.definition = item.definition, n.layer = item.layer, "
                    "    n.layer_label = item.label, n.embedding = item.embedding "
                    "RETURN count(*)"
                )
                
                batch = [
                    {
                        "name": e.name, 
                        "type": e.entity_type, 
                        "context": e.context, 
                        "definition": e.definition, 
                        "layer": e.layer,
                        "embedding": e.embedding.tolist() if e.embedding is not None else None,
                        "label": f"L{e.layer}_Entity"
                    } for e in chunk
                ]
                try:
                    session.run(query_simple, batch=batch)
                except Exception as e:
                    print(f"Neo4j Sync Entities Error (chunk {i//chunk_size}): {e}")

    def sync_relationships(self, relationships: list[Relationship], progress_callback=None):
        if not self.driver or not relationships: return
        
        total = len(relationships)
        chunk_size = 1000
        
        for i in range(0, total, chunk_size):
            chunk = relationships[i : i + chunk_size]
            if progress_callback:
                progress_callback(f"🔗 Syncing Relationships: {i}/{total}…")
                
            with self.driver.session() as session:
                query = (
                    "UNWIND $batch AS item "
                    "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
                    "MERGE (s)-[rel:RELATED_TO {type: item.rel_type}]->(t) "
                    "RETURN count(*)"
                )
                batch = [{"source": r.source, "target": r.target, "rel_type": r.relation} for r in chunk]
                try:
                    session.run(query, batch=batch)
                except Exception as e:
                    print(f"Neo4j Sync Relationships Error (chunk {i//chunk_size}): {e}")

    def add_cross_layer_edge(self, source_name: str, target_name: str, relation_type: str, similarity: float):
        if not self.driver: return
        with self.driver.session() as session:
            try:
                session.run(
                    "MATCH (s:Entity {name: $source}), (t:Entity {name: $target}) "
                    "MERGE (s)-[rel:LINK {type: $rel_type}]->(t) "
                    "SET rel.similarity = $sim",
                    source=source_name, target=target_name, rel_type=relation_type, sim=similarity
                )
            except Exception as e:
                print(f"Neo4j Sync Cross-Layer Edge Error: {e}")

    def batch_add_cross_layer_edges(self, edges: list[tuple[str, str, str, float]], chunk_size: int = 1000):
        if not self.driver or not edges: return
        
        # Chunk the data to prevent ConnectionReset/Timeout on massive transactions
        for i in range(0, len(edges), chunk_size):
            chunk = edges[i : i + chunk_size]
            with self.driver.session() as session:
                query = (
                    "UNWIND $batch AS edge "
                    "MATCH (s:Entity {name: edge.source}), (t:Entity {name: edge.target}) "
                    "MERGE (s)-[rel:LINK {type: edge.rel_type}]->(t) "
                    "SET rel.similarity = edge.sim "
                    "RETURN count(*)"
                )
                batch = [{"source": e[0], "target": e[1], "rel_type": e[2], "sim": e[3]} for e in chunk]
                try:
                    session.run(query, batch=batch)
                except Exception as e:
                    print(f"Neo4j Batch Sync Cross-Layer Edge Error (chunk {i//chunk_size}): {e}")

    def get_k_hop_neighbors(self, entity_name: str, k: int) -> list[Entity]:
        """
        Retrieves all entities within k hops of the given entity from Neo4j.
        """
        if not self.driver: return []
        entities = []
        with self.driver.session() as session:
            # Cypher wildcard for up to k hops regardless of direction and edge type
            query = f"""
            MATCH (start:Entity {{name: $name}})-[*1..{k}]-(neighbor:Entity)
            RETURN DISTINCT neighbor.name AS name, neighbor.type AS type, 
                   neighbor.context AS context, neighbor.definition AS definition, 
                   neighbor.layer AS layer
            LIMIT 50
            """
            try:
                result = session.run(query, name=entity_name)
                for record in result:
                    e = Entity(
                        name=record["name"],
                        entity_type=record["type"] or "Medical Concept",
                        context=record["context"] or "",
                        definition=record["definition"] or "",
                        layer=record["layer"] or 3
                    )
                    entities.append(e)
            except Exception as e:
                print(f"Neo4j get_k_hop_neighbors Error: {e}")
        return entities

class PubMedClient:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def fetch_abstracts(self, query: str, max_results: int = 5) -> list[str]:
        try:
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
            res = requests.get(search_url, params=params, timeout=10)
            res.raise_for_status()
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return []
                
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {"db": "pubmed", "id": ",".join(id_list), "rettype": "abstract", "retmode": "text"}
            f_res = requests.get(fetch_url, params=fetch_params, timeout=15)
            f_res.raise_for_status()
            
            raw_text = f_res.text
            articles = [a.strip() for a in raw_text.split("\n\n\n") if a.strip()]
            return articles if articles else [raw_text.strip()]
        except Exception as e:
            print(f"PubMed API Error: {e}")
            return []
