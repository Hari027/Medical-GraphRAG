from typing import Optional
import requests
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
            except Exception as e:
                print(f"Neo4j Connection Error: {e}")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_db(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Only delete L1 and L2 entities and their relationships
            session.run("MATCH (n:Entity) WHERE n.layer IN [1, 2] DETACH DELETE n")

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

    def sync_entities(self, entities: list[Entity]):
        if not self.driver: return
        with self.driver.session() as session:
            # Batch MERGE for efficiency
            query = (
                "UNWIND $batch AS item "
                "MERGE (n:Entity {name: item.name}) "
                "SET n.type = item.type, n.context = item.context, "
                "    n.definition = item.definition, n.layer = item.layer "
                "WITH n, item "
                "CALL apoc.create.addLabels(n, [item.label]) YIELD node "
                "RETURN count(*)"
            )
            # fallback if APOC is not installed
            query_simple = (
                "UNWIND $batch AS item "
                "MERGE (n:Entity {name: item.name}) "
                "SET n.type = item.type, n.context = item.context, "
                "    n.definition = item.definition, n.layer = item.layer, "
                "    n.layer_label = item.label "
                "RETURN count(*)"
            )
            
            batch = [
                {
                    "name": e.name, 
                    "type": e.entity_type, 
                    "context": e.context, 
                    "definition": e.definition, 
                    "layer": e.layer,
                    "label": f"L{e.layer}_Entity"
                } for e in entities
            ]
            try:
                session.run(query_simple, batch=batch)
            except Exception as e:
                print(f"Neo4j Sync Entities Error: {e}")

    def sync_relationships(self, relationships: list[Relationship]):
        if not self.driver: return
        with self.driver.session() as session:
            query = (
                "UNWIND $batch AS item "
                "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
                "MERGE (s)-[rel:RELATED_TO {type: item.rel_type}]->(t) "
                "RETURN count(*)"
            )
            batch = [
                {
                    "source": r.source, 
                    "target": r.target, 
                    "rel_type": r.relation
                } for r in relationships
            ]
            try:
                session.run(query, batch=batch)
            except Exception as e:
                print(f"Neo4j Sync Relationships Error: {e}")

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
