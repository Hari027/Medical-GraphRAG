import os
import sys
from neo4j import GraphDatabase

def load_umls_to_neo4j(mrconso_path: str, mrsty_path: str, uri: str, user: str, password: str, progress_callback=None):
    if not os.path.exists(mrconso_path) or not os.path.exists(mrsty_path):
        raise FileNotFoundError("MRCONSO.RRF or MRSTY.RRF not found in the specified path.")

    # 1. Parse MRSTY.RRF to map CUI -> Semantic Type
    print("Loading Semantic Types from MRSTY.RRF...")
    if progress_callback: progress_callback(0.0, "Loading Semantic Types from MRSTY.RRF...")
    
    cui_to_sty = {}
    with open(mrsty_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                cui_to_sty[parts[0]] = parts[3]
                
    # 2. Parse MRCONSO.RRF and batch insert
    print(f"Loaded {len(cui_to_sty)} semantic types. Streaming MRCONSO.RRF...")
    if progress_callback: progress_callback(0.1, "Streaming MRCONSO.RRF and importing to Neo4j...")
    
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # CREATE INDEX to prevent O(N^2) slowdown during MERGE operations
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    except Exception as e:
        raise Exception(f"Neo4j Connection Error: {e}")
    
    batch_size = 5000  # Lower batch size so Neo4j doesn't hang
    batch = []
    seen_cuis = set()
    total_inserted = 0
    
    file_size = os.path.getsize(mrconso_path)
    processed_bytes = 0

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f, driver.session() as session:
        for line in f:
            processed_bytes += len(line.encode('utf-8', errors='ignore'))
            parts = line.split('|')
            if len(parts) < 15: continue
            
            cui = parts[0]
            lat = parts[1]
            str_name = parts[14]
            
            if lat == 'ENG' and cui not in seen_cuis:
                seen_cuis.add(cui)
                sty = cui_to_sty.get(cui, "Medical Concept")
                
                batch.append({
                    "name": str_name[:200],  # Truncate
                    "type": sty,
                    "context": f"UMLS Concept {cui}",
                    "definition": f"UMLS CUI: {cui}. Semantic Type: {sty}",
                    "layer": 3,
                    "label": "L3_Entity"
                })
                
                if len(batch) >= batch_size:
                    _insert_batch(session, batch)
                    total_inserted += len(batch)
                    batch = []
                    
                    prog_msg = f"Imported {total_inserted} unique UMLS concepts..."
                    print(prog_msg)
                    sys.stdout.flush()
                    if progress_callback:
                        progress = 0.1 + 0.9 * (processed_bytes / file_size)
                        progress_callback(min(progress, 0.99), prog_msg)
                        
        if batch:
            _insert_batch(session, batch)
            total_inserted += len(batch)
            
    driver.close()
    if progress_callback: progress_callback(1.0, f"✅ Completed! Imported {total_inserted} UMLS concepts.")
    return total_inserted

def _insert_batch(session, batch):
    query = (
        "UNWIND $batch AS item "
        "MERGE (n:Entity {name: item.name}) "
        "SET n.type = item.type, n.context = item.context, "
        "    n.definition = item.definition, n.layer = item.layer, "
        "    n.layer_label = item.label "
    )
    session.run(query, batch=batch)

def load_umls_relationships_to_neo4j(mrconso_path: str, mrrel_path: str, uri: str, user: str, password: str, progress_callback=None):
    if not os.path.exists(mrconso_path) or not os.path.exists(mrrel_path):
        raise FileNotFoundError("MRCONSO.RRF or MRREL.RRF not found in the specified path.")

    print("Building CUI -> Name mapping from MRCONSO.RRF...")
    if progress_callback: progress_callback(0.0, "Building CUI to Name mapping from MRCONSO.RRF...")
    
    cui_to_name = {}
    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 15: continue
            
            cui = parts[0]
            lat = parts[1]
            str_name = parts[14]
            
            if lat == 'ENG' and cui not in cui_to_name:
                cui_to_name[cui] = str_name[:200]
                
    print(f"Loaded {len(cui_to_name)} CUIs. Streaming MRREL.RRF...")
    if progress_callback: progress_callback(0.2, f"Loaded {len(cui_to_name)} CUIs. Processing MRREL.RRF...")
    
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        raise Exception(f"Neo4j Connection Error: {e}")
        
    batch_size = 10000 
    batch = []
    total_inserted = 0
    
    file_size = os.path.getsize(mrrel_path)
    processed_bytes = 0

    with open(mrrel_path, 'r', encoding='utf-8', errors='ignore') as f, driver.session() as session:
        for line in f:
            processed_bytes += len(line.encode('utf-8', errors='ignore'))
            parts = line.split('|')
            if len(parts) < 8: continue
            
            cui1 = parts[0]
            rel = parts[3]
            cui2 = parts[4]
            rela = parts[7]
            
            if not rela: rela = rel
            
            if cui1 in cui_to_name and cui2 in cui_to_name:
                batch.append({
                    "source": cui_to_name[cui1],
                    "target": cui_to_name[cui2],
                    "relation": rela[:50]
                })
                
                if len(batch) >= batch_size:
                    _insert_rel_batch(session, batch)
                    total_inserted += len(batch)
                    batch = []
                    
                    prog_msg = f"Imported {total_inserted} UMLS relationships..."
                    print(prog_msg)
                    sys.stdout.flush()
                    if progress_callback:
                        progress = 0.2 + 0.8 * (processed_bytes / file_size)
                        progress_callback(min(progress, 0.99), prog_msg)
                        
        if batch:
            _insert_rel_batch(session, batch)
            total_inserted += len(batch)
            
    driver.close()
    if progress_callback: progress_callback(1.0, f"✅ Completed! Imported {total_inserted} UMLS relationships.")
    return total_inserted

def _insert_rel_batch(session, batch):
    query = (
        "UNWIND $batch AS item "
        "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
        "MERGE (s)-[:RELATED_TO {type: item.relation}]->(t)"
    )
    session.run(query, batch=batch)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    load_umls_to_neo4j("MRCONSO.RRF", "MRSTY.RRF", uri, user, password)
