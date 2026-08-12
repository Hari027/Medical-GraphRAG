"""
Bulk PubMed Importer — Layer 2 Literature Pipeline
Fetches PubMed articles via NCBI E-utilities, parses structured MeSH metadata,
and ingests article + MeSH term nodes into Neo4j as PubMed_Entity (layer=2).
No OpenAI required — leverages PubMed's own MeSH annotations.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import requests
from neo4j import GraphDatabase


# ~200 broad medical keywords for wide PubMed coverage
PUBMED_KEYWORDS = [
    # Top diseases
    "hypertension", "type 2 diabetes", "coronary artery disease", "heart failure",
    "atrial fibrillation", "chronic kidney disease", "COPD", "asthma",
    "pneumonia", "sepsis", "stroke", "myocardial infarction",
    "alzheimer disease", "parkinson disease", "epilepsy", "multiple sclerosis",
    "depression", "anxiety disorder", "schizophrenia", "bipolar disorder",
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
    "leukemia", "lymphoma", "melanoma", "pancreatic cancer",
    "HIV AIDS", "tuberculosis", "hepatitis B", "hepatitis C", "malaria",
    "rheumatoid arthritis", "osteoarthritis", "systemic lupus erythematosus",
    "psoriasis", "crohn disease", "ulcerative colitis", "celiac disease",
    "cirrhosis", "fatty liver disease", "pancreatitis", "cholecystitis",
    "pulmonary embolism", "deep vein thrombosis", "aortic aneurysm",
    "hyperlipidemia", "hypothyroidism", "hyperthyroidism", "cushing syndrome",
    "sickle cell disease", "hemophilia", "thrombocytopenia", "anemia",
    "obesity", "metabolic syndrome", "gout", "osteoporosis",
    "glaucoma", "macular degeneration", "cataracts", "retinopathy",
    "chronic pain", "migraine", "fibromyalgia", "sleep apnea",
    "urinary tract infection", "pyelonephritis", "nephrotic syndrome",
    "preeclampsia", "gestational diabetes", "endometriosis",
    "appendicitis", "diverticulitis", "irritable bowel syndrome",
    # Top drugs / treatments
    "metformin", "insulin therapy", "atorvastatin", "lisinopril",
    "amlodipine", "metoprolol", "omeprazole", "levothyroxine",
    "warfarin", "apixaban", "rivaroxaban", "clopidogrel",
    "aspirin", "ibuprofen", "acetaminophen", "prednisone",
    "amoxicillin", "azithromycin", "ciprofloxacin", "vancomycin",
    "immunotherapy cancer", "chemotherapy", "radiation therapy",
    "monoclonal antibodies", "CAR T cell therapy",
    # Procedures / diagnostics
    "coronary artery bypass", "angioplasty stent", "dialysis",
    "mechanical ventilation", "echocardiography", "endoscopy",
    "colonoscopy", "MRI brain", "CT angiography", "PET scan",
    "liver transplant", "kidney transplant", "bone marrow transplant",
    # Broad clinical topics
    "drug interactions", "antibiotic resistance", "vaccine efficacy",
    "clinical trial outcomes", "precision medicine", "gene therapy",
    "stem cell therapy", "biomarker diagnosis", "telemedicine",
    "palliative care", "intensive care unit", "emergency medicine",
    "pediatric cardiology", "neonatal sepsis", "geriatric medicine",
    "surgical complications", "postoperative infection", "wound healing",
    "diabetic nephropathy", "diabetic retinopathy", "diabetic neuropathy",
    "cardiac rehabilitation", "pulmonary rehabilitation",
    "nutritional deficiency", "vitamin D deficiency", "iron deficiency",
    # Public health
    "COVID-19 treatment", "influenza vaccination", "maternal mortality",
    "mental health intervention", "substance abuse treatment",
    "smoking cessation", "alcohol use disorder", "opioid crisis",
]


def bulk_fetch_pubmed(keywords: list[str], max_per_keyword: int, 
                      uri: str, user: str, password: str, 
                      progress_callback=None):
    """
    Fetches PubMed articles for each keyword, parses MeSH terms from XML,
    and inserts article + MeSH nodes into Neo4j as Layer 2 (PubMed_Entity).
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    
    total_articles = 0
    total_mesh = 0
    total_keywords = len(keywords)
    
    for ki, keyword in enumerate(keywords):
        if progress_callback:
            pct = ki / total_keywords
            progress_callback(pct, f"[{ki+1}/{total_keywords}] Searching: {keyword}")
        
        try:
            # 1. Search for PMIDs
            search_url = f"{base_url}/esearch.fcgi"
            params = {"db": "pubmed", "term": keyword, "retmax": max_per_keyword, "retmode": "json"}
            res = requests.get(search_url, params=params, timeout=15)
            res.raise_for_status()
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                continue
            
            time.sleep(0.35)  # Rate limit: ~3 req/sec
            
            # 2. Fetch full XML (includes MeSH terms)
            fetch_url = f"{base_url}/efetch.fcgi"
            fetch_params = {"db": "pubmed", "id": ",".join(id_list), "rettype": "xml", "retmode": "xml"}
            f_res = requests.get(fetch_url, params=fetch_params, timeout=30)
            f_res.raise_for_status()
            
            time.sleep(0.35)  # Rate limit
            
            # 3. Parse XML
            root = ET.fromstring(f_res.text)
            article_batch = []
            mesh_batch = []
            rel_batch = []
            
            for article in root.findall(".//PubmedArticle"):
                # Extract PMID
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None
                if not pmid:
                    continue
                
                # Extract title
                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None else "Untitled"
                if not title:
                    title = "Untitled"
                title = title[:300]
                
                # Extract abstract
                abstract_parts = []
                for abs_el in article.findall(".//AbstractText"):
                    if abs_el.text:
                        abstract_parts.append(abs_el.text)
                abstract = " ".join(abstract_parts)[:2000] if abstract_parts else ""
                
                # Article node
                article_batch.append({
                    "name": title,
                    "type": "PubMed Article",
                    "context": abstract if abstract else f"PubMed article PMID:{pmid}",
                    "definition": f"PMID: {pmid}. Source: PubMed/MEDLINE.",
                    "layer": 2,
                    "label": "PubMed_Entity"
                })
                total_articles += 1
                
                # Extract MeSH terms
                for mesh_heading in article.findall(".//MeshHeading"):
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is not None and descriptor.text:
                        mesh_name = descriptor.text[:200]
                        mesh_ui = descriptor.get("UI", "")
                        
                        mesh_batch.append({
                            "name": mesh_name,
                            "type": "MeSH Descriptor",
                            "context": f"MeSH term: {mesh_name}. UI: {mesh_ui}",
                            "definition": f"MeSH UI: {mesh_ui}. NLM controlled vocabulary.",
                            "layer": 2,
                            "label": "PubMed_Entity"
                        })
                        
                        rel_batch.append({
                            "source": title,
                            "target": mesh_name,
                            "relation": "HAS_MESH"
                        })
                        total_mesh += 1
            
            # 4. Push to Neo4j
            with driver.session() as session:
                if article_batch:
                    _insert_pubmed_batch(session, article_batch)
                if mesh_batch:
                    _insert_pubmed_batch(session, mesh_batch)
                if rel_batch:
                    _insert_pubmed_rel_batch(session, rel_batch)
                    
        except Exception as e:
            print(f"Error fetching keyword '{keyword}': {e}")
            continue
    
    driver.close()
    
    if progress_callback:
        progress_callback(1.0, f"Done! {total_articles} articles, {total_mesh} MeSH terms ingested.")
    
    print(f"Bulk PubMed import complete: {total_articles} articles, {total_mesh} MeSH terms.")
    return {"articles": total_articles, "mesh_terms": total_mesh}


def _insert_pubmed_batch(session, batch):
    query = (
        "UNWIND $batch AS item "
        "MERGE (n:Entity {name: item.name}) "
        "SET n.type = item.type, n.context = item.context, "
        "    n.definition = item.definition, n.layer = item.layer, "
        "    n.layer_label = item.label "
    )
    session.run(query, batch=batch)


def _insert_pubmed_rel_batch(session, batch):
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
    
    # Quick test with 3 keywords
    bulk_fetch_pubmed(PUBMED_KEYWORDS[:3], max_per_keyword=10, 
                      uri=uri, user=user, password=password)
