from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

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
