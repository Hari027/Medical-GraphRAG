# Architecture Overview

```mermaid
flowchart TD
    User([User Question]) --> QP[Query Processing & Embedding]
    QP --> TDR[Top-Down Retrieval]
    
    subgraph Graph [Triple-Layer Medical Graph]
        L1[Layer 1: Clinical/Patient]
        L2[Layer 2: PubMed Literature]
        L3[Layer 3: UMLS Terminology]
        
        L1 -->|the_reference_of| L2
        L2 -->|the_definition_of| L3
    end
    
    TDR --> Graph
    Graph --> TNE[Triple-Neighbour Expansion]
    TNE --> IAG[Initial Answer Generation]
    IAG --> BUR[Bottom-Up Refinement]
    BUR --> FA([Final Grounded Answer])
```

## Layers Explained

1. **Layer 1 (Clinical/Patient)**: Patient specific data extracted via LLM (Entity-to-Entity relationship extraction) during ingestion.
2. **Layer 2 (PubMed)**: Evidentiary grounding from PubMed abstracts, pre-computed in batches.
3. **Layer 3 (UMLS)**: Core medical dictionary seeded from the UMLS metathesaurus for standardization and definition lookups.
