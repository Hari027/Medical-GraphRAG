"""
MedQA-USMLE Benchmark for MedGraphRAG Pipeline
================================================
For each question in the test set:
  1. Clear Layer 1 (keep L2 + L3 persistent)
  2. Ingest the vignette text as Layer 1 (chunk → extract entities → embed → Neo4j sync)
  3. Link L1 → L2 via cosine similarity (L2 → L3 edges already exist)
  4. Run U-Retrieval (top-down tag tree → top-N entities → k-hop triple neighbours)
  5. Prompt the LLM with retrieved context + original question + options → pick an answer
  6. Compare against ground truth and log results

Usage:
    python benchmark_medqa.py [--limit N] [--output results.json]
"""

import os
import sys
import json
import time
import argparse
import textwrap
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "MedGraphRAG-Benchmark/1.0"

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from medgraphrag.pipeline import MedGraphRAG


# ─── Configuration ──────────────────────────────────────────────────────────

DATASET_PATH = os.path.join("MedQA-USMLE", "questions", "US", "test.jsonl")

DEFAULT_OUTPUT = "benchmark_results.json"


def load_questions(path: str, limit: int | None = None) -> list[dict]:
    """Load MedQA-USMLE questions from a JSONL file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            questions.append(q)
            if limit and len(questions) >= limit:
                break
    return questions


def format_options(options: dict) -> str:
    """Format options dict into a readable string."""
    return "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))


def build_answer_prompt(
    question: str,
    options: dict,
    retrieved_entities: list,
    triple_neighbours: list,
    target_graph,
    refined_answer: str,
) -> str:
    """
    Build the final MCQ prompt that gives the LLM:
      - The original vignette/question
      - The answer options
      - All retrieved context from the pipeline
    And asks it to pick exactly one option letter.
    """
    # Entity context
    entity_lines = []
    for e in retrieved_entities:
        line = f"• {e.name} ({e.entity_type}, L{e.layer}): {e.context[:200]}"
        if e.definition:
            line += f"\n  Definition: {e.definition[:200]}"
        entity_lines.append(line)
    entity_text = "\n".join(entity_lines) if entity_lines else "(none)"

    # Triple neighbour context
    neighbour_lines = []
    for e in triple_neighbours:
        line = f"• {e.name} ({e.entity_type}, L{e.layer}): {e.context[:200]}"
        if e.definition:
            line += f"\n  Definition: {e.definition[:200]}"
        neighbour_lines.append(line)
    neighbour_text = "\n".join(neighbour_lines) if neighbour_lines else "(none)"

    # Graph relationships
    rel_text = "(none)"
    if target_graph and target_graph.relationships:
        rel_lines = []
        for r in target_graph.relationships[:15]:
            rel_lines.append(f"  {r.source} --[{r.relation}]--> {r.target}")
        rel_text = "\n".join(rel_lines)

    options_text = format_options(options)

    prompt = textwrap.dedent(f"""\
        You are an expert medical professional answering a USMLE-style multiple choice question.

        QUESTION:
        {question}

        OPTIONS:
        {options_text}

        === RETRIEVED MEDICAL KNOWLEDGE (from Knowledge Graph) ===

        RETRIEVED ENTITIES:
        {entity_text}

        CROSS-LAYER NEIGHBOURS (Literature + UMLS Definitions):
        {neighbour_text}

        GRAPH RELATIONSHIPS:
        {rel_text}

        GRAPH-BASED ANALYSIS:
        {refined_answer[:800] if refined_answer else "(none)"}

        === INSTRUCTIONS ===
        Based on the question, the options, and the retrieved medical knowledge above,
        select the single best answer. Think step by step, then state your final answer
        as EXACTLY one letter (A, B, C, D, or E) on the last line of your response,
        formatted as:
        ANSWER: X
    """)
    return prompt


def extract_answer_letter(llm_response: str) -> str:
    """Extract the answer letter from the LLM response."""
    # Look for "ANSWER: X" pattern (last occurrence)
    lines = llm_response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip().upper()
        if line.startswith("ANSWER:"):
            letter = line.replace("ANSWER:", "").strip()
            if letter and letter[0] in "ABCDE":
                return letter[0]

    # Fallback: look for standalone letter at end
    for line in reversed(lines):
        line = line.strip()
        if line and line[0] in "ABCDE" and len(line) <= 3:
            return line[0]

    return "?"


def run_benchmark(
    rag: MedGraphRAG,
    llm: ChatOpenAI,
    questions: list[dict],
    output_path: str,
    start_idx: int = 0,
    start_correct: int = 0,
    start_done: int = 0,
):
    """Run the full benchmark pipeline."""
    results = []
    
    if start_idx > 0 and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data.get("results", [])
        except Exception as e:
            print(f"Warning: could not load existing {output_path} to resume: {e}")

    correct = start_correct
    done = start_done
    total = len(questions)

    print(f"\n{'='*70}")
    print(f"  MedGraphRAG Benchmark — {total} questions")
    if start_idx > 0:
        print(f"  Resuming from index {start_idx} (Prior correct: {correct}/{done})")
    print(f"{'='*70}\n")

    for i in range(start_idx, total):
        idx = i
        q = questions[i]
        q_text = q["question"]
        options = q["options"]
        gt_letter = q["answer_idx"]
        gt_answer = q["answer"]

        print(f"\n[{idx+1}/{total}] Processing question...")
        print(f"  GT: {gt_letter}) {gt_answer[:80]}...")
        t0 = time.time()

        # ── Step 1: Clear Layer 1 ──
        rag.clear_layer1()

        # ── Step 2: Ingest vignette as Layer 1 ──
        try:
            stats = rag.load_documents(user_text=q_text)
        except Exception as e:
            print(f"  ❌ Ingestion failed: {e}")
            results.append({
                "index": idx,
                "question": q_text[:200],
                "gt_answer": gt_letter,
                "predicted": "?",
                "correct": False,
                "error": str(e),
                "time_sec": time.time() - t0,
            })
            continue

        # ── Step 3+4: U-Retrieval ──
        try:
            retrieval = rag.query(q_text)
        except Exception as e:
            print(f"  ❌ U-Retrieval failed: {e}")
            results.append({
                "index": idx,
                "question": q_text[:200],
                "gt_answer": gt_letter,
                "predicted": "?",
                "correct": False,
                "error": str(e),
                "time_sec": time.time() - t0,
            })
            continue

        # ── Step 5: Final MCQ Prompt ──
        prompt = build_answer_prompt(
            question=q_text,
            options=options,
            retrieved_entities=retrieval["top_entities"],
            triple_neighbours=retrieval["triple_neighbours"],
            target_graph=retrieval["target_graph"],
            refined_answer=retrieval["answer"],
        )

        try:
            resp = llm.invoke(prompt)
            llm_raw = resp.content.strip()
            predicted = extract_answer_letter(llm_raw)
        except Exception as e:
            print(f"  ❌ LLM answer failed: {e}")
            predicted = "?"
            llm_raw = str(e)

        is_correct = predicted == gt_letter
        if is_correct:
            correct += 1
        
        done += 1

        elapsed = time.time() - t0
        accuracy_so_far = correct / done * 100

        status = "✅" if is_correct else "❌"
        print(f"  {status} Predicted: {predicted} | GT: {gt_letter} | "
              f"Time: {elapsed:.1f}s | Running Acc: {accuracy_so_far:.1f}%")

        # Serialize entities grouped by layer
        def _entity_to_dict(e):
            return {
                "name": e.name,
                "type": e.entity_type,
                "context": e.context[:300],
                "definition": (e.definition[:300] if e.definition else ""),
                "layer": e.layer,
            }

        all_entities = retrieval.get("top_entities", []) + retrieval.get("triple_neighbours", [])
        l1_ents = [_entity_to_dict(e) for e in all_entities if e.layer == 1]
        l2_ents = [_entity_to_dict(e) for e in all_entities if e.layer == 2]
        l3_ents = [_entity_to_dict(e) for e in all_entities if e.layer == 3]

        # Log details — full info only for wrong answers to keep JSON small
        result_entry = {
            "index": idx,
            "question": q_text[:300],
            "gt_answer": gt_letter,
            "gt_answer_text": gt_answer,
            "predicted": predicted,
            "correct": is_correct,
            "time_sec": round(elapsed, 2),
            "l1_entities_count": stats.get("l1_entities", 0),
            "cross_layer_edges": stats.get("cross_layer_edges", 0),
            "retrieved_entities_count": len(retrieval.get("top_entities", [])),
            "triple_neighbours_count": len(retrieval.get("triple_neighbours", [])),
            "refinement_steps": len(retrieval.get("refinement_log", [])),
        }

        # Only save full entity lists and LLM response for wrong answers
        if not is_correct:
            result_entry["layer1_entities"] = l1_ents
            result_entry["layer2_entities"] = l2_ents
            result_entry["layer3_entities"] = l3_ents
            result_entry["llm_response"] = llm_raw[:500]

        results.append(result_entry)

        # Save incrementally so we don't lose progress on crash
        _save_results(output_path, results, correct, done, total)

    # ── Final Summary ──
    final_accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS: {correct}/{total} correct ({final_accuracy:.1f}%)")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}\n")

    _save_results(output_path, results, correct, total, total)


def _save_results(path, results, correct, done, total):
    """Save results to JSON incrementally."""
    output = {
        "benchmark": "MedQA-USMLE",
        "timestamp": datetime.now().isoformat(),
        "progress": f"{done}/{total}",
        "correct": correct,
        "total_done": done,
        "accuracy_pct": round(correct / done * 100, 2) if done > 0 else 0,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MedGraphRAG against MedQA-USMLE")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process (default: all)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH,
                        help=f"Path to MedQA JSONL file (default: {DATASET_PATH})")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Index to start from (e.g. 200)")
    parser.add_argument("--start-correct", type=int, default=0,
                        help="Number of correct answers prior to start-idx")
    parser.add_argument("--start-done", type=int, default=0,
                        help="Number of completed answers prior to start-idx")
    args = parser.parse_args()

    # ── Validate environment ──
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD", "password")
    umls_api_key = os.environ.get("UMLS_API_KEY")

    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at {args.dataset}")
        sys.exit(1)

    # ── Load questions ──
    questions = load_questions(args.dataset, limit=args.limit)
    print(f"Loaded {len(questions)} questions from {args.dataset}")

    # ── Initialize pipeline ──
    print("Initializing MedGraphRAG pipeline...")
    llm = ChatOllama(model="gemma4:31b-cloud", temperature=0.0)

    embedder = HuggingFaceEmbeddings(
        model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        encode_kwargs={"normalize_embeddings": True},
    )

    neo4j_creds = {
        "uri": neo4j_uri,
        "user": neo4j_user,
        "password": neo4j_pass,
    }

    rag = MedGraphRAG(
        llm=llm,
        embedder=embedder,
        umls_api_key=umls_api_key,
        neo4j_creds=neo4j_creds,
    )

    print(f"  L2 entities loaded: {len(rag.repo_entities_l2)}")
    print(f"  L3 count (Neo4j): {rag.neo4j.count_layer3() if rag.neo4j else 0}")
    print(f"  Neo4j: {'Connected' if rag.neo4j and rag.neo4j.driver else 'Disconnected'}")

    # ── Run ──
    run_benchmark(
        rag, llm, questions, args.output, 
        start_idx=args.start_idx, 
        start_correct=args.start_correct, 
        start_done=args.start_done
    )


if __name__ == "__main__":
    main()
