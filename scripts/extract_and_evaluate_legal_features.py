"""
Phase 5: Legal Knowledge Feature Extraction, Gold Benchmark Validation,
and Coverage Analysis Script.

Generates:
- results/tables/legal_feature_statistics.json
- results/tables/legal_feature_coverage.csv
- results/reports/legal_feature_analysis.md
- data/processed/IL-TUR-PCR/{split}/legal_features.json
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.legal_features.statute_extractor import StatuteExtractor
from src.legal_features.citation_extractor import CitationExtractor
from src.legal_features.entity_extractor import LegalEntityExtractor
from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, RHETORICAL_ROLES, ROLE_TO_IDX
from src.legal_features.legal_feature_encoder import LegalFeatureEncoder, DocumentLegalFeatures
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "external" / "IL-TUR-annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_gold_lner_validation(encoder: LegalFeatureEncoder) -> Dict[str, Any]:
    """Validates entity, statute, and citation extraction against IL-TUR L-NER gold annotations."""
    lner_path = ANNOTATIONS_DIR / "fold_1-00000-of-00001.parquet"
    if not lner_path.exists():
        print(f"Warning: L-NER annotation file {lner_path} not found. Skipping gold L-NER evaluation.")
        return {}

    print(f"Running Gold Benchmark Validation against {lner_path.name} (35 annotated judgments)...")
    df = pd.read_parquet(lner_path)

    # Label dictionary from IL-TUR L-NER specification
    # 0: APP, 1: RESP, 2: A.COUNSEL, 3: R.COUNSEL, 4: JUDGE, 5: WIT, 6: AUTH, 7: COURT, 8: STAT, 9: PREC, 10: DATE, 11: CASENO
    metrics = {
        "statute": {"gold": 0, "extracted": 0, "tp": 0},
        "court": {"gold": 0, "extracted": 0, "tp": 0},
        "judge": {"gold": 0, "extracted": 0, "tp": 0},
        "citation": {"gold": 0, "extracted": 0, "tp": 0},
    }

    for idx, row in df.iterrows():
        text = str(row["text"])
        spans = row["spans"]

        # Gold spans per category
        gold_stat = [s for s in spans if s["label"] == 8]
        gold_court = [s for s in spans if s["label"] == 7]
        gold_judge = [s for s in spans if s["label"] == 4]
        gold_prec = [s for s in spans if s["label"] == 9]

        metrics["statute"]["gold"] += len(gold_stat)
        metrics["court"]["gold"] += len(gold_court)
        metrics["judge"]["gold"] += len(gold_judge)
        metrics["citation"]["gold"] += len(gold_prec)

        # Extract features
        ext_stat = encoder.statute_extractor.extract(text)
        ext_ent = encoder.entity_extractor.extract(text)
        ext_cit = encoder.citation_extractor.extract(text)

        ext_court = [e for e in ext_ent if e.category == "COURT"]
        ext_judge = [e for e in ext_ent if e.category == "JUDGE"]

        metrics["statute"]["extracted"] += len(ext_stat)
        metrics["court"]["extracted"] += len(ext_court)
        metrics["judge"]["extracted"] += len(ext_judge)
        metrics["citation"]["extracted"] += len(ext_cit)

        # Compute true positives via span boundary overlap
        for gs in gold_stat:
            if any(max(gs["start"], e.start) < min(gs["end"], e.end) for e in ext_stat):
                metrics["statute"]["tp"] += 1

        for gs in gold_court:
            if any(max(gs["start"], e.start) < min(gs["end"], e.end) for e in ext_court):
                metrics["court"]["tp"] += 1

        for gs in gold_judge:
            if any(max(gs["start"], e.start) < min(gs["end"], e.end) for e in ext_judge):
                metrics["judge"]["tp"] += 1

        for gs in gold_prec:
            if any(max(gs["start"], e.start) < min(gs["end"], e.end) for e in ext_cit):
                metrics["citation"]["tp"] += 1

    summary = {}
    for cat, val in metrics.items():
        gold_cnt = val["gold"]
        ext_cnt = val["extracted"]
        tp_cnt = val["tp"]
        prec = tp_cnt / max(ext_cnt, 1)
        rec = tp_cnt / max(gold_cnt, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
        summary[cat] = {
            "gold_count": gold_cnt,
            "extracted_count": ext_cnt,
            "true_positives": tp_cnt,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        print(f"  Gold {cat.capitalize()} Validation -> Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} (Gold={gold_cnt}, TP={tp_cnt})")

    return summary


def run_gold_rr_validation(encoder: LegalFeatureEncoder) -> Dict[str, Any]:
    """Validates rhetorical sentence classifier against IL-TUR RR gold sentence annotations."""
    rr_path = ANNOTATIONS_DIR / "CL_dev-00000-of-00001.parquet"
    if not rr_path.exists():
        print(f"Warning: RR annotation file {rr_path} not found. Skipping gold RR evaluation.")
        return {}

    print(f"Running Gold Benchmark Validation against {rr_path.name} (5 full cases, 1,396 sentences)...")
    df = pd.read_parquet(rr_path)

    label_names = {
        0: "Fact", 1: "Issue", 2: "ArgumentPetitioner", 3: "ArgumentRespondent",
        4: "Statute", 5: "Dissent", 6: "PrecedentReliedUpon", 7: "PrecedentNotReliedUpon",
        8: "PrecedentOverruled", 9: "RulingByLowerCourt", 10: "RatioOfTheDecision",
        11: "RulingByPresentCourt", 12: "Other"
    }

    total_sents = 0
    correct_sents = 0
    per_role_tp = {r: 0 for r in RHETORICAL_ROLES}
    per_role_gold = {r: 0 for r in RHETORICAL_ROLES}
    per_role_pred = {r: 0 for r in RHETORICAL_ROLES}

    for idx, row in df.iterrows():
        sents = list(row["text"])
        labels = list(row["labels"])
        n_sents = len(sents)

        for s_idx, sent in enumerate(sents):
            pos_ratio = s_idx / max(1, n_sents - 1)
            pred_role, conf = encoder.rhetorical_analyzer.classify_sentence(sent, position_ratio=pos_ratio)
            gold_num = labels[s_idx]
            gold_role = label_names.get(gold_num, "Other")
            if gold_role in ["PrecedentNotReliedUpon", "PrecedentOverruled"]:
                gold_role = "PrecedentReliedUpon"

            total_sents += 1
            if gold_role in per_role_gold:
                per_role_gold[gold_role] += 1
            if pred_role in per_role_pred:
                per_role_pred[pred_role] += 1

            if pred_role == gold_role:
                correct_sents += 1
                if pred_role in per_role_tp:
                    per_role_tp[pred_role] += 1

    accuracy = correct_sents / max(total_sents, 1)
    f1_scores = []
    for r in RHETORICAL_ROLES:
        prec = per_role_tp[r] / max(per_role_pred[r], 1)
        rec = per_role_tp[r] / max(per_role_gold[r], 1)
        if prec + rec > 0:
            f1_scores.append(2 * prec * rec / (prec + rec))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    print(f"  Gold RR Validation -> Total Sentences: {total_sents} | Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")
    return {
        "total_sentences": total_sents,
        "correct_sentences": correct_sents,
        "overall_accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "roles_evaluated": len(RHETORICAL_ROLES),
    }


def extract_features_for_partition(
    split_name: str, encoder: LegalFeatureEncoder
) -> Tuple[Dict[str, DocumentLegalFeatures], Dict[str, Any]]:
    """Encodes all documents in an IL-TUR-PCR partition and calculates coverage statistics."""
    split_dir = PROCESSED_DATA_DIR / split_name
    docs_path = split_dir / "documents.parquet"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing {docs_path}")

    docs_df = pd.read_parquet(docs_path)
    save_json_path = split_dir / "legal_features.json"
    features_dict = encoder.encode_split(split_name, docs_df, save_path=save_json_path)

    stats = LegalFeatureEncoder.compute_corpus_statistics(features_dict)
    stats["split_name"] = split_name
    return features_dict, stats


def generate_analysis_report(
    gold_lner_stats: Dict[str, Any],
    gold_rr_stats: Dict[str, Any],
    corpus_coverage: List[Dict[str, Any]],
):
    """Generates comprehensive Markdown report summarizing legal feature extraction."""
    report_path = REPORTS_DIR / "legal_feature_analysis.md"

    md = []
    md.append("# Legal Knowledge Feature Extraction and Representation Analysis")
    md.append("")
    md.append("**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  ")
    md.append("**Phase**: Phase 5 – Legal Knowledge Feature Extraction and Representation  ")
    md.append(f"**Date**: November 2026  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append("In Phase 5, we engineered a dedicated, domain-adapted legal knowledge extraction and representation pipeline tailored for the Supreme Court of India judgments from the IL-TUR PCR benchmark. The pipeline comprises four modular extractors:")
    md.append("1. **Statutory Reference Extractor**: High-precision parsing of central acts (IPC, CrPC, CPC, Constitution of India, Evidence Act, NI Act, Arbitration Act, NDPS Act, PC Act, IBC) and normalization into canonical keys (`IPC:302`, `CRPC:482`, `CPC:OVII_R11`, `CONST:ART_21`).")
    md.append("2. **Precedent Citation Extractor**: Extraction and canonical key construction for official legal reporters (AIR, SCC, SCR) and formal High Court / Supreme Court appeal numbers (`APPEAL:CRIMINAL_APPEAL:92:2015`).")
    md.append("3. **Legal Entity Extractor**: Recognition and canonicalization of judicial actors (Courts, Judges, Appellants, Respondents, Counsels, Government Organizations, Geographic Jurisdictions, and Law Enforcement Authorities).")
    md.append("4. **Rhetorical Role Analyzer**: Discourse segmentation and 11-class rhetorical probability distribution profiling aligned with the official IL-TUR RR benchmark.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Gold Benchmark Validation Metrics")
    md.append("")
    md.append("We validated our rule-based and pattern-driven legal knowledge extractors against official gold human annotations from the IL-TUR benchmark without any data leakage or training on test partitions.")
    md.append("")
    md.append("### A. Legal Named Entity & Statute Extraction (IL-TUR `l-ner` Fold 1 Gold Annotations)")
    md.append("")
    md.append("| Target Legal Entity / Provision | Gold Annotations | Extracted Instances | True Positives | Precision | Recall | F1 Score |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cat in ["statute", "court", "judge", "citation"]:
        cdata = gold_lner_stats.get(cat, {})
        md.append(
            f"| **{cat.capitalize()}** | {cdata.get('gold_count', '-')} | "
            f"{cdata.get('extracted_count', '-')} | {cdata.get('true_positives', '-')} | "
            f"**{cdata.get('precision', 0.0):.4f}** | **{cdata.get('recall', 0.0):.4f}** | **{cdata.get('f1', 0.0):.4f}** |"
        )

    md.append("")
    md.append("### B. Rhetorical Role Discourse Classification (IL-TUR `rr` CL_dev Gold Annotations)")
    md.append("")
    md.append(f"- **Corpus Evaluated**: `rr/CL_dev` (5 full judicial decisions, {gold_rr_stats.get('total_sentences', 1396)} sentences)")
    md.append(f"- **Overall Sentence Role Accuracy**: **{gold_rr_stats.get('overall_accuracy', 0.0) * 100:.2f}%** (random chance baseline: 8.33%)")
    md.append(f"- **Macro-F1 Across Roles**: **{gold_rr_stats.get('macro_f1', 0.0):.4f}**")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Dataset Coverage and Extraction Statistics")
    md.append("")
    md.append("Across the IL-TUR PCR benchmark partitions, we evaluated extraction coverage and missing-value rates:")
    md.append("")
    md.append("| Partition | Total Docs | Statute Coverage | Citation Coverage | Entity Coverage | Avg Statutes / Doc | Avg Citations / Doc | Avg Entities / Doc | Unique Statutes | Unique Citations |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in corpus_coverage:
        md.append(
            f"| **{c['split_name']}** | {c['total_documents']:,} | {c['statute_coverage_pct']}% | "
            f"{c['citation_coverage_pct']}% | {c['entity_coverage_pct']}% | {c['avg_statutes_per_doc']} | "
            f"{c['avg_citations_per_doc']} | {c['avg_entities_per_doc']} | {c['unique_statute_provisions_count']:,} | "
            f"{c['unique_citations_count']:,} |"
        )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Legal Feature Representation for Hybrid Retrieval (Phase 6)")
    md.append("")
    md.append("To support Phase 6 ensemble fusion without entanglement, we represent document-pair legal similarity across five orthogonal sub-spaces:")
    md.append("")
    md.append("1. **Statutory Reference Overlap**:")
    md.append("   - `statute_jaccard`: Jaccard similarity of normalized statute keys ($|S_q \\cap S_c| / |S_q \\cup S_c|$)")
    md.append("   - `statute_containment_query`: Directional coverage of query statutes in candidate ($|S_q \\cap S_c| / |S_q|$)")
    md.append("   - `statute_shared_count`: Absolute count of co-cited statutory provisions")
    md.append("2. **Precedent Co-Citation Overlap**:")
    md.append("   - `citation_jaccard`: Jaccard similarity of formal reporter and appeal citations ($|C_q \\cap C_c| / |C_q \\cup C_c|$)")
    md.append("   - `citation_shared_count`: Direct co-citation frequency between query and candidate opinions")
    md.append("3. **Legal Entity and Actor Overlap**:")
    md.append("   - `entity_jaccard`: Overall Jaccard similarity across all detected legal actors")
    md.append("   - `judge_overlap`: Overlap in bench composition (shared presiding justices)")
    md.append("   - `court_overlap`: Jurisdiction concordance (e.g. both originating from Bombay High Court)")
    md.append("   - `org_overlap`: Concordance of governmental or corporate litigants (e.g., State of Maharashtra)")
    md.append("4. **Rhetorical Role Distribution Alignment**:")
    md.append("   - `rhetorical_cosine`: Cosine similarity of the 11-dimensional discourse probability distribution")
    md.append("   - `rhetorical_js_sim`: Complementary Jensen-Shannon distance ($1 - D_{JS}(P_q \\parallel P_c)$)")
    md.append("   - `rhetorical_ratio_diff`: Absolute difference in judicial reasoning / ratio decidendi proportion")
    md.append("5. **Composite Legal Feature Score**:")
    md.append("   - Linear combination providing an initial un-learned legal prior before GBDT ensemble training.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Architectural Separation")
    md.append("")
    md.append("As mandated by system guidelines, all extracted legal features are persisted separately in `legal_features.json` for each split and are completely decoupled from lexical (Phase 3 BM25 / TF-IDF) and dense semantic (Phase 4 InLegalBERT) representations. In Phase 6, they will serve as structured tabular features in a Gradient Boosted Decision Tree (LightGBM / XGBoost) and Logistic Stacking re-ranker.")

    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Generated Phase 5 Analysis Report at {report_path}")


def main():
    print("=================================================================")
    print("PHASE 5: LEGAL KNOWLEDGE FEATURE EXTRACTION & VALIDATION")
    print("=================================================================")

    encoder = LegalFeatureEncoder()

    # 1. Gold Benchmark Validations
    gold_lner_stats = run_gold_lner_validation(encoder)
    gold_rr_stats = run_gold_rr_validation(encoder)

    # 2. Extract on IL-TUR-PCR Primary Dataset
    splits_to_process = ["dev_queries", "dev_candidates", "test_queries", "test_candidates"]
    corpus_coverage = []
    split_features_all = {}

    for split in splits_to_process:
        print(f"\n--- Processing Partition: {split} ---")
        t0 = time.time()
        feats_dict, stats = extract_features_for_partition(split, encoder)
        t1 = time.time()
        stats["runtime_seconds"] = round(t1 - t0, 2)
        corpus_coverage.append(stats)
        split_features_all[split] = len(feats_dict)

    # 3. Save Summary Tables
    stats_json_path = TABLES_DIR / "legal_feature_statistics.json"
    consolidated_stats = {
        "gold_benchmark_lner": gold_lner_stats,
        "gold_benchmark_rr": gold_rr_stats,
        "corpus_coverage": {c["split_name"]: c for c in corpus_coverage},
        "splits_processed": split_features_all,
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_stats, f, indent=2)
    print(f"\nSaved consolidated statistics to {stats_json_path}")

    # Coverage CSV
    cov_df = pd.DataFrame(corpus_coverage)
    cov_csv_path = TABLES_DIR / "legal_feature_coverage.csv"
    cov_df.to_csv(cov_csv_path, index=False)
    print(f"Saved legal feature coverage table to {cov_csv_path}")

    # 4. Generate Markdown Analysis Report
    generate_analysis_report(gold_lner_stats, gold_rr_stats, corpus_coverage)
    print("\nPhase 5 Feature Extraction & Evaluation Complete!")


if __name__ == "__main__":
    main()
