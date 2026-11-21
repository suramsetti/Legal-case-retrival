"""Creates the interactive Jupyter notebook notebooks/05_legal_feature_extraction.ipynb."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 5: Legal Knowledge Feature Extraction & Representation\n",
                "\n",
                "**Project**: Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning  \n",
                "**Dataset**: Indian Legal Prior Case Retrieval (IL-TUR PCR)  \n",
                "**Goal**: Implement and evaluate structured legal knowledge extractors for:\n",
                "1. **Statutory Reference Extraction** (IPC, CrPC, CPC, Constitution, and special Acts)\n",
                "2. **Precedent Citation Extraction** (AIR, SCC, SCR, and formal Appeal numbers)\n",
                "3. **Legal Named Entity Extraction** (Courts, Judges, Counsels, Parties, Organizations, Locations)\n",
                "4. **Rhetorical Role Analysis** (11-class discourse distribution profiling)\n",
                "5. **Pairwise Legal Overlap Encoding** (Jaccard, containment, JS-divergence, composite legal priors)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import json\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "# Ensure project root is in path\n",
                "PROJECT_ROOT = Path(\"..\").resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "from src.legal_features.statute_extractor import StatuteExtractor\n",
                "from src.legal_features.citation_extractor import CitationExtractor\n",
                "from src.legal_features.entity_extractor import LegalEntityExtractor\n",
                "from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, RHETORICAL_ROLES\n",
                "from src.legal_features.legal_feature_encoder import LegalFeatureEncoder\n",
                "\n",
                "print(\"All Phase 5 legal feature modules imported successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Statutory Reference Extraction & Normalization\n",
                "\n",
                "Indian court judgments cite statutory provisions in diverse syntactic structures (e.g. `Section 302 IPC`, `Section 304B read with Section 34 of the Indian Penal Code`, `Article 21 of the Constitution`, `Section 482 CrPC`, `Order VII Rule 11 CPC`).\n",
                "The `StatuteExtractor` parses these provisions with exact character spans and maps them to canonical keys."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "stat_extractor = StatuteExtractor()\n",
                "sample_text = (\n",
                "    \"The appellant was convicted under Section 302 read with Section 34 of the Indian Penal Code. \"\n",
                "    \"Subsequently, a petition under Section 482 of the Code of Criminal Procedure was preferred \"\n",
                "    \"invoking Article 21 and Article 226 of the Constitution of India. \"\n",
                "    \"In civil proceedings, an application under Order VII Rule 11 of the Code of Civil Procedure was filed.\"\n",
                ")\n",
                "\n",
                "refs = stat_extractor.extract(sample_text)\n",
                "print(f\"Extracted {len(refs)} statutory references:\")\n",
                "for r in refs:\n",
                "    print(f\"  • Key: {r.normalized_key:<20} | Act: {r.act:<25} | Span: [{r.start}:{r.end}] | Raw: '{r.raw}'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Precedent Citation Extraction & Canonicalization\n",
                "\n",
                "Indian jurisprudence heavily relies on formal reporter citations (SCC, AIR, SCR) and official appeal numbers (`Criminal Appeal No. ...`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cit_extractor = CitationExtractor()\n",
                "citation_text = (\n",
                "    \"Reliance was placed on the landmark decision in (1973) 4 SCC 225 and AIR 1980 SC 1160. \"\n",
                "    \"The earlier ruling reported in [1962] 2 SCR 100 was distinguished. \"\n",
                "    \"The present proceedings originate from Criminal Appeal No. 92 of 2015.\"\n",
                ")\n",
                "\n",
                "cits = cit_extractor.extract(citation_text)\n",
                "print(f\"Extracted {len(cits)} formal legal citations:\")\n",
                "for c in cits:\n",
                "    print(f\"  • Key: {c.normalized_key:<25} | Reporter: {c.reporter:<15} | Year: {c.year} | Raw: '{c.raw}'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Legal Named Entity Extraction\n",
                "\n",
                "Extracts key judicial actors: Courts, Judges, Parties, Counsels, Government Organizations, Jurisdictions, and Authorities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ent_extractor = LegalEntityExtractor()\n",
                "entity_text = (\n",
                "    \"IN THE SUPREME COURT OF INDIA\\n\"\n",
                "    \"Hon'ble Mr. Justice D.Y. Chandrachud and Justice R. Banumathi.\\n\"\n",
                "    \"Between: State of Maharashtra (Appellant) and Union of India (Respondent).\\n\"\n",
                "    \"Mr. Harish Salve, learned Senior Counsel appeared for the appellant in New Delhi.\\n\"\n",
                "    \"The judgment of the High Court of Judicature at Bombay was challenged.\"\n",
                ")\n",
                "\n",
                "entities = ent_extractor.extract(entity_text)\n",
                "print(f\"Extracted {len(entities)} legal entities:\")\n",
                "for e in entities:\n",
                "    print(f\"  • [{e.category:<10}] {e.name:<30} | Key: {e.normalized_key}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Rhetorical Role Analysis & Discourse Profiling\n",
                "\n",
                "Judgments follow functional stages of legal reasoning. The `RhetoricalAnalyzer` classifies sentence-level discourse and produces a normalized 11-dimensional probability profile."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "analyzer = RhetoricalAnalyzer()\n",
                "sample_sentences = [\n",
                "    \"The case of the prosecution is that on 15th August 2012, the incident took place.\",\n",
                "    \"The short question that arises for our consideration is whether the conviction is sustainable.\",\n",
                "    \"Learned senior counsel appearing on behalf of the appellant contended that witness testimonies were contradictory.\",\n",
                "    \"A plain reading of Section 302 of the Indian Penal Code stipulates the requisite mens rea.\",\n",
                "    \"In our considered opinion, the chain of circumstantial evidence remains incomplete.\",\n",
                "    \"In the result, the appeal is allowed and the judgment of the High Court is set aside.\"\n",
                "]\n",
                "\n",
                "sentence_roles, distribution = analyzer.analyze_document(sample_sentences)\n",
                "\n",
                "print(\"Sentence-Level Classifications:\")\n",
                "for sr in sentence_roles:\n",
                "    print(f\"  [{sr.predicted_role:<22}] ({sr.confidence:.2f}) {sr.text[:70]}...\")\n",
                "\n",
                "print(\"\\nDocument-Level Rhetorical Role Distribution Vector:\")\n",
                "for role, prob in zip(RHETORICAL_ROLES, distribution):\n",
                "    if prob > 0:\n",
                "        print(f\"  {role:<22}: {prob * 100:.1f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Pairwise Legal Overlap & Feature Encoding\n",
                "\n",
                "Combines all four extractors into a pairwise similarity vector between a query case and a candidate case for hybrid re-ranking in Phase 6."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "encoder = LegalFeatureEncoder()\n",
                "\n",
                "case_q = (\n",
                "    \"IN THE SUPREME COURT OF INDIA\\n\"\n",
                "    \"Hon'ble Justice Chandrachud.\\n\"\n",
                "    \"The appellant was convicted under Section 302 read with Section 34 of the Indian Penal Code. \"\n",
                "    \"Reliance was placed on the decision in (1973) 4 SCC 225. \"\n",
                "    \"In our considered opinion, the circumstantial evidence is insufficient. \"\n",
                "    \"In the result, the appeal is allowed.\"\n",
                ")\n",
                "\n",
                "case_c = (\n",
                "    \"IN THE SUPREME COURT OF INDIA\\n\"\n",
                "    \"Hon'ble Justice Chandrachud.\\n\"\n",
                "    \"The accused was prosecuted under Section 302 IPC. \"\n",
                "    \"The Court cited (1973) 4 SCC 225 and AIR 1980 SC 1160. \"\n",
                "    \"We are of the view that the conviction must be upheld. \"\n",
                "    \"In the result, the appeal is dismissed.\"\n",
                ")\n",
                "\n",
                "doc_q = encoder.encode_document(\"Query_101\", case_q)\n",
                "doc_c = encoder.encode_document(\"Cand_502\", case_c)\n",
                "\n",
                "pairwise = encoder.compute_pairwise_features(doc_q, doc_c)\n",
                "\n",
                "print(\"Pairwise Legal Similarity Features:\")\n",
                "for feat_name, val in pairwise.items():\n",
                "    print(f\"  {feat_name:<28}: {val:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Gold Benchmark Validation & Corpus Coverage\n",
                "\n",
                "Loading empirical validation metrics evaluated against IL-TUR `l-ner` and `rr` gold annotations, alongside coverage statistics across the primary PCR dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cov_path = PROJECT_ROOT / \"results\" / \"tables\" / \"legal_feature_coverage.csv\"\n",
                "stats_path = PROJECT_ROOT / \"results\" / \"tables\" / \"legal_feature_statistics.json\"\n",
                "\n",
                "if cov_path.exists():\n",
                "    cov_df = pd.read_csv(cov_path)\n",
                "    print(\"=== Legal Feature Coverage Across PCR Partitions ===\")\n",
                "    display(cov_df)\n",
                "\n",
                "if stats_path.exists():\n",
                "    with open(stats_path, \"r\", encoding=\"utf-8\") as f:\n",
                "        stats = json.load(f)\n",
                "    print(\"\\n=== Gold Benchmark Validation (L-NER) ===\")\n",
                "    print(pd.DataFrame(stats.get(\"gold_benchmark_lner\", {})).T)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

target_nb = Path("notebooks/05_legal_feature_extraction.ipynb")
target_nb.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Created {target_nb}")

