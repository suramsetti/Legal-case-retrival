# Enhanced Legal Document Similarity and Case Retrieval

**Enhanced Legal Document Similarity and Case Retrieval Using Hybrid BERT, Legal Knowledge, and Ensemble Learning** is a modular framework designed to advance information retrieval and semantic case matching for legal jurisprudence.

---

## 🏛️ Project Overview

Legal case retrieval requires capturing nuanced judicial semantics, domain-specific statutory references, precedent citations, and structural case elements. This project implements a hybrid approach combining:
- **Transformer Embeddings (BERT / Domain-specific Legal BERT variants)** for deep contextual representation.
- **Legal Knowledge & Feature Engineering** to extract legal citations, statutes, court hierarchies, and domain entities.
- **Hybrid Similarity & Ensemble Learning** to rank and retrieve relevant legal precedents accurately.
- **Explainability & Evaluation** to provide transparent, interpretable ranking metrics for judicial decision support.

---

## 📁 Repository Structure

```text
legal-case-retrieval/
├── data/
│   ├── raw/Jud-IPL/          # Original dataset (Jud-IPL legal case repository)
│   ├── interim/              # Intermediate data undergoing transformation
│   ├── processed/            # Cleaned, tokenized, and model-ready data
│   └── external/             # External knowledge sources (statutes, acts, lexicons)
├── notebooks/                # Exploratory data analysis and experimental prototyping
├── src/                      # Core application source code
│   ├── data/                 # Data loading and ingestion pipelines
│   ├── preprocessing/        # Legal text cleaning, tokenization, and segmentation
│   ├── embeddings/           # Embedding generation (BERT, dense representations)
│   ├── retrieval/            # First-stage and dense retrieval mechanisms
│   ├── similarity/           # Similarity computation functions and metrics
│   ├── legal_features/       # Legal entity extraction, citation parsing, domain features
│   ├── ranking/              # Re-ranking models and ensemble learning
│   ├── evaluation/           # Retrieval metrics (Precision@K, Recall@K, MAP, NDCG)
│   └── explainability/       # Interpretability, feature attribution, and explanations
├── models/
│   ├── bert/                 # Fine-tuned or specialized legal BERT weights
│   ├── embeddings/           # Precomputed index and vector stores
│   └── checkpoints/          # Training checkpoints
├── results/
│   ├── figures/              # Generated plots and visualization charts
│   ├── tables/               # Tabular benchmark outputs
│   ├── evaluation/           # Detailed evaluation logs and metric summaries
│   └── reports/              # Final research summaries and documentation
├── backend/                  # API service layer
│   ├── api/                  # API endpoints and route definitions
│   ├── services/             # Core business and retrieval services
│   ├── schemas/              # Pydantic request and response schemas
│   └── utils/                # API helpers and middleware
├── frontend/                 # Interactive user interface
├── configs/                  # Modular configuration files
├── docs/                     # Documentation
│   ├── architecture/         # System architecture and design diagrams
│   ├── dataset/              # Dataset documentation and schema notes
│   └── research/             # Literature review, experiment logs, notes
├── tests/                    # Automated testing suite
│   ├── unit/                 # Unit tests for individual components
│   └── integration/          # End-to-end integration tests
├── logs/                     # Application and execution logs
├── scripts/                  # Automation, execution, and utility scripts
├── .gitignore                # Git exclusions
├── config.yaml               # Global configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview and documentation
```

---

## 🧩 Planned Modules

1. **`src/data` & `src/preprocessing`**:
   - Ingestion of raw legal documents (Jud-IPL dataset).
   - Legal text normalization, paragraph segmentation, and legal boilerplate filtering.
2. **`src/embeddings` & `models/`**:
   - Dense representations using BERT and legal-domain transformers (e.g., Legal-BERT).
   - Efficient indexing for document representations.
3. **`src/legal_features`**:
   - Extraction of case citations, statutory references, judge opinions, and jurisdiction features.
4. **`src/retrieval`, `src/similarity` & `src/ranking`**:
   - Multi-stage retrieval combining lexical and dense similarity.
   - Ensemble re-ranking incorporating legal domain signals.
5. **`src/evaluation` & `src/explainability`**:
   - Benchmarking via Mean Average Precision (MAP), NDCG@K, Recall@K.
   - Highlighting influential statutory sections and precedent matches.
6. **`backend/` & `frontend/`**:
   - Scalable RESTful API serving retrieval pipelines.
   - Search interface for case queries and precedent analysis.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Recommended: A dedicated virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
System parameters and paths are configured in [`config.yaml`](file:///c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval/config.yaml).

