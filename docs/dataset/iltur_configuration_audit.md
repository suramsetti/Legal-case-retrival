# IL-TUR Dataset Configuration Audit

**Dataset Identifier**: `Exploration-Lab/IL-TUR`  
**Full Title**: IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning  
**Authors/Affiliation**: Abhinav Joshi, Shounak Paul, Akshat Sharma, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi (IIT Kanpur & IIT Kharagpur; ACL 2024 / arXiv:2407.05399)  
**Audit Date**: November 2026  
**License**: CC BY-NC-SA 4.0 (Non-commercial research use only)  
**Access Model**: Gated / Controlled-access Hugging Face Repository  

---

## 1. Executive Summary & Access Mechanics

`Exploration-Lab/IL-TUR` is an umbrella benchmark suite developed to evaluate natural language processing (NLP) and machine learning models on Indian legal jurisprudence. 

### Key Access Findings
* **Gated Repository**: The dataset is hosted as a **gated resource** on the Hugging Face Hub (`https://huggingface.co/datasets/Exploration-Lab/IL-TUR`). Users must manually log into Hugging Face, accept the data agreement terms (acknowledging non-commercial research use), and authenticate via an API token (`huggingface-cli login` or `HF_TOKEN`).
* **Custom Loader Script**: The repository utilizes a custom Python dataset loading script (`revision="script"` with `trust_remote_code=True`). Standard dynamic discovery methods without authentication or in isolated offline environments will trigger access errors.
* **Multi-task Organization**: The benchmark encapsulates 8 distinct tasks spanning classification, sequence labeling, structured extraction, summarization, translation, and information retrieval.

---

## 2. Available Dataset Configurations

When querying configurations using the Hugging Face `datasets` library:
```python
from datasets import get_dataset_config_names

# Note: Requires prior Hugging Face login authentication
configs = get_dataset_config_names("Exploration-Lab/IL-TUR")
```

The 8 officially defined configurations and their foundational purposes are:

| Configuration ID | Task Name | Domain & Nature | Underlying Source Corpus |
| :--- | :--- | :--- | :--- |
| `l-ner` | Legal Named Entity Recognition | Token-level sequence labeling (12 legal entity classes) | Indian Supreme Court Judgments |
| `rr` | Rhetorical Role Prediction | Sentence-level structural segmentation & classification | Indian Supreme Court Judgments |
| `cjpe` | Court Judgment Prediction & Explanation | Binary document classification + sentence extraction | ILDC (Indian Legal Documents Corpus) |
| `bail` | Bail Prediction | Binary document classification (in Hindi) | HLDC (Hindi Legal Documents Corpus) |
| `lsi` | Legal Statute Identification | Multi-label classification (statutory provisions) | ILSI Corpus |
| `pcr` | Prior Case Retrieval | Ad-hoc document ranking & precedent retrieval | IL-PCR / IndianKanoon SC Cases |
| `summ` | Legal Document Summarization | Abstractive / Extractive document summarization | IN-Abs / IN-Ext Judgments |
| `l-mt` | Legal Machine Translation | Sentence & passage translation across 9 Indian languages | Legal Text Translation Corpus |

---

## 3. Configuration Inspection: Schemas, Splits, and Data Types

### 3.1. `pcr` — Prior Case Retrieval (Information Retrieval Task)

* **Purpose**: Given a query case judgment (with direct citations masked), retrieve and rank relevant precedent judgments from a historical candidate pool.
* **Splits**:
  * `train_queries` (~827 cases)
  * `train_candidates` (~4,320 cases)
  * `dev_queries` / `val_queries` (~118 cases)
  * `dev_candidates` (~1,023 cases)
  * `test_queries` (~234 - 237 cases)
  * `test_candidates` (~1,727 cases)
* **Total Document Pool**: ~8,000 Supreme Court of India judgments.
* **Schema & Feature Specification**:
  * **Query Splits (`train_queries`, `dev_queries`, `test_queries`)**:
    * `id`: `string` — Unique IndianKanoon case identifier (e.g., `"1234567"`).
    * `text`: `List[string]` — Judgment text segmented into ordered sentence tokens.
    * `relevant_candidates`: `List[string]` — Ground-truth list of relevant candidate document IDs cited by this case.
  * **Candidate Splits (`train_candidates`, `dev_candidates`, `test_candidates`)**:
    * `id`: `string` — Unique IndianKanoon case identifier.
    * `text`: `List[string]` — Candidate judgment text segmented into ordered sentence tokens.
    * `relevant_candidates`: `None` / empty list.
* **Sample Structure (Illustrative)**:
  ```json
  {
    "id": "1849201",
    "text": [
      "This appeal is directed against the judgment of the High Court of Punjab and Haryana...",
      "The appellant was convicted under Section 302 read with Section 34 of the Indian Penal Code..."
    ],
    "relevant_candidates": ["143892", "981240"]
  }
  ```

---

### 3.2. `l-ner` — Legal Named Entity Recognition

* **Purpose**: Sequence labeling for legal actors and entities.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `tokens`: `List[string]`
  * `ner_tags`: `List[int]` (BIO-formatted tags spanning 12 legal categories: Appellant, Respondent, Judge, Appellant Counsel, Respondent Counsel, Court, Authority, Witness, Statute, Precedent, Date, Case Number)

---

### 3.3. `rr` — Rhetorical Role Prediction

* **Purpose**: Functional sentence classification characterizing legal reasoning stages.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `sentences`: `List[string]`
  * `labels`: `List[string]` (Roles: `FAC` (Facts), `ARG` (Arguments), `PRE` (Precedent), `STA` (Statute), `RATIO` (Ratio of Decision), `RLC` (Ruling by Lower Court), `RPC` (Ruling by Present Court))

---

### 3.4. `cjpe` — Court Judgment Prediction with Explanation

* **Purpose**: Predict whether an appeal is accepted or rejected and isolate explanatory sentences.
* **Splits**: `train`, `validation`, `test` (deriving from ILDC-multi and ILDC-expert)
* **Schema**:
  * `id`: `string`
  * `text`: `string`
  * `label`: `int` (Binary classification: `0` = Rejected / Dismissed, `1` = Accepted / Allowed)
  * `explanation`: `List[string]` (Salient sentences extracted by legal experts justifying the judgment)

---

### 3.5. `bail` — Bail Prediction

* **Purpose**: Binary prediction of bail application grant/refusal from district court case records in Hindi.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `text`: `string` (De-identified case text in Hindi script)
  * `label`: `int` (Binary classification: `0` = Bail Denied, `1` = Bail Granted)
  * `salience_labels`: `List[int]` (Judge-aligned salience sentence indicators)

---

### 3.6. `lsi` — Legal Statute Identification

* **Purpose**: Multi-label classification mapping facts of a case to relevant statutory sections.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `text`: `string` (Factual summary of the offense or legal dispute)
  * `labels`: `List[string]` (Applicable statutory provisions, e.g., `["IPC_302", "IPC_34"]`)

---

### 3.7. `summ` — Legal Summarization

* **Purpose**: Long-document summarization of judicial opinions.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `text`: `string` (Full judgment text)
  * `summary`: `string` (Reference headnote or legal summary)

---

### 3.8. `l-mt` — Legal Machine Translation

* **Purpose**: Multilingual translation of legal texts between English and 9 Indic languages.
* **Splits**: `train`, `validation`, `test`
* **Schema**:
  * `id`: `string`
  * `translation`: `Dict[string, string]` (Source and target text pairs)

---

## 4. Retrieval Requirements & Verification Matrix

| Requirement | Present in IL-TUR? | Configuration Details |
| :--- | :---: | :--- |
| **Query Case IDs** | **YES** | Available in `pcr` (`train_queries`, `dev_queries`, `test_queries` via `id`) |
| **Candidate Case IDs** | **YES** | Available in `pcr` (`train_candidates`, `dev_candidates`, `test_candidates` via `id`) |
| **Relevant Candidate IDs** | **YES** | Available in `pcr` query splits via `relevant_candidates` field |
| **Relevance Judgments** | **YES** (Binary) | Ground truth citations between queries and candidates serve as gold positive matches |
| **Indian Legal Judgment Text** | **YES** | Full Supreme Court of India judgment text represented as sentence lists |
| **Genuine Prior Case Retrieval (PCR)** | **YES** | Explicitly supported under the `pcr` configuration |

### Distinction Between Classification Labels and Retrieval Relevance
> [!IMPORTANT]
> The audit explicitly confirms that **classification labels in `cjpe`, `bail`, and `lsi` are NOT retrieval relevance labels**:
> * `cjpe` binary labels (`0/1`) indicate appeal outcome (allowed vs dismissed).
> * `bail` binary labels (`0/1`) indicate bail decision.
> * `lsi` labels are multi-label statutory identifiers.
> * **Only the `pcr` configuration** provides genuine Information Retrieval (IR) query-candidate mappings with citation-based ground-truth relevance sets (`relevant_candidates`).

---

## 5. Prior Case Retrieval (PCR) Task Analysis

### Task Setup & Characteristics
1. **Query-Candidate Relationship**: Newer Supreme Court cases cite historical precedent cases. The citations are parsed and masked from the query text.
2. **Corpus Size**: Sourced from IndianKanoon Supreme Court judgments (8,000+ total judgments).
3. **Retrieval Formulation**:
   * Given query case $q \in Q$, retrieve a ranked list $R_q = [d_1, d_2, \dots, d_K]$ from candidate pool $C$.
   * Evaluate against relevant precedent set $P_q = \text{relevant\_candidates}$.
4. **Primary Benchmark Metric**: Micro-averaged F1@K, complemented by Mean Reciprocal Rank (MRR), Precision@K, Recall@K, and NDCG@K.

---

## 6. Suitability Evaluation for the Legal Case Retrieval Project

### Advantages of IL-TUR
1. **Genuine Indian Legal Retrieval Benchmark**: The `pcr` configuration is an established, peer-reviewed (ACL 2024) benchmark specifically dedicated to Indian legal precedent retrieval.
2. **Multi-Task Complementary Data**: Other configurations (`rr` for rhetorical roles, `l-ner` for legal entities, `lsi` for statute identification) can be directly leveraged to train or evaluate our proposed **Legal Knowledge** feature extractors (`src/legal_features`).

### Constraints & Caveats
1. **Gating & Authentication**: Downloading requires Hugging Face authentication credentials and explicit user agreement on the Hugging Face web interface.
2. **Licensing**: Distributed under `CC BY-NC-SA 4.0`, restricting usage to non-commercial academic research.
3. **Binary Relevance Only**: Relevance in `pcr` is binary (1 if cited, 0 if not cited). It does not provide graded/continuous relevance judgments.
4. **Domain Scope**: IL-TUR focuses primarily on general Supreme Court of India jurisprudence, whereas our project workspace includes references to `Jud-IPL` (Indian Patent / Intellectual Property Law judgments).

---

## 7. Conclusion & Recommendations

### Final Verdict:
* **Is a genuine PCR task available in IL-TUR?** **Yes.** The `pcr` configuration directly models prior case precedent retrieval with query IDs, candidate IDs, judgment texts, and ground-truth relevant candidates.
* **Is IL-TUR directly suitable for our project?**
  * **For Indian Case Precedent Retrieval & BERT Finetuning**: **Yes**, `Exploration-Lab/IL-TUR` (configuration `pcr`) provides a solid benchmark.
  * **For Standalone Offline / Specialized Evaluation**: A separate local retrieval benchmark (such as the pre-configured `Jud-IPL` dataset or AILA / FIRE legal retrieval tracks) is strongly recommended alongside or as the primary offline benchmark to avoid Hugging Face gating friction and ensure domain alignment with patent/specialized legal corpora.

### Next Steps for the Project:
1. If using IL-TUR, obtain and configure a Hugging Face User Access Token (`HF_TOKEN`) and accept terms on `Exploration-Lab/IL-TUR`.
2. Inspect the local `data/raw/Jud-IPL` directory to determine its internal structure and decide whether `Jud-IPL` serves as the primary dataset while IL-TUR serves as an external comparative benchmark.

