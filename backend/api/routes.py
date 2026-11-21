"""
Flask REST API routes for Legal Case Retrieval and Explainability.
"""

import json
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from backend.services.pdf_service import PDFExtractionService, PDFExtractionError
from backend.services.judipl_explanation_service import JudIPLExplanationService

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health", methods=["GET"])
def health():
    """System health check endpoint."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    cand_count = len(retrieval_service.candidate_docs) if retrieval_service else 0
    return jsonify({
        "status": "healthy",
        "service": "Legal Case Retrieval & Explainability API",
        "version": "1.0.0",
        "models": {
            "lexical": "TF-IDF (Weight: 0.65)",
            "dense": "InLegalBERT (Weight: 0.25)",
            "legal_knowledge": "Indian Legal Knowledge Graphs & Statutes (Weight: 0.10)",
            "bm25": "BM25 (Weight: 0.00)",
        },
        "candidate_pool_size": cand_count,
        "normalization": "Query-level min-max scaling",
        "frozen_checkpoint": "models/checkpoints/weighted_fusion_weights.json",
    })


@api_bp.route("/datasets", methods=["GET"])
def datasets():
    """Returns the independent dataset workspaces available to the dashboard."""
    judipl = current_app.config.get("JUDIPL_SERVICE")
    retrieval = current_app.config.get("RETRIEVAL_SERVICE")
    return jsonify({
        "datasets": [
            {"id": "jud-ipl", "name": "Jud-IPL", "task": "Judgment outcome classification",
             "available": judipl is not None, "metadata": judipl.metadata() if judipl else None},
            {"id": "il-tur", "name": "IL-TUR", "task": "Prior case retrieval",
             "available": retrieval is not None,
             "metadata": {
                 "candidate_pool_size": len(retrieval.candidate_docs) if retrieval else 0,
                 "model_name": "Weighted Linear Score Fusion",
                 "weights": retrieval.weights if retrieval else {},
                 "test_queries_count": 237,
                 "eligible_test_queries": 234,
             }},
        ]
    })


@api_bp.route("/judipl/metrics", methods=["GET"])
def judipl_metrics():
    service = current_app.config.get("JUDIPL_SERVICE")
    if not service:
        return jsonify({"error": "Jud-IPL service unavailable."}), 503
    return jsonify(service.metrics())


@api_bp.route("/iltur/metrics", methods=["GET"])
def iltur_metrics():
    """Returns genuine IL-TUR PCR benchmark evaluation metrics."""
    project_root = current_app.config.get("PROJECT_ROOT", Path.cwd())
    metrics_file = project_root / "results" / "evaluation" / "final_test_metrics.json"
    if not metrics_file.exists():
        return jsonify({"error": "IL-TUR evaluation metrics not found."}), 404
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        current_app.logger.exception("Failed to load IL-TUR metrics")
        return jsonify({"error": "Failed to read IL-TUR metrics."}), 500


@api_bp.route("/judipl/analyze", methods=["POST"])
def judipl_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    model_id = data.get("model_id", "M4")
    service = current_app.config.get("JUDIPL_SERVICE")
    if not service:
        return jsonify({"error": "Jud-IPL service unavailable."}), 503
    try:
        return jsonify(service.predict(text, model_id=model_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Jud-IPL inference failed")
        return jsonify({"error": "Jud-IPL inference failed."}), 500


@api_bp.route("/judipl/extract-pdf", methods=["POST"])
def judipl_extract_pdf():
    """Extract text from an uploaded judgment PDF using genuine pypdf parsing."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No PDF file provided in request. Expected form field 'file'."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"success": False, "error": "Empty filename provided."}), 400

    service = PDFExtractionService()
    try:
        result = service.extract_from_file(uploaded_file.stream, uploaded_file.filename)
        return jsonify(result), 200
    except PDFExtractionError as exc:
        return jsonify({
            "success": False,
            "error": exc.message,
            "scanned": exc.scanned,
            "page_count": exc.page_count,
            "filename": Path(uploaded_file.filename).name
        }), exc.status_code
    except Exception as exc:
        current_app.logger.exception("Unexpected error in PDF extraction")
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred during PDF text extraction.",
            "filename": Path(uploaded_file.filename).name
        }), 500


@api_bp.route("/judipl/explain", methods=["POST"])
def judipl_explain():
    """Generates structured legal explanation of judgment outcome via Gemini."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"success": False, "error": "Judgment text is required for legal explanation."}), 400

    model_id = data.get("model_id", "M4")
    predicted_label = data.get("predicted_label", "")
    probabilities = data.get("probabilities", {})
    metadata = data.get("metadata", {})

    service = JudIPLExplanationService()
    try:
        explanation = service.explain_judgment(
            judgment_text=text,
            model_id=model_id,
            predicted_label=predicted_label,
            probabilities=probabilities,
            metadata=metadata,
        )
        return jsonify({"success": True, "explanation": explanation, **explanation}), 200
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Jud-IPL legal explanation failed")
        return jsonify({
            "success": False,
            "error": "Failed to generate legal explanation due to service error."
        }), 500


@api_bp.route("/explorer", methods=["GET"])
def explorer():
    """Return bounded, dataset-specific metadata for the explorer UI."""
    dataset_id = request.args.get("dataset", "jud-ipl")
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(50, max(1, int(request.args.get("page_size", 10))))
    except (TypeError, ValueError):
        return jsonify({"error": "page and page_size must be integers."}), 400
    search_text = request.args.get("search", "").strip()
    if dataset_id == "jud-ipl":
        service = current_app.config.get("JUDIPL_SERVICE")
        if not service:
            return jsonify({"error": "Jud-IPL service unavailable."}), 503
        return jsonify(service.explorer(page=page, page_size=page_size, search=search_text))
    if dataset_id == "il-tur":
        service = current_app.config.get("RETRIEVAL_SERVICE")
        if not service:
            return jsonify({"error": "IL-TUR retrieval service unavailable."}), 503
        ids = sorted(service.candidate_docs)
        if search_text:
            ids = [case_id for case_id in ids if search_text.casefold() in case_id.casefold()]
        total = len(ids)
        start = (page - 1) * page_size
        rows = []
        for case_id in ids[start:start + page_size]:
            doc = service.candidate_docs[case_id]
            rows.append({
                "candidate_id": case_id,
                "word_count": doc.get("word_count", 0),
                "sentence_count": doc.get("sentence_count", 0),
                "excerpt": doc.get("snippet", ""),
            })
        return jsonify({
            "dataset": "il-tur",
            "page": page,
            "page_size": page_size,
            "total": total,
            "rows": rows,
            "available_fields": ["candidate_id", "word_count", "sentence_count", "excerpt"],
            "limitations": ["Candidate browsing exposes metadata and excerpts; full corpus browsing is not enabled."],
        })
    return jsonify({"error": "Unknown dataset."}), 400


@api_bp.route("/explorer/document", methods=["GET"])
def explorer_document():
    """Return full details for a single judgment or candidate."""
    dataset_id = request.args.get("dataset", "jud-ipl")
    doc_id = request.args.get("id", "").strip()
    if not doc_id:
        return jsonify({"error": "Document ID is required."}), 400

    if dataset_id == "jud-ipl":
        service = current_app.config.get("JUDIPL_SERVICE")
        if not service:
            return jsonify({"error": "Jud-IPL service unavailable."}), 503
        doc = service.get_document(doc_id)
        print(f"DEBUG_DOC: dataset={dataset_id}, id={doc_id}, found={bool(doc)}", flush=True)
        if not doc:
            return jsonify({"error": f"Judgment '{doc_id}' not found."}), 404
        return jsonify({"dataset": "jud-ipl", "document": doc})

    elif dataset_id == "il-tur":
        service = current_app.config.get("RETRIEVAL_SERVICE")
        if not service:
            return jsonify({"error": "IL-TUR retrieval service unavailable."}), 503
        doc = service.candidate_docs.get(doc_id)
        if not doc:
            return jsonify({"error": f"Candidate '{doc_id}' not found."}), 404
        legal = service.candidate_legal_features.get(doc_id, {})
        return jsonify({
            "dataset": "il-tur",
            "document": {
                "candidate_id": doc_id,
                "word_count": doc.get("word_count", 0),
                "sentence_count": doc.get("sentence_count", 0),
                "full_text": doc.get("full_text", ""),
                "snippet": doc.get("snippet", ""),
                "statutory_provisions": legal.get("statutory_provisions", []),
                "precedent_citations": legal.get("precedent_citations", []),
                "courts": legal.get("courts", []),
                "judges": legal.get("judges", []),
            }
        })
    return jsonify({"error": "Unknown dataset."}), 400


@api_bp.route("/search", methods=["POST"])
def search():
    """
    Search legal precedent judgments for a query description or benchmark query ID.
    Supports Phase 11 baseline and optional Gemini augmentation.
    Payload:
    {
        "query": "text or query_id",
        "top_k": 10,
        "filter_statute": optional string,
        "enable_gemini": optional boolean,
        "enable_expansion": optional boolean,
        "enable_reranking": optional boolean
    }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object.", "results": []}), 400
    query = data.get("query", "")
    if not isinstance(query, str):
        return jsonify({"error": "Query must be a string.", "results": []}), 400
    query = query.strip()
    try:
        top_k = int(data.get("top_k", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer.", "results": []}), 400
    if top_k < 1 or top_k > 100:
        return jsonify({"error": "top_k must be between 1 and 100.", "results": []}), 400
    filter_statute = data.get("filter_statute")
    enable_gemini = bool(data.get("enable_gemini", False))
    enable_expansion = bool(data.get("enable_expansion", False))
    enable_reranking = bool(data.get("enable_reranking", False))

    if not query:
        return jsonify({
            "error": "Query cannot be empty. Please provide a legal query description or case ID.",
            "results": [],
        }), 400

    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "IL-TUR retrieval service unavailable."}), 503

    try:
        res = retrieval_service.search(
            query,
            top_k=top_k,
            filter_statute=filter_statute,
            enable_gemini=enable_gemini,
            enable_expansion=enable_expansion,
            enable_reranking=enable_reranking,
        )
        return jsonify(res)
    except Exception:
        current_app.logger.exception("IL-TUR retrieval failed")
        return jsonify({"error": "IL-TUR retrieval failed.", "results": []}), 500


# =========================================================================
# Phase 13: Gemini API Integration Endpoints
# =========================================================================

@api_bp.route("/gemini/status", methods=["GET"])
def gemini_status():
    """Returns Gemini service configuration, health, and availability."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service or not hasattr(retrieval_service, "gemini_service"):
        return jsonify({"configured": False, "status": "uninitialized"}), 200

    gs = retrieval_service.gemini_service
    return jsonify({
        "status": "ready" if gs.is_available else "offline",
        "config": gs.config.to_dict(),
        "metrics": gs.get_operational_metrics(),
    })


@api_bp.route("/gemini/expand-query", methods=["POST"])
def gemini_expand_query():
    """Module A: Analyzes and expands natural-language legal query using Gemini."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "Service unavailable."}), 500

    expansion = retrieval_service.query_expander.expand_query(query)
    return jsonify(expansion)


@api_bp.route("/gemini/extract-reasoning/<case_id>", methods=["GET", "POST"])
def gemini_extract_reasoning(case_id: str):
    """Module B: Extracts grounded ratio decidendi and judicial reasoning using Gemini."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "Service unavailable."}), 500

    cid = case_id.strip()
    c_info = retrieval_service.get_case(cid)
    if not c_info:
        return jsonify({"error": f"Case ID '{cid}' not found."}), 404

    full_text = c_info.get("full_text", "")
    reasoning = retrieval_service.reasoning_extractor.extract_reasoning(cid, full_text)
    return jsonify(reasoning)


@api_bp.route("/gemini/explain/<query_id>/<case_id>", methods=["GET", "POST"])
def gemini_explain(query_id: str, case_id: str):
    """Module D: Generates 8-point evidence-grounded legal explanation for query-precedent pair."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "Service unavailable."}), 500

    qid = query_id.strip()
    cid = case_id.strip()

    # Get query text
    query_text = ""
    for sq in retrieval_service.sample_queries:
        if sq["query_id"] == qid:
            query_text = sq.get("full_text", "")
            break
    if not query_text and request.is_json:
        query_text = request.json.get("query_text", "")

    # Get candidate text
    c_info = retrieval_service.get_case(cid)
    cand_text = c_info.get("full_text", "") if c_info else ""

    explanation = retrieval_service.gemini_explainer.explain_precedent(
        query_id=qid,
        candidate_id=cid,
        query_text=query_text or f"Legal controversy in Case #{qid}",
        candidate_text=cand_text,
        hybrid_score=0.85,
    )
    return jsonify(explanation)


@api_bp.route("/gemini/metrics", methods=["GET"])
def gemini_metrics():
    """Module Performance: Returns latency, tokens, cost, and cache metrics."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "Service unavailable."}), 500
    return jsonify(retrieval_service.gemini_service.get_operational_metrics())


@api_bp.route("/case/<case_id>", methods=["GET"])
def get_case(case_id: str):
    """Retrieves full case details, metadata, statutes, citations, and coram."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    if not retrieval_service:
        return jsonify({"error": "Retrieval service not initialized."}), 500

    case_info = retrieval_service.get_case(case_id.strip())
    if not case_info:
        return jsonify({"error": f"Case ID '{case_id}' not found in candidate repository."}), 404

    return jsonify(case_info)


@api_bp.route("/explanation/<query_id>/<case_id>", methods=["GET"])
def get_explanation(query_id: str, case_id: str):
    """Retrieves structured explanation and score decomposition for a query-candidate pair."""
    explanation_service = current_app.config.get("EXPLANATION_SERVICE")
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")

    if not explanation_service or not retrieval_service:
        return jsonify({"error": "Services not initialized."}), 500

    qid = query_id.strip()
    cid = case_id.strip()

    # Retrieve features from benchmark dataframe if available
    pair_features = {}
    if retrieval_service.df_test_features is not None:
        matched = retrieval_service.df_test_features[
            (retrieval_service.df_test_features["query_id"] == qid) &
            (retrieval_service.df_test_features["candidate_id"] == cid)
        ]
        if not matched.empty:
            pair_features = matched.iloc[0].to_dict()

    exp = explanation_service.explain(qid, cid, pair_features)
    return jsonify(exp)


@api_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Serves official final test evaluation metrics and model comparison data."""
    project_root = current_app.config.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)
    metrics_path = project_root / "results/evaluation/final_test_metrics.json"

    if not metrics_path.exists():
        return jsonify({"error": "Metrics file not found. Run final evaluation pipeline first."}), 404

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    return jsonify(metrics_data)


@api_bp.route("/sample-queries", methods=["GET"])
def get_sample_queries():
    """Provides sample benchmark queries for 1-click exploration in the UI."""
    retrieval_service = current_app.config.get("RETRIEVAL_SERVICE")
    samples = retrieval_service.sample_queries if retrieval_service else []
    return jsonify({"sample_queries": samples})
