"""
Flask Application Entry Point for Legal Case Retrieval and Explainability.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, send_from_directory, render_template_string
from flask_cors import CORS

from backend.api.routes import api_bp
from backend.services.retrieval_service import RetrievalService
from backend.services.judipl_service import JudIPLService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app() -> Flask:
    """Application factory for the Legal Retrieval web application."""
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize domain services
    try:
        retrieval_service = RetrievalService(project_root=PROJECT_ROOT)
    except ImportError as exc:
        app.logger.warning("Optional IL-TUR retrieval service unavailable: %s", exc)
        retrieval_service = None
    try:
        from backend.services.explanation_service import ExplanationService
        explanation_service = ExplanationService(project_root=PROJECT_ROOT)
    except ImportError as exc:
        app.logger.warning("Optional explainability service unavailable: %s", exc)
        explanation_service = None
    judipl_service = JudIPLService(project_root=PROJECT_ROOT)

    app.config["PROJECT_ROOT"] = PROJECT_ROOT
    app.config["RETRIEVAL_SERVICE"] = retrieval_service
    app.config["EXPLANATION_SERVICE"] = explanation_service
    app.config["JUDIPL_SERVICE"] = judipl_service

    # Register API blueprint
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        """Serves the single-page React frontend application."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return send_from_directory(str(FRONTEND_DIR), "index.html")
        return "Legal Case Retrieval API is running. Access endpoints under /api/"

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
