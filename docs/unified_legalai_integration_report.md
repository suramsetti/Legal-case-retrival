# Unified LegalAI Platform Integration Report

## Inspection scope

Repository inspection was completed before application changes. The existing project is a Python/Flask legal retrieval application with a static single-page frontend. It already contains independent Jud-IPL and IL-TUR assets; no combined-corpus inference path is required.

## Existing surfaces

| Area | Existing implementation | Integration decision |
|---|---|---|
| Backend | `backend/app.py` Flask application factory | Extend the existing app and blueprint |
| API | `backend/api/routes.py` under `/api` | Preserve retrieval/Gemini routes and add dataset-scoped routes |
| IL-TUR PCR | `backend/services/retrieval_service.py` using `data/processed/IL-TUR-PCR` and frozen retrieval features | Reuse unchanged as the IL-TUR pipeline |
| Jud-IPL models | Phase 20 frozen joblib checkpoints under `models/checkpoints/phase20` | Add a read-only inference service using M4 and M8 checkpoints |
| Jud-IPL evaluation | Phase 21B frozen predictions and Phase 21B-C corrected outputs | Serve metrics from existing correction artifacts; do not retrain or regenerate predictions |
| Jud-IPL data | `data/processed/judipl`, including phase18c three-class partitions | Use only for metadata/explorer and model inference |
| Frontend | `frontend/index.html`, a standalone React 18/Tailwind CDN application | Upgrade the existing page in place rather than introduce a second frontend |
| Persistence | No existing history database identified | Use bounded in-memory history unless SQLite is explicitly needed later |
| Build tooling | No `package.json`, Vite source tree, or frontend build pipeline identified | Do not claim Vite is present; retain the working Flask-served static app |

## Frozen model and evaluation provenance

- M4: `m4_section_aware_tfidf.joblib`
- M8: `m8_hybrid_fusion.joblib`
- Canonical classes: `Accepted`, `Other`, `Rejected`
- Phase 21B-C eligible evaluation population: 6,311 Jud-IPL documents
- Phase 21B-C outputs are under `results/phase21b_correction/`

## Constraints and unavailable surfaces

- The existing repository does not contain a separate React/Vite source application or package manifest.
- The IL-TUR retrieval implementation is available and remains independent from Jud-IPL classification.
- The Jud-IPL frozen checkpoints support text classification inference. They do not provide a trained PDF parser, so PDF uploads require text extraction before classification; unsupported or empty extraction must be surfaced as an explicit API error.
- No authentication layer is present and none is introduced.

## Planned integration

1. Add a read-only Jud-IPL inference service that loads frozen M4/M8 checkpoints once.
2. Add dataset metadata, model metrics, Jud-IPL analysis, and unified dashboard routes to the existing Flask blueprint.
3. Keep IL-TUR search on its existing service and route.
4. Replace the current static presentation with a unified dataset-aware dashboard while preserving the Flask entry point.
5. Add backend tests for route contracts, model isolation, and frozen checkpoint loading.
