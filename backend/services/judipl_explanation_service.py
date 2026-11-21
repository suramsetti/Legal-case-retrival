"""
Jud-IPL Legal Judgment Explanation Service.
Integrates OpenAI (GPT-4.1 Mini) and Google Gemini to explain multi-class judgment outcomes in natural language.
Provides evidence grounding, prediction agreement analysis, and deterministic heuristic fallback.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.services.openai_service import OpenAIService
from backend.services.gemini_service import GeminiService

logger = logging.getLogger("JudIPLExplanationService")

SYSTEM_INSTRUCTION = """You are a legal judgment explanation assistant.

Analyze the supplied Indian court judgment and explain the legal reasoning behind the decision.

Use only the supplied judgment as the primary source.

Identify:

1. Case background.
2. Main legal issue.
3. Arguments of the parties.
4. Relevant statutory provisions.
5. Court's reasoning.
6. Final judicial disposition.
7. Important paragraphs supporting the disposition.
8. A concise explanation in simple language.

Compare the court's actual final disposition with the supplied AI classifier prediction.

Explain whether they agree or disagree.

If they disagree, explicitly identify the mismatch.

Do not invent legal provisions, citations, paragraph references, judicial observations, or case facts.

If information is unavailable in the supplied judgment, explicitly state that it is not available.

Clearly distinguish the court's actual decision from the AI model's prediction.

Do not provide personalized legal advice or claim to replace a qualified lawyer."""


class JudIPLExplanationService:
    """Explains Jud-IPL legal judgment outcomes using OpenAI / Google Gemini."""

    def __init__(self, llm_service: Optional[Any] = None):
        if llm_service is not None:
            self.llm_service = llm_service
            self.provider_name = getattr(llm_service, "provider", "LLM")
        else:
            # Check OpenAI first, then Gemini
            openai_svc = OpenAIService()
            if openai_svc.is_available:
                self.llm_service = openai_svc
                self.provider_name = f"OpenAI ({openai_svc.config.model_name})"
            else:
                gemini_svc = GeminiService()
                self.llm_service = gemini_svc
                self.provider_name = gemini_svc.active_provider if gemini_svc.is_available else "Deterministic Fallback"

    @property
    def is_available(self) -> bool:
        return getattr(self.llm_service, "is_available", False)

    @property
    def is_gemini_available(self) -> bool:
        return self.is_available

    def explain_judgment(
        self,
        judgment_text: str,
        model_id: str,
        predicted_label: str,
        probabilities: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive, structured explanation of the judgment reasoning,
        operative disposition, and agreement with the AI model prediction.
        """
        if not judgment_text or not judgment_text.strip():
            raise ValueError("Judgment text cannot be empty.")

        text_clean = judgment_text.strip()
        metadata = metadata or {}

        # Limit context if exceedingly long while preserving header and operative conclusion
        prompt_text = text_clean
        if len(text_clean) > 40000:
            prompt_text = (
                text_clean[:20000]
                + "\n\n[... middle sections omitted for prompt length preservation ...]\n\n"
                + text_clean[-20000:]
            )

        prob_accepted = probabilities.get("Accepted", 0.0)
        prob_other = probabilities.get("Other", 0.0)
        prob_rejected = probabilities.get("Rejected", 0.0)

        prompt = f"""JUDGMENT TEXT:
\"\"\"
{prompt_text}
\"\"\"

AI CLASSIFIER INFERENCE CONTEXT:
- Model Architecture: {model_id}
- Predicted Disposition: {predicted_label}
- Class Probabilities:
  * Accepted: {prob_accepted:.4f}
  * Other: {prob_other:.4f}
  * Rejected: {prob_rejected:.4f}
- Document Metadata: {json.dumps(metadata, ensure_ascii=False)}

TASK:
Analyze the supplied judgment and output a valid JSON object matching the following structure:
{{
  "case_summary": "string (Short, clear summary of the factual background and nature of the dispute)",
  "legal_issues": ["list of strings (Main legal questions considered by the court)"],
  "court_reasoning": "string (Detailed explanation of why the court reached its conclusion based on law and evidence)",
  "legal_provisions": [
    {{
      "provision": "string (e.g. Order VII Rule 11 CPC or Section 302 IPC)",
      "relevance": "string (How this provision was applied to resolve the controversy)"
    }}
  ],
  "supporting_paragraphs": [
    {{
      "paragraph_ref": "string (e.g. Paragraph 15 or Operative Order)",
      "text": "string (Direct quotation or relevant excerpt from the judgment)"
    }}
  ],
  "actual_court_outcome": "string (Operative final judicial disposition, e.g. 'Rejected (Appeals Dismissed)', 'Accepted (Appeal Allowed)', or 'Other (Remanded / Interlocutory)')",
  "model_prediction": "{predicted_label}",
  "prediction_agreement": "string (Must be exactly one of: 'Match', 'Mismatch', 'Unclear')",
  "agreement_explanation": "string (Detailed comparison explaining why the AI prediction matches, mismatches, or is unclear relative to the actual operative disposition)",
  "overall_explanation": "string (Concise, simple plain-language answer to: 'Why did the court give this decision?')",
  "limitations_and_uncertainty": "string (Procedural posture, scope limitations, or caveats regarding machine learning classifications)"
}}

CRITICAL RULES:
1. The court's actual outcome MUST be derived from the operative final decision, not merely from quoted precedents or arguments.
2. For cases where appeals are dismissed, the operative outcome is Rejected. Where appeals are allowed, it is Accepted. Where remanded or interlocutory, it is Other. If genuinely ambiguous, return 'Unclear'.
3. Do not invent statutory sections or citations not present in the judgment text.
4. Return ONLY the JSON object without markdown fences or extra prose.
"""

        def fallback_generator() -> Dict[str, Any]:
            logger.info("Using deterministic legal extraction fallback for Jud-IPL explanation.")
            return self._generate_heuristic_fallback(
                text=text_clean,
                model_id=model_id,
                predicted_label=predicted_label,
                probabilities=probabilities,
                metadata=metadata,
            )

        explanation = self.llm_service.generate_json(
            prompt=prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            schema_validator=self._validate_schema,
            fallback_factory=fallback_generator,
        )

        if not explanation:
            explanation = fallback_generator()
            explanation["engine"] = "Deterministic Heuristic Extraction"
        else:
            if "engine" not in explanation:
                explanation["engine"] = self.provider_name

        # Augment with canonical execution metadata and disclaimers
        explanation["model_id"] = model_id
        explanation["model_prediction"] = predicted_label
        explanation["probabilities"] = probabilities
        explanation["disclaimer"] = (
            "This explanation is generated by an automated AI legal intelligence assistant based on "
            "the uploaded judgment text and does not constitute formal legal opinion or substitute for "
            "advice from a qualified legal advocate."
        )

        return explanation

    @staticmethod
    def _validate_schema(data: Any) -> bool:
        """Validates that LLM returned a compliant structured response."""
        if not isinstance(data, dict):
            return False
        required_keys = [
            "case_summary",
            "legal_issues",
            "court_reasoning",
            "legal_provisions",
            "supporting_paragraphs",
            "actual_court_outcome",
            "prediction_agreement",
            "agreement_explanation",
            "overall_explanation",
        ]
        return all(key in data for key in required_keys)

    def _generate_heuristic_fallback(
        self,
        text: str,
        model_id: str,
        predicted_label: str,
        probabilities: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministic, rule-based legal explanation generator grounded in the actual
        judgment text when external LLM API key is unconfigured or unavailable.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        last_chunk = " ".join(lines[-25:]).lower() if len(lines) >= 25 else text.lower()
        first_chunk = " ".join(lines[:20])

        # 1. Determine Operative Disposition
        actual_outcome = "Unclear"
        actual_class = "Unclear"
        matched_excerpt = ""

        # Check for Dismissal / Rejection
        if re.search(r"\b(appeals?\s+(are\s+|is\s+)?dismissed|petition\s+(is\s+)?dismissed|find\s+no\s+merit|devoid\s+of\s+merit)\b", last_chunk):
            actual_outcome = "Rejected (Appeals Dismissed)"
            actual_class = "Rejected"
            matched_excerpt = "The appeal was dismissed by the appellate court upon finding no merit in the contentions."
        # Check for Allowed / Reversal (Accepted)
        elif re.search(r"\b(appeals?\s+(are\s+|is\s+)?allowed|petition\s+(is\s+)?allowed|impugned\s+judgment\s+is\s+set\s+aside|conviction\s+is\s+set\s+aside)\b", last_chunk):
            actual_outcome = "Accepted (Appeal Allowed)"
            actual_class = "Accepted"
            matched_excerpt = "The appeal was allowed and the impugned judgment or order was set aside."
        # Check for Remand / Modification / Disposed of (Other)
        elif re.search(r"\b(remanded|sent\s+back|partly\s+allowed|disposed\s+of|interim\s+order)\b", last_chunk):
            actual_outcome = "Other (Remanded / Partly Allowed)"
            actual_class = "Other"
            matched_excerpt = "The court disposed of the matter with specific directions or remanded for fresh consideration."

        # 2. Extract Cited Statutory Provisions
        provisions = []
        found_provisions = set()

        patterns = [
            (r"\b(order\s+[ivxlcdm]+\s+rule\s+\d+(\s+of\s+cpc)?)\b", "Code of Civil Procedure, 1908"),
            (r"\b(section\s+302(\s+ipc|\s+of\s+the\s+indian\s+penal\s+code)?)\b", "Indian Penal Code (Murder)"),
            (r"\b(section\s+138(\s+of\s+ni\s+act|\s+negotiable\s+instruments)?)\b", "Negotiable Instruments Act (Dishonour of Cheque)"),
            (r"\b(section\s+34(\s+of\s+arbitration|\s+and\s+conciliation)?)\b", "Arbitration and Conciliation Act, 1996"),
            (r"\b(article\s+(14|19|21|32|136|226|265))\b", "Constitution of India"),
        ]

        for pat, act_name in patterns:
            matches = re.finditer(pat, text, re.IGNORECASE)
            for m in matches:
                prov_str = m.group(0).strip()
                norm_key = prov_str.lower()
                if norm_key not in found_provisions:
                    found_provisions.add(norm_key)
                    provisions.append({
                        "provision": f"{prov_str.title()} - {act_name}",
                        "relevance": f"Applied by the court to evaluate the legal validity of the dispute and claims under {act_name}."
                    })
                if len(provisions) >= 4:
                    break
            if len(provisions) >= 4:
                break

        if not provisions:
            provisions.append({
                "provision": "General Appellate & Procedural Jurisdiction",
                "relevance": "Standard statutory principles governing appellate appreciation of findings and evidence."
            })

        # 3. Calculate Prediction Agreement
        if actual_class == "Unclear":
            agreement = "Unclear"
            agreement_expl = (
                f"The operative judicial disposition could not be conclusively determined from the text excerpts. "
                f"The model predicted '{predicted_label}', which cannot be verified with certainty without inspecting the complete decree."
            )
        elif actual_class == predicted_label:
            agreement = "Match"
            agreement_expl = (
                f"The model's predicted disposition of '{predicted_label}' matches the court's actual operative ruling "
                f"('{actual_outcome}'). The model successfully recognized rhetorical patterns and indicators corresponding to this outcome."
            )
        else:
            agreement = "Mismatch"
            agreement_expl = (
                f"The model predicted '{predicted_label}', whereas the court's operative ruling is '{actual_outcome}'. "
                f"This divergence highlights rhetorical nuance or boundary conditions where trial court reasoning contrasts with final appellate decree."
            )

        # 4. Construct Summary and Reasoning
        summary = (
            f"The proceedings arise from appellate scrutiny of the impugned decree. "
            f"The appellant petitioned the court challenging the adverse determination rendered below. "
            f"The controversy centers on whether the legal thresholds and statutory prerequisites were properly adhered to."
        )

        reasoning = (
            f"The court scrutinized the evidentiary record, procedural compliance, and statutory mandates. "
            f"Upon weighing the rival submissions and relevant legal precedents, the bench determined that the findings below "
            f"{'merited affirmation' if actual_class == 'Rejected' else 'warranted appellate intervention' if actual_class == 'Accepted' else 'required modification or further procedural action'}. "
            f"The operative conclusion was accordingly pronounced."
        )

        if actual_class == "Rejected":
            relief_clause = "not support the relief claimed by the appellant"
        elif actual_class == "Accepted":
            relief_clause = "support the appellant's plea for relief"
        else:
            relief_clause = "require procedural clarification"

        overall = (
            f"The court rendered its decision of {actual_outcome} based on strict statutory interpretation and evaluation of the record. "
            f"The legal requirements were found to {relief_clause}."
        )

        # Supporting Paragraphs
        supporting = []
        if matched_excerpt:
            supporting.append({
                "paragraph_ref": "Operative Judicial Ruling",
                "text": matched_excerpt
            })

        for line in lines[-10:]:
            if any(w in line.lower() for w in ["dismissed", "allowed", "remanded", "costs", "directed", "ordered", "appeal"]):
                if len(line) > 30 and len(supporting) < 3:
                    supporting.append({
                        "paragraph_ref": "Appellate Conclusion Excerpt",
                        "text": line[:300]
                    })

        if not supporting:
            supporting.append({
                "paragraph_ref": "Concluding Passage",
                "text": lines[-1] if lines else "Operative judgment excerpt."
            })

        return {
            "case_summary": summary,
            "legal_issues": [
                "Whether the impugned decision conforms to statutory requirements and applicable legal precedents.",
                "Whether the appellant established sufficient legal grounds to warrant appellate interference."
            ],
            "court_reasoning": reasoning,
            "legal_provisions": provisions,
            "supporting_paragraphs": supporting,
            "actual_court_outcome": actual_outcome,
            "model_prediction": predicted_label,
            "prediction_agreement": agreement,
            "agreement_explanation": agreement_expl,
            "overall_explanation": overall,
            "limitations_and_uncertainty": (
                "Heuristic extraction based on operative judgment syntax. Multi-class outcome classification "
                "is probabilistic and reflects rhetorical document structure rather than exhaustive legal advocacy."
            ),
            "engine": "Deterministic Heuristic Extraction"
        }
