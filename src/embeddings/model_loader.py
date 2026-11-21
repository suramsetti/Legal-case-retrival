"""
Model Loader for Transformer Legal Encoders.

Supports loading BERT, Legal-BERT, and InLegalBERT with automatic
device detection (CUDA/CPU), half-precision where available, and model verification.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "bert-base-uncased": {
        "hf_id": "bert-base-uncased",
        "description": "General English BERT Base (110M parameters)",
        "hidden_size": 768,
        "max_length": 512,
        "license": "Apache-2.0",
        "domain": "General English",
    },
    "legal-bert-base-uncased": {
        "hf_id": "nlpaueb/legal-bert-base-uncased",
        "description": "Legal-BERT Base trained on EU, UK, US legislation and court cases",
        "hidden_size": 768,
        "max_length": 512,
        "license": "CC-BY-SA-4.0",
        "domain": "English Legal (International)",
    },
    "inlegalbert": {
        "hf_id": "law-ai/InLegalBERT",
        "description": "InLegalBERT Base pre-trained on Supreme Court and High Courts of India",
        "hidden_size": 768,
        "max_length": 512,
        "license": "Apache-2.0",
        "domain": "Indian Legal Judgments",
    },
}


class ModelLoader:
    """Manages loading of transformer architectures and tokenizers for dense retrieval."""

    @staticmethod
    def resolve_model_id(model_name: str) -> str:
        """Resolves friendly alias or official Hugging Face model identifier."""
        clean_key = model_name.lower().strip()
        if clean_key in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[clean_key]["hf_id"]
        for key, info in SUPPORTED_MODELS.items():
            if model_name == info["hf_id"]:
                return info["hf_id"]
        return model_name

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name_or_id: str,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ) -> Tuple[AutoModel, AutoTokenizer, str]:
        """
        Loads Hugging Face model and tokenizer in evaluation mode.
        
        Args:
            model_name_or_id: Hugging Face model ID or alias.
            device: 'cuda', 'cpu', or None (auto-detect).
            use_fp16: Whether to convert model to fp16 (recommended for CUDA).
            
        Returns:
            (model, tokenizer, active_device)
        """
        hf_id = cls.resolve_model_id(model_name_or_id)

        # Device determination
        if device is None:
            active_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            active_device = device

        # Check if local model directory exists in workspace
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        local_dir = project_root / "models" / "bert" / model_name_or_id.lower().replace("/", "_").replace("-", "_")
        local_dir_alt = project_root / "models" / "bert" / model_name_or_id.lower()
        
        target_path = hf_id
        if local_dir.exists() and (local_dir / "config.json").exists():
            target_path = str(local_dir)
        elif local_dir_alt.exists() and (local_dir_alt / "config.json").exists():
            target_path = str(local_dir_alt)

        try:
            tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
            model = AutoModel.from_pretrained(target_path, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(target_path)
            model = AutoModel.from_pretrained(target_path)
        model.eval()

        if use_fp16 and active_device == "cuda" and torch.cuda.is_bf16_supported():
            model = model.to(active_device, dtype=torch.bfloat16)
        elif use_fp16 and active_device == "cuda":
            model = model.to(active_device, dtype=torch.float16)
        else:
            model = model.to(active_device)

        return model, tokenizer, active_device

    @classmethod
    def get_model_metadata(cls, model_name_or_id: str) -> Dict[str, Any]:
        """Returns structured metadata for a supported model."""
        hf_id = cls.resolve_model_id(model_name_or_id)
        for key, info in SUPPORTED_MODELS.items():
            if hf_id == info["hf_id"]:
                return info.copy()

        config = AutoConfig.from_pretrained(hf_id)
        return {
            "hf_id": hf_id,
            "description": f"Custom model {hf_id}",
            "hidden_size": config.hidden_size,
            "max_length": getattr(config, "max_position_embeddings", 512),
            "license": getattr(config, "license", "unknown"),
            "domain": "custom",
        }

