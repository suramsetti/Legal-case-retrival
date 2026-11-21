"""Legal knowledge feature extraction package."""

from src.legal_features.statute_extractor import StatuteExtractor, StatuteReference
from src.legal_features.citation_extractor import CitationExtractor, LegalCitation
from src.legal_features.entity_extractor import LegalEntityExtractor, LegalEntity
from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, SentenceRole, RHETORICAL_ROLES
from src.legal_features.legal_feature_encoder import (
    LegalFeatureEncoder,
    DocumentLegalFeatures,
)

__all__ = [
    "StatuteExtractor",
    "StatuteReference",
    "CitationExtractor",
    "LegalCitation",
    "LegalEntityExtractor",
    "LegalEntity",
    "RhetoricalAnalyzer",
    "SentenceRole",
    "RHETORICAL_ROLES",
    "LegalFeatureEncoder",
    "DocumentLegalFeatures",
]

