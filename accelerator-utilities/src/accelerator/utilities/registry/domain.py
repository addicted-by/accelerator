"""Domain classification for registered objects."""

from enum import Enum


class Domain(Enum):
    """Machine learning domain classification for registered objects."""

    CROSS = "cross"  # Cross-domain/general purpose (default)
    CV = "cv"  # Computer Vision
    NLP = "nlp"  # Natural Language Processing
    AUDIO = "audio"  # Audio processing
    TABLE = "table"  # Tabular data
    TIMESERIES = "timeseries"  # Time series data
