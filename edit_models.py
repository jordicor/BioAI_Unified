"""
Models for the intelligent editing system
==========================================

This module defines data models for the incremental editing system that:
1. Identifies specific text ranges with problems
2. Supports different edit types (replace, remove, add, rephrase)
3. Enables efficient targeted corrections without full regeneration
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, AliasChoices
from enum import Enum


class MarkerMode(str, Enum):
    """Marker identification mode for smart edit paragraph location"""
    PHRASE = "phrase"        # Use n-word phrase markers (default)
    WORD_INDEX = "word_index"  # Use word map indices (fallback for repetitive content)


class EditType(str, Enum):
    """Supported edit operation types"""
    REPLACE = "replace"       # Replace fragment with new content
    REMOVE = "remove"         # Remove fragment entirely
    ADD_AFTER = "add_after"   # Add content after fragment
    ADD_BEFORE = "add_before" # Add content before fragment
    REPHRASE = "rephrase"     # Rephrase maintaining meaning


class SeverityLevel(str, Enum):
    """Problem severity levels"""
    CRITICAL = "critical"  # Serious content errors requiring immediate fix
    MAJOR = "major"       # Important problems affecting quality
    MINOR = "minor"       # Minor improvements or polish


class EditBaseModel(BaseModel):
    """Base model enabling population by field name while preserving Pydantic's namespace safeguards."""

    model_config = {"populate_by_name": True}


class MarkerConfig(EditBaseModel):
    """
    Configuration for paragraph markers in smart edit.

    This is computed via pre-scan before QA evaluation and determines
    how paragraphs will be identified for editing.
    """

    mode: MarkerMode = Field(
        default=MarkerMode.PHRASE,
        description="How paragraphs are identified: 'phrase' uses text markers, 'word_index' uses indices"
    )
    phrase_length: Optional[int] = Field(
        default=None,
        ge=4,
        le=12,
        description="Number of words for phrase markers (4-12), None if word_index mode"
    )
    word_map: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word map tokens for word_index mode (from _build_word_map)"
    )
    word_map_formatted: Optional[str] = Field(
        default=None,
        description="Formatted word map string for QA prompt inclusion"
    )


class TextEditRange(EditBaseModel):
    """
    Identifies a specific text range to edit.

    Supports two identification modes:
    - phrase mode: Uses paragraph_start/paragraph_end text markers
    - word_index mode: Uses start_word_index/end_word_index from word map
    """

    # Marker mode selection
    marker_mode: str = Field(
        default="phrase",
        description="Identification mode: 'phrase' (text markers) or 'word_index' (word map indices)"
    )

    # Paragraph identification - phrase mode (now optional for word_index mode)
    paragraph_start: str = Field(
        default="",
        description="First N words of the paragraph (phrase mode)"
    )
    paragraph_end: str = Field(
        default="",
        description="Last N words of the paragraph (phrase mode)"
    )

    # Paragraph identification - word_index mode
    start_word_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start word index from word map (word_index mode)"
    )
    end_word_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="End word index from word map, inclusive (word_index mode)"
    )

    # Specific fragment to edit (optional, used as fallback)
    exact_fragment: str = Field(
        default="",
        description="Exact text fragment to modify (5-20 words)"
    )

    # Edit details
    edit_type: EditType = Field(
        default=EditType.REPLACE,
        description="Type of edit operation"
    )
    new_content: Optional[str] = Field(
        default=None,
        description="New content for replace/add operations"
    )
    edit_instruction: str = Field(
        default="",
        description="Specific instruction for what to change"
    )

    # Metadata
    issue_severity: SeverityLevel = Field(
        default=SeverityLevel.MINOR,
        description="Severity level of the issue"
    )
    issue_description: str = Field(
        default="",
        description="Clear description of the detected problem"
    )

    # Validation
    is_unique: bool = Field(
        default=True,
        description="Whether the fragment is unique in the text"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in problem identification (0-1)"
    )


class EditContext(EditBaseModel):
    """Adaptive context for applying edits efficiently"""

    full_text: Optional[str] = None
    context_window: Optional[str] = None
    window_size: int
    total_length: int
    strategy: str  # "full_text" or "windowed"

    # Additional metadata
    style_sample: Optional[str] = Field(
        None,
        description="Sample of writing style (first 200 words)"
    )
    tone_indicators: Optional[List[str]] = Field(
        None,
        description="Detected tone indicators"
    )


class EditDecision(EditBaseModel):
    """Strategic decision about correction approach"""

    strategy: str = Field(
        ...,
        description="'incremental_edit' or 'full_regeneration'"
    )
    reason: str = Field(
        ...,
        description="Justification for the chosen strategy"
    )

    # For incremental editing
    edit_ranges: Optional[List[TextEditRange]] = None
    total_issues: int = 0
    editable_issues: int = 0
    estimated_tokens_saved: int = 0

    # Applied thresholds
    applied_thresholds: Optional[Dict[str, Any]] = None


class QAEvaluationWithRanges(EditBaseModel):
    """Extended QA evaluation with identified text ranges"""

    score: float
    feedback: str
    has_deal_breakers: bool = False
    deal_breaker_reason: Optional[str] = None

    # NEW: Identified issue ranges
    identified_issues: Optional[List[TextEditRange]] = None

    # Metadata
    used_model: str = Field(
        ...,
        description="Model that produced this QA evaluation",
        validation_alias=AliasChoices("model_used", "used_model"),
        serialization_alias="model_used",
    )
    layer_name: str
    evaluation_time: float
