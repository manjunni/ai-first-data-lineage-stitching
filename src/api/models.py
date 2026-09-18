from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

class StartRunRequest(BaseModel):
    catalog: str = Field("dry_run", description="Target catalog: alation | collibra | dry_run")


class RunSummary(BaseModel):
    run_id: str
    status: str
    iteration: int
    node_count: int
    edge_count: int
    gaps_open: int
    gaps_total: int
    sme_questions_pending: int
    catalog_published: bool
    target_catalog: str


class RunListItem(BaseModel):
    run_id: str
    status: str
    node_count: int
    edge_count: int
    gaps_open: int


# ---------------------------------------------------------------------------
# Lineage graph
# ---------------------------------------------------------------------------

class NodeResponse(BaseModel):
    id: str
    name: str
    system: str
    full_path: str
    description: Optional[str]
    owner: Optional[str]


class ColumnMappingResponse(BaseModel):
    source_column: str
    target_column: str
    transformation: Optional[str]
    confidence: float
    confidence_level: str
    evidence: Optional[str]
    sme_approved_by: Optional[str]


class EdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    transformation: Optional[str]
    system_boundary: Optional[str]
    confidence: float
    confidence_level: str
    evidence: Optional[str]
    sme_approved_by: Optional[str]
    column_mappings: list[ColumnMappingResponse] = Field(default_factory=list)
    review_status: str = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_reason: Optional[str] = None


class LineageGraphResponse(BaseModel):
    run_id: str
    node_count: int
    edge_count: int
    nodes: list[NodeResponse]
    edges: list[EdgeResponse]


# ---------------------------------------------------------------------------
# Gaps and questions
# ---------------------------------------------------------------------------

class GapResponse(BaseModel):
    id: str
    gap_type: str
    description: str
    node_id: Optional[str]
    edge_id: Optional[str]
    sme_email: Optional[str]
    resolved: bool


class GapListResponse(BaseModel):
    run_id: str
    total: int
    open: int
    resolved: int
    gaps: list[GapResponse]


class QuestionResponse(BaseModel):
    gap_id: str
    gap_type: str
    question: str
    sme_email: Optional[str]
    answered: bool
    send_status: Optional[str]


class QuestionListResponse(BaseModel):
    run_id: str
    total: int
    pending: int
    questions: list[QuestionResponse]


# ---------------------------------------------------------------------------
# SME responses
# ---------------------------------------------------------------------------

class SmeResponsePayload(BaseModel):
    """Raw email reply dicts produced by parse_email_responses()."""
    responses: list[dict]


class SmeFileUploadPayload(BaseModel):
    """
    Multi-format file upload for SME gap-fixing responses.
    
    Supports .txt, .docx, .xlsx, .pdf, .json files.
    Can be combined with direct text input.
    """
    text: Optional[str] = Field(
        None,
        description="Direct text input (alternative to files or in addition to files)"
    )
    sme_email: Optional[str] = Field(
        None,
        description="SME email address (required if not from email webhook)"
    )
    run_id: Optional[str] = Field(
        None,
        description="Run ID (optional, can be extracted from text)"
    )


class EmailWebhookPayload(BaseModel):
    """Inbound email payload from Azure Communication Services / SendGrid webhook."""
    sender: Optional[str] = Field(None, alias="from")
    subject: str = ""
    body: Optional[str] = None
    text: Optional[str] = None

    model_config = {"populate_by_name": True}

    def body_text(self) -> str:
        return self.body or self.text or ""


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class ConfidenceSummaryResponse(BaseModel):
    run_id: str
    total_edges: int
    avg_confidence: Optional[float]
    min_confidence: Optional[float]
    max_confidence: Optional[float]
    by_level: dict[str, int]
    sme_confirmed: int
    cross_system_edges: int
    unresolved_adf_edges: int
    pending_review: int = 0
    rejected_edges: int = 0


class LineageReviewRequest(BaseModel):
    decision: str = Field(description="approve or reject")
    reviewer: str = Field(min_length=1)
    reason: Optional[str] = None


class LineageReviewResponse(BaseModel):
    run_id: str
    edge_id: str
    review_status: str
    reviewed_by: str
    reviewed_at: str
    review_reason: Optional[str]
    confidence: float
    confidence_level: str


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

class VisualsResponse(BaseModel):
    run_id: str
    mermaid: str
    graphviz: str
    summary_html: Optional[str]


# ---------------------------------------------------------------------------
# Catalog publish
# ---------------------------------------------------------------------------

class PublishRequest(BaseModel):
    catalog: str = Field("dry_run", description="Target catalog: alation | collibra | dry_run")


class PublishResponse(BaseModel):
    run_id: str
    catalog: str
    status: str
    success: bool
    nodes_published: int
    edges_published: int
    errors: list[str]
    catalog_urls: list[str]
    payload_summary: Optional[dict]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    catalog_config: dict[str, bool]
