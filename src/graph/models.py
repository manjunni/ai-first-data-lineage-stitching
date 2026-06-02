from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class SystemType(str, Enum):
    ADLS = "adls"
    DATABRICKS = "databricks"
    DLT = "dlt"
    ADF = "adf"
    SQL = "sql"
    BI = "bi"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # parsed directly from artifact
    MEDIUM = "medium"   # inferred by LLM
    LOW = "low"         # unresolved gap
    CONFIRMED = "confirmed"  # SME approved


class LineageNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    system: SystemType
    full_path: str
    description: Optional[str] = None
    owner: Optional[str] = None


class LineageEdge(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    transformation: Optional[str] = None
    system_boundary: Optional[str] = None  # e.g. "ADLS→Databricks"
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    evidence: Optional[str] = None         # artifact name / SME name
    sme_approved_by: Optional[str] = None
    audit_trail: list[dict] = Field(default_factory=list)


class LineageFragment(BaseModel):
    """Lineage extracted from a single artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_name: str
    artifact_type: str
    nodes: list[LineageNode] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)
    raw_metadata: dict = Field(default_factory=dict)


class LineageGraph(BaseModel):
    """Stitched cross-system lineage graph."""
    nodes: dict[str, LineageNode] = Field(default_factory=dict)
    edges: list[LineageEdge] = Field(default_factory=list)

    def add_fragment(self, fragment: LineageFragment) -> None:
        for node in fragment.nodes:
            self.nodes[node.id] = node
        self.edges.extend(fragment.edges)


class Gap(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    gap_type: str           # "missing_source", "missing_target", "unknown_transform"
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    description: str
    suggested_question: Optional[str] = None
    sme_email: Optional[str] = None
    resolved: bool = False
