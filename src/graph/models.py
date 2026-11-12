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
    EXTERNAL = "external"  # producer outside the data platform (SAP CDC, WMS, SFTP partner…)
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


class ColumnMapping(BaseModel):
    source_column: str
    target_column: str
    transformation: Optional[str] = None
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    evidence: Optional[str] = None
    sme_approved_by: Optional[str] = None


class LineageEdge(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    transformation: Optional[str] = None
    system_boundary: Optional[str] = None  # e.g. "ADLS→Databricks"
    # Orchestrator that runs the step producing this edge, e.g.
    # "adf:pl_lakehouse_orchestration_nightly/nb_silver_pos_sales". ADF is a
    # scheduler, not a data store, so it is recorded here rather than as nodes.
    orchestrated_by: Optional[str] = None
    # Artifacts (filenames) whose parsed code produces this edge.
    produced_by: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    evidence: Optional[str] = None         # artifact name / SME name
    sme_approved_by: Optional[str] = None
    column_mappings: list[ColumnMapping] = Field(default_factory=list)
    review_status: str = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_reason: Optional[str] = None
    audit_trail: list[dict] = Field(default_factory=list)


class LineageFragment(BaseModel):
    """Lineage extracted from a single artifact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_name: str
    artifact_type: str
    nodes: list[LineageNode] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)
    raw_metadata: dict = Field(default_factory=dict)


class ProducerClaim(BaseModel):
    """One artifact's claim to produce a target, with the evidence for it."""
    artifact: str
    notebook_path: Optional[str] = None
    orchestrated_by: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    matches_catalog_last_write: bool = False


class ProducerConflict(BaseModel):
    """
    Two or more artifacts write the same target (e.g. margin_v1 and margin_v2
    both write gold.fact_store_week_sales). The agent records the evidence for
    each claim and asks an SME which is live rather than guessing.
    """
    target_id: str
    claims: list[ProducerClaim]
    catalog_last_write: Optional[dict] = None
    resolved_artifact: Optional[str] = None  # set once an SME names the live producer
    resolved_by: Optional[str] = None


class LineageGraph(BaseModel):
    """Stitched cross-system lineage graph."""
    nodes: dict[str, LineageNode] = Field(default_factory=dict)
    edges: list[LineageEdge] = Field(default_factory=list)
    producer_conflicts: list[ProducerConflict] = Field(default_factory=list)

    def open_conflicts(self) -> list[ProducerConflict]:
        return [c for c in self.producer_conflicts if not c.resolved_artifact]

    def contested_edge_ids(self) -> set[str]:
        """Edges into a target whose producer is still disputed."""
        targets = {c.target_id for c in self.open_conflicts()}
        return {e.id for e in self.edges if e.target_id in targets}

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
