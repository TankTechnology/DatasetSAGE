from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPaper:
    """Input paper represented only by title and abstract."""

    paper_id: str
    title: str
    abstract: str
    task_goal: str = ""

    def to_text(self) -> str:
        parts = [self.title.strip(), self.abstract.strip()]
        if self.task_goal.strip():
            parts.append(self.task_goal.strip())
        return "\n".join([p for p in parts if p])


@dataclass
class UsageRecord:
    """
    Evidence unit in DatasetSAGE.

    One paper-dataset usage edge converted into an evidence card.
    """

    record_id: str
    source_paper_id: str
    source_title: str
    dataset_entity: str
    dataset_name: str
    task: str
    modality: str
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def evidence_card(self) -> str:
        return (
            f"Source paper: {self.source_title}\n"
            f"Dataset: {self.dataset_name}\n"
            f"Task: {self.task}\n"
            f"Modality: {self.modality}\n"
            f"Evidence: {self.evidence}"
        )


@dataclass
class CandidateRecord:
    """Candidate usage record returned by retrieval channels."""

    usage_record: UsageRecord
    channels: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    reason_score: float = 0.0
    reason_text: str = ""

    @property
    def dataset_entity(self) -> str:
        return self.usage_record.dataset_entity

    @property
    def dataset_name(self) -> str:
        return self.usage_record.dataset_name

    @property
    def record_id(self) -> str:
        return self.usage_record.record_id


@dataclass
class DatasetCard:
    """Aggregated dataset-level evidence from validated records."""

    dataset_entity: str
    dataset_name: str
    support_records: list[CandidateRecord] = field(default_factory=list)
    score: float = 0.0
    global_reason: str = ""


@dataclass
class RecommendationResult:
    dataset_entity: str
    dataset_name: str
    score: float
    rationale: str
    supporting_records: list[str]
