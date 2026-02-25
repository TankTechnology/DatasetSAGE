from __future__ import annotations

import json
from pathlib import Path

from .models import QueryPaper, UsageRecord


def load_usage_records(path: str | Path) -> list[UsageRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for item in payload:
        records.append(
            UsageRecord(
                record_id=item["record_id"],
                source_paper_id=item["source_paper_id"],
                source_title=item["source_title"],
                dataset_entity=item["dataset_entity"],
                dataset_name=item.get("dataset_name", item["dataset_entity"]),
                task=item.get("task", ""),
                modality=item.get("modality", ""),
                evidence=item.get("evidence", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return records


def load_query_paper(path: str | Path) -> QueryPaper:
    item = json.loads(Path(path).read_text(encoding="utf-8"))
    return QueryPaper(
        paper_id=item["paper_id"],
        title=item["title"],
        abstract=item["abstract"],
        task_goal=item.get("task_goal", ""),
    )
