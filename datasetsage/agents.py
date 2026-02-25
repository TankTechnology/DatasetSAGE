from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .models import CandidateRecord, DatasetCard, QueryPaper, RecommendationResult
from .prompts import (
    REASON_CORE_INSTRUCTION,
    RERANK_CORE_INSTRUCTION,
    REVIEW_CORE_INSTRUCTION,
)


@dataclass
class ReasonConfig:
    threshold: float = 0.2


class ReasonAgent:
    """
    Reason Agent:
    validate record-level relevance with calibrated score and brief rationale.
    """

    def __init__(self, llm_client: LLMClient, cfg: ReasonConfig | None = None):
        self.llm = llm_client
        self.cfg = cfg or ReasonConfig()

    def validate(self, paper: QueryPaper, candidates: list[CandidateRecord]) -> list[CandidateRecord]:
        validated: list[CandidateRecord] = []
        for cand in candidates:
            score, reason = self._llm_reason_score(paper, cand)
            if score < self.cfg.threshold:
                continue
            cand.reason_score = score
            cand.reason_text = reason
            validated.append(cand)
        validated.sort(key=lambda x: x.reason_score, reverse=True)
        return validated

    def _llm_reason_score(self, paper: QueryPaper, cand: CandidateRecord) -> tuple[float, str]:
        system_prompt = (
            "You are the Reason Agent in DatasetSAGE. "
            "Judge whether a usage record is valid supporting evidence for the query paper.\n"
            f"Core instruction: {REASON_CORE_INSTRUCTION}"
        )
        user_prompt = (
            f"Query title: {paper.title}\n"
            f"Query abstract: {paper.abstract}\n"
            f"Task goal: {paper.task_goal}\n\n"
            f"Candidate evidence card:\n{cand.usage_record.evidence_card()}\n\n"
            f"Retrieval channels: {cand.channels}\n"
            f"Retrieval score: {cand.retrieval_score:.4f}\n\n"
            "Return JSON: {\"confidence\": float(0-1), \"reason\": string}."
        )
        payload = self.llm.chat_json(system_prompt, user_prompt)
        try:
            score = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        reason = str(payload.get("reason", "")).strip() or "No reason provided."
        return score, reason


class ReviewAgent:
    """
    Review Agent:
    inspect validated evidence and propose a refinement query for next round.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def refine_query(self, paper: QueryPaper, validated: list[CandidateRecord]) -> tuple[str, str]:
        evidence_summary = []
        for cand in validated[:12]:
            evidence_summary.append(
                {
                    "record_id": cand.record_id,
                    "dataset": cand.dataset_name,
                    "task": cand.usage_record.task,
                    "modality": cand.usage_record.modality,
                    "reason_score": cand.reason_score,
                }
            )
        system_prompt = (
            "You are the Review Agent in DatasetSAGE. "
            "Detect evidence gaps and propose the next-round retrieval query.\n"
            f"Core instruction: {REVIEW_CORE_INSTRUCTION}"
        )
        user_prompt = (
            f"Query title: {paper.title}\n"
            f"Query abstract: {paper.abstract}\n"
            f"Task goal: {paper.task_goal}\n\n"
            f"Validated evidence summary: {evidence_summary}\n\n"
            "Return JSON: {\"new_query\": string, \"feedback\": string}."
        )
        payload = self.llm.chat_json(system_prompt, user_prompt)
        new_query = str(payload.get("new_query", "")).strip()
        feedback = str(payload.get("feedback", "")).strip() or "No feedback provided."
        if not new_query:
            new_query = f"{paper.title}. Retrieve datasets with stronger usage evidence."
        return new_query, feedback


class RerankAgent:
    """
    Re-ranking Agent:
    aggregate validated records by dataset and output final ranking.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def rerank(
        self,
        paper: QueryPaper,
        validated: list[CandidateRecord],
        top_k: int = 10,
    ) -> list[RecommendationResult]:
        grouped: dict[str, DatasetCard] = {}
        for cand in validated:
            entity = cand.dataset_entity
            if entity not in grouped:
                grouped[entity] = DatasetCard(
                    dataset_entity=entity,
                    dataset_name=cand.dataset_name,
                    support_records=[],
                    score=0.0,
                )
            card = grouped[entity]
            # Deduplicate by record_id across rounds, keep the stronger validation score.
            existing = {r.record_id: r for r in card.support_records}
            prev = existing.get(cand.record_id)
            if prev is None or cand.reason_score > prev.reason_score:
                existing[cand.record_id] = cand
                card.support_records = list(existing.values())

        cards_payload = []
        for card in grouped.values():
            cards_payload.append(
                {
                    "dataset_entity": card.dataset_entity,
                    "dataset_name": card.dataset_name,
                    "records": [
                        {
                            "record_id": r.record_id,
                            "task": r.usage_record.task,
                            "modality": r.usage_record.modality,
                            "reason_score": r.reason_score,
                            "reason_text": r.reason_text,
                        }
                        for r in card.support_records
                    ],
                }
            )

        if cards_payload:
            system_prompt = (
                "You are the Re-ranking Agent in DatasetSAGE. "
                "Aggregate dataset cards and output final ranking with grounded rationales.\n"
                f"Core instruction: {RERANK_CORE_INSTRUCTION}"
            )
            user_prompt = (
                f"Query title: {paper.title}\n"
                f"Query abstract: {paper.abstract}\n"
                f"Task goal: {paper.task_goal}\n\n"
                f"Dataset cards: {cards_payload}\n\n"
                "Return JSON: {\"ranked\": [{\"dataset_entity\": str, \"score\": float(0-1), \"rationale\": str}]}"
            )
            payload = self.llm.chat_json(system_prompt, user_prompt)
            rank_map = {
                str(x.get("dataset_entity", "")).strip(): x
                for x in payload.get("ranked", [])
                if isinstance(x, dict)
            }
        else:
            rank_map = {}

        for card in grouped.values():
            llm_row = rank_map.get(card.dataset_entity, {})
            if llm_row:
                try:
                    score = float(llm_row.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                card.score = max(0.0, min(1.0, score))
                card.global_reason = str(llm_row.get("rationale", "")).strip() or (
                    f"{card.dataset_name} supported by validated usage records."
                )
            else:
                max_reason = max((c.reason_score for c in card.support_records), default=0.0)
                card.score = max_reason
                card.global_reason = (
                    f"{card.dataset_name} supported by {len(card.support_records)} validated records."
                )

        ranked = sorted(grouped.values(), key=lambda c: c.score, reverse=True)
        return [
            RecommendationResult(
                dataset_entity=card.dataset_entity,
                dataset_name=card.dataset_name,
                score=card.score,
                rationale=card.global_reason,
                supporting_records=[r.record_id for r in card.support_records],
            )
            for card in ranked[:top_k]
        ]
