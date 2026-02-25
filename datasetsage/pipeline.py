from __future__ import annotations

from dataclasses import dataclass, field

from .agents import ReasonAgent, ReviewAgent, RerankAgent
from .llm import LLMClient, LLMConfig
from .models import QueryPaper, RecommendationResult, UsageRecord
from .retrieval import MultiChannelRetriever, RetrievalConfig


@dataclass
class PipelineConfig:
    rounds: int = 2
    top_k: int = 30
    reason_threshold: float = 0.2
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


class DatasetSAGEPipeline:
    """
    Compact DatasetSAGE pipeline.

    Open-loop: Retrieval -> Reason -> Re-ranking
    Closed-loop: Retrieval -> Reason -> Review -> Retrieval ... -> Re-ranking
    """

    def __init__(self, usage_records: list[UsageRecord], cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.llm_client = LLMClient(self.cfg.llm)
        self.retriever = MultiChannelRetriever(
            usage_records,
            llm_client=self.llm_client,
            cfg=self.cfg.retrieval,
        )
        self.reason_agent = ReasonAgent(self.llm_client)
        self.reason_agent.cfg.threshold = self.cfg.reason_threshold
        self.review_agent = ReviewAgent(self.llm_client)
        self.rerank_agent = RerankAgent(self.llm_client)

    def run(self, paper: QueryPaper) -> dict[str, object]:
        rounds = max(1, self.cfg.rounds)
        all_validated = []
        round_traces = []

        round_query = paper
        for round_idx in range(rounds):
            # Paper-aligned closed-loop: in round 0 use all channels;
            # in later rounds only dense retrieval is re-invoked.
            if round_idx == 0:
                retrieved = self.retriever.retrieve(round_query, enabled_channels=["dense", "graph", "open_world"])
            else:
                retrieved = self.retriever.retrieve(round_query, enabled_channels=["dense"])
            validated = self.reason_agent.validate(paper, retrieved)
            all_validated.extend(validated)

            trace = {
                "round": round_idx,
                "query_text": round_query.to_text(),
                "channels": ["dense", "graph", "open_world"] if round_idx == 0 else ["dense"],
                "retrieved_count": len(retrieved),
                "validated_count": len(validated),
                "top_validated_record_ids": [c.record_id for c in validated[:8]],
            }

            if round_idx < rounds - 1:
                refined_query, feedback = self.review_agent.refine_query(paper, validated)
                round_query = QueryPaper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    abstract=refined_query,
                    task_goal=paper.task_goal,
                )
                trace["review_feedback"] = feedback
                trace["refined_query"] = refined_query
            round_traces.append(trace)

        final_results = self.rerank_agent.rerank(
            paper,
            all_validated,
            top_k=self.cfg.top_k,
        )
        return {
            "paper_id": paper.paper_id,
            "mode": "closed_loop" if rounds > 1 else "open_loop",
            "rounds": rounds,
            "round_traces": round_traces,
            "recommendations": [self._to_dict(r) for r in final_results],
        }

    @staticmethod
    def _to_dict(result: RecommendationResult) -> dict[str, object]:
        return {
            "dataset_entity": result.dataset_entity,
            "dataset_name": result.dataset_name,
            "score": round(result.score, 4),
            "rationale": result.rationale,
            "supporting_records": result.supporting_records,
        }
