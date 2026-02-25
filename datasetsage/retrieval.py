from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Literal

from .llm import LLMClient
from .models import CandidateRecord, QueryPaper, UsageRecord
from .prompts import OPEN_WORLD_CORE_INSTRUCTION

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@dataclass
class RetrievalConfig:
    # Paper-aligned defaults from Appendix hyperparameters.
    dense_top_k: int = 50
    graph_top_k: int = 30
    open_world_top_k: int = 20
    dense_weight: float = 1.0
    graph_weight: float = 0.9
    open_world_weight: float = 0.65
    # Graph-contrastive retriever settings (paper-inspired).
    gnn_layers: int = 2
    contrastive_temperature: float = 0.07
    contrastive_epochs: int = 100
    contrastive_lr: float = 0.001
    contrastive_batch_size: int = 256
    contrastive_negatives: int = 50


class MultiChannelRetriever:
    """
    Retrieval Agent (paper-aligned):
    - Dense semantic retrieval over usage-record evidence cards.
    - Graph-contrastive structural retrieval over paper-dataset bipartite graph.
    - Open-world LLM generation (mapped to canonical dataset inventory).
    """

    def __init__(
        self,
        usage_records: list[UsageRecord],
        llm_client: LLMClient,
        cfg: RetrievalConfig | None = None,
    ):
        self.records = usage_records
        self.llm = llm_client
        self.cfg = cfg or RetrievalConfig()
        self._dataset_to_records: dict[str, list[UsageRecord]] = defaultdict(list)
        self._paper_to_records: dict[str, list[UsageRecord]] = defaultdict(list)
        self._paper_profile_text: dict[str, str] = {}
        self._paper_dataset_graph: dict[str, set[str]] = defaultdict(set)
        self._dataset_degree: dict[str, int] = defaultdict(int)
        self._record_embeddings: dict[str, list[float]] = {}
        self._paper_embeddings: dict[str, list[float]] = {}
        self._dataset_embeddings_raw: dict[str, list[float]] = {}
        self._dataset_embeddings_graph: dict[str, list[float]] = {}
        self._dataset_alias: dict[str, str] = {}
        self._projection_matrix: list[list[float]] | None = None
        self._build_graph_index()
        self._build_embeddings()
        self._fit_graph_contrastive_projection()

    def _build_graph_index(self) -> None:
        for rec in self.records:
            self._dataset_to_records[rec.dataset_entity].append(rec)
            self._paper_to_records[rec.source_paper_id].append(rec)
            self._paper_dataset_graph[rec.source_paper_id].add(rec.dataset_entity)
            self._dataset_alias[rec.dataset_entity.lower()] = rec.dataset_entity
            self._dataset_alias[rec.dataset_name.lower()] = rec.dataset_entity
        for datasets in self._paper_dataset_graph.values():
            for dataset_entity in datasets:
                self._dataset_degree[dataset_entity] += 1

        for paper_id, records in self._paper_to_records.items():
            title = records[0].source_title
            task_blob = " ".join(sorted({r.task for r in records if r.task}))
            modality_blob = " ".join(sorted({r.modality for r in records if r.modality}))
            evidence_blob = " ".join([r.evidence for r in records[:4]])
            self._paper_profile_text[paper_id] = (
                f"Paper: {title}\nTasks: {task_blob}\nModalities: {modality_blob}\nEvidence: {evidence_blob}"
            )

    def _build_embeddings(self) -> None:
        record_texts = [r.evidence_card() for r in self.records]
        record_embeds = self.llm.embed_texts(record_texts)
        for rec, emb in zip(self.records, record_embeds):
            self._record_embeddings[rec.record_id] = emb

        paper_ids = list(self._paper_profile_text.keys())
        paper_texts = [self._paper_profile_text[pid] for pid in paper_ids]
        paper_embeds = self.llm.embed_texts(paper_texts)
        for pid, emb in zip(paper_ids, paper_embeds):
            self._paper_embeddings[pid] = emb

        dataset_entities = list(self._dataset_to_records.keys())
        dataset_texts = []
        for entity in dataset_entities:
            rows = self._dataset_to_records.get(entity, [])
            dataset_name = rows[0].dataset_name if rows else entity
            task_blob = " ".join(sorted({r.task for r in rows if r.task}))
            modality_blob = " ".join(sorted({r.modality for r in rows if r.modality}))
            evidence_blob = " ".join([r.evidence for r in rows[:4]])
            dataset_texts.append(
                f"Dataset: {dataset_name}\nTasks: {task_blob}\nModalities: {modality_blob}\nEvidence: {evidence_blob}"
            )
        dataset_embeds = self.llm.embed_texts(dataset_texts)
        for entity, emb in zip(dataset_entities, dataset_embeds):
            self._dataset_embeddings_raw[entity] = emb

        self._dataset_embeddings_graph = self._lightgcn_dataset_embeddings(self.cfg.gnn_layers)

    def _lightgcn_dataset_embeddings(self, layers: int) -> dict[str, list[float]]:
        """LightGCN-style propagation on bipartite graph to enrich dataset embeddings."""
        if not self._paper_embeddings or not self._dataset_embeddings_raw:
            return dict(self._dataset_embeddings_raw)

        paper_state = {k: v[:] for k, v in self._paper_embeddings.items()}
        dataset_state = {k: v[:] for k, v in self._dataset_embeddings_raw.items()}
        dataset_accum = {k: v[:] for k, v in dataset_state.items()}

        for _ in range(max(0, layers)):
            next_paper: dict[str, list[float]] = {}
            for pid, datasets in self._paper_dataset_graph.items():
                if not datasets:
                    next_paper[pid] = paper_state[pid]
                    continue
                neighbors = [dataset_state[d] for d in datasets if d in dataset_state]
                next_paper[pid] = _avg_vectors(neighbors) if neighbors else paper_state[pid]

            paper_to_datasets = self._paper_dataset_graph
            next_dataset: dict[str, list[float]] = {}
            dataset_to_papers: dict[str, list[str]] = defaultdict(list)
            for pid, datasets in paper_to_datasets.items():
                for did in datasets:
                    dataset_to_papers[did].append(pid)
            for did, emb in dataset_state.items():
                pids = dataset_to_papers.get(did, [])
                if not pids:
                    next_dataset[did] = emb
                    continue
                neighbors = [next_paper[p] for p in pids if p in next_paper]
                next_dataset[did] = _avg_vectors(neighbors) if neighbors else emb

            paper_state = next_paper
            dataset_state = next_dataset
            for did, emb in dataset_state.items():
                dataset_accum[did] = _add_vectors(dataset_accum[did], emb)

        scale = float(max(1, layers + 1))
        return {did: _scale_vector(emb, 1.0 / scale) for did, emb in dataset_accum.items()}

    def _fit_graph_contrastive_projection(self) -> None:
        """
        Train query projection f_phi with InfoNCE (paper equation style):
        z_p = f_phi(Emb(x_p)), z_d from LightGCN-enriched dataset embeddings.
        """
        if torch is None or F is None:
            # Keep deterministic fallback when torch is unavailable.
            self._projection_matrix = None
            return
        paper_ids = list(self._paper_embeddings.keys())
        dataset_ids = list(self._dataset_embeddings_graph.keys())
        if not paper_ids or not dataset_ids:
            self._projection_matrix = None
            return

        dim = len(next(iter(self._paper_embeddings.values())))
        if dim == 0:
            self._projection_matrix = None
            return

        dataset_index = {did: i for i, did in enumerate(dataset_ids)}
        d_matrix = torch.tensor(
            [self._dataset_embeddings_graph[did] for did in dataset_ids],
            dtype=torch.float32,
        )
        d_matrix = F.normalize(d_matrix, dim=1)

        # Positive pairs from observed usage edges.
        positives: list[tuple[str, str]] = []
        seen_pairs = set()
        for pid, datasets in self._paper_dataset_graph.items():
            for did in datasets:
                key = (pid, did)
                if key not in seen_pairs and did in dataset_index:
                    positives.append(key)
                    seen_pairs.add(key)
        if not positives:
            self._projection_matrix = None
            return

        linear = torch.nn.Linear(dim, dim, bias=False)
        optimizer = torch.optim.Adam(linear.parameters(), lr=self.cfg.contrastive_lr)
        tau = self.cfg.contrastive_temperature
        neg_k = self.cfg.contrastive_negatives
        batch_size = self.cfg.contrastive_batch_size

        for _ in range(self.cfg.contrastive_epochs):
            random.shuffle(positives)
            for start in range(0, len(positives), batch_size):
                batch = positives[start : start + batch_size]
                p_emb = torch.tensor(
                    [self._paper_embeddings[pid] for pid, _ in batch],
                    dtype=torch.float32,
                )
                z_p = F.normalize(linear(p_emb), dim=1)
                pos_indices = torch.tensor(
                    [dataset_index[did] for _, did in batch],
                    dtype=torch.long,
                )

                logits_rows = []
                labels = []
                for i in range(len(batch)):
                    pos_idx = int(pos_indices[i].item())
                    all_indices = list(range(len(dataset_ids)))
                    all_indices.remove(pos_idx)
                    sampled_neg = random.sample(all_indices, k=min(neg_k, len(all_indices)))
                    cand_indices = [pos_idx] + sampled_neg
                    z_d = d_matrix[cand_indices]
                    sim = torch.matmul(z_p[i : i + 1], z_d.T).squeeze(0) / tau
                    logits_rows.append(sim)
                    labels.append(0)
                logits = torch.stack(logits_rows)
                target = torch.tensor(labels, dtype=torch.long)
                loss = F.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._projection_matrix = linear.weight.detach().cpu().tolist()

    def retrieve(
        self,
        paper: QueryPaper,
        enabled_channels: list[Literal["dense", "graph", "open_world"]] | None = None,
    ) -> list[CandidateRecord]:
        fused: dict[str, CandidateRecord] = {}
        query_embedding = self.llm.embed_texts([paper.to_text()])[0]
        channels = enabled_channels or ["dense", "graph", "open_world"]

        if "dense" in channels:
            dense_hits = self._dense_retrieve(query_embedding)
            for hit in dense_hits:
                self._merge_candidate(fused, hit, "dense", self.cfg.dense_weight)

        if "graph" in channels:
            graph_hits = self._graph_retrieve(query_embedding)
            for hit in graph_hits:
                self._merge_candidate(fused, hit, "graph", self.cfg.graph_weight)

        if "open_world" in channels:
            open_world_hits = self._open_world_generate(paper)
            for hit in open_world_hits:
                self._merge_candidate(fused, hit, "open_world", self.cfg.open_world_weight)

        return sorted(fused.values(), key=lambda x: x.retrieval_score, reverse=True)

    def _dense_retrieve(self, query_embedding: list[float]) -> list[CandidateRecord]:
        scored: list[tuple[float, UsageRecord]] = []
        for rec in self.records:
            rec_embedding = self._record_embeddings.get(rec.record_id, [])
            score = cosine_similarity(query_embedding, rec_embedding)
            if score > 0.0:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            CandidateRecord(usage_record=rec, retrieval_score=score)
            for score, rec in scored[: self.cfg.dense_top_k]
        ]

    def _graph_retrieve(self, query_embedding: list[float]) -> list[CandidateRecord]:
        z_query = self._project_query_to_graph_space(query_embedding)
        dataset_scores: list[tuple[float, str]] = []
        for dataset_entity, dataset_emb in self._dataset_embeddings_graph.items():
            score = cosine_similarity(z_query, dataset_emb)
            if score > 0.0:
                dataset_scores.append((score, dataset_entity))
        dataset_rank = sorted(dataset_scores, key=lambda x: x[0], reverse=True)
        hits: list[CandidateRecord] = []
        for score, dataset_entity in dataset_rank[: self.cfg.graph_top_k]:
            rep = self._select_best_record_for_dataset(dataset_entity, query_embedding)
            hits.append(CandidateRecord(usage_record=rep, retrieval_score=score))
        return hits

    def _open_world_generate(self, paper: QueryPaper) -> list[CandidateRecord]:
        dataset_inventory = sorted({r.dataset_entity for r in self.records})
        system_prompt = (
            "You are the Open-World Retrieval Agent in DatasetSAGE. "
            "Given a query paper, propose potentially relevant datasets by parametric knowledge, "
            "then map to dataset inventory when possible.\n"
            f"Core instruction: {OPEN_WORLD_CORE_INSTRUCTION}"
        )
        user_prompt = (
            f"Query paper title: {paper.title}\n"
            f"Query abstract: {paper.abstract}\n"
            f"Task goal: {paper.task_goal}\n\n"
            f"Dataset inventory: {dataset_inventory}\n\n"
            "Return JSON with key 'candidates' as a list of objects. "
            "Each object has dataset_name, mapped_dataset_entity, confidence (0-1), reason."
        )
        payload = self.llm.chat_json(system_prompt, user_prompt)
        candidates = payload.get("candidates", [])
        hits: list[CandidateRecord] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            mapped = str(item.get("mapped_dataset_entity", "")).strip().lower()
            dataset_name = str(item.get("dataset_name", "")).strip().lower()
            entity = self._dataset_alias.get(mapped) or self._dataset_alias.get(dataset_name)
            if not entity:
                continue
            rec = self._dataset_to_records[entity][0]
            try:
                score = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                score = 0.5
            score = max(0.0, min(1.0, score))
            hits.append(CandidateRecord(usage_record=rec, retrieval_score=score))
            if len(hits) >= self.cfg.open_world_top_k:
                break
        return hits

    def _project_query_to_graph_space(self, query_embedding: list[float]) -> list[float]:
        if not self._projection_matrix:
            return query_embedding
        # z_p = f_phi(Emb(x_p)), linear projection.
        out = []
        for row in self._projection_matrix:
            total = 0.0
            for val, weight in zip(query_embedding, row):
                total += val * weight
            out.append(total)
        norm = sum(v * v for v in out) ** 0.5
        if norm == 0.0:
            return out
        return [v / norm for v in out]

    def _select_best_record_for_dataset(
        self,
        dataset_entity: str,
        query_embedding: list[float],
    ) -> UsageRecord:
        records = self._dataset_to_records.get(dataset_entity, [])
        if not records:
            raise RuntimeError(f"Dataset '{dataset_entity}' has no usage records.")
        best = records[0]
        best_score = -1.0
        for rec in records:
            rec_embedding = self._record_embeddings.get(rec.record_id, [])
            score = cosine_similarity(query_embedding, rec_embedding)
            if score > best_score:
                best_score = score
                best = rec
        return best

    @staticmethod
    def _merge_candidate(
        fused: dict[str, CandidateRecord],
        candidate: CandidateRecord,
        channel: str,
        weight: float,
    ) -> None:
        key = candidate.record_id
        weighted = candidate.retrieval_score * weight
        if key not in fused:
            fused[key] = CandidateRecord(
                usage_record=candidate.usage_record,
                channels=[channel],
                retrieval_score=weighted,
            )
            return
        existing = fused[key]
        if channel not in existing.channels:
            existing.channels.append(channel)
        # RRF-like accumulation keeps cross-channel evidence.
        existing.retrieval_score += weighted


def _avg_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            sums[i] += val
    n = float(len(vectors))
    return [v / n for v in sums]


def _add_vectors(left: list[float], right: list[float]) -> list[float]:
    return [l + r for l, r in zip(left, right)]


def _scale_vector(vec: list[float], scale: float) -> list[float]:
    return [v * scale for v in vec]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for lval, rval in zip(left, right):
        dot += lval * rval
        left_sq += lval * lval
        right_sq += rval * rval
    denom = (left_sq ** 0.5) * (right_sq ** 0.5)
    if denom == 0.0:
        return 0.0
    return dot / denom
