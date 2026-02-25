"""Paper-aligned agent instruction templates (Appendix)."""

OPEN_WORLD_CORE_INSTRUCTION = (
    "Suggest K datasets highly relevant to this research. "
    "For each, provide the dataset name, a brief reasoning, and a relevance score in [0,1]."
)

REASON_CORE_INSTRUCTION = (
    "For each candidate, assess whether the evidence supports relevance to the query paper. "
    "Output a reason r_u (1-2 sentences) and a score s_u in [0,1] where 0 = unrelated, 1 = strong match."
)

REVIEW_CORE_INSTRUCTION = (
    "Summarize current coverage: what has been found and what might be missing. "
    "Propose the next retrieval query q^(t+1) to address gaps "
    "(e.g., alternative terminology, missing modalities)."
)

RERANK_CORE_INSTRUCTION = (
    "Compare aggregated evidence cards and rank the top-K datasets by relevance. "
    "For each, assign a final confidence in [0,1] and provide a brief justification."
)
