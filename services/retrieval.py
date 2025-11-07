from typing import List, Dict
from sqlalchemy import text
from sqlalchemy.engine import Engine
from services.embeddings import embed_query
from services.reranker import rerank_impacts
from config.settings import DEFAULT_CANDIDATES_TOP_K
from utils.logger import logger
# ======================================================
# 🧩 Core: Vector DB search for impacts
def search_impacts_by_text(engine: Engine, query_text: str, top_k: int = DEFAULT_CANDIDATES_TOP_K) -> List[Dict]:
    q_emb = embed_query(query_text)
    with engine.begin() as conn:
        sql = text("""
        SELECT impact, industry, industry_conf, business_group, business_group_conf,
               use_case, use_case_conf, source_type, source_file,
               1 - (embedding <=> :qemb) AS similarity
        FROM scout.impact_repository
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :qemb
        LIMIT :k
        """)
        rows = conn.execute(sql, {"qemb": q_emb, "k": top_k}).mappings().all()

    candidates = []
    for r in rows:
        candidates.append({
            "Impact": r["impact"],
            "Industry": r["industry"] or "",
            "BusinessGroup": r["business_group"] or "",
            "UseCase": r["use_case"] or "",
            "similarity": float(r["similarity"] or 0.0),
            "SourceType": r["source_type"],
            "SourceFile": r["source_file"],
        })
    return candidates

def retrieve_and_rerank(engine: Engine, persona_info: str, kpis: List[str], top_k: int = DEFAULT_CANDIDATES_TOP_K) -> List[Dict]:
    if not kpis:
        kpis = ["Operational Efficiency"]
    query = " ".join(kpis)
    logger.info("Retrieval query: %s", query)
    candidates = search_impacts_by_text(engine, query, top_k=top_k)

    # Optional: lightweight diversity pre-filter (drop exact duplicate impact lines)
    seen = set(); deduped = []
    for c in candidates:
        t = (c["Impact"] or "").strip()
        if t and t not in seen:
            seen.add(t); deduped.append(c)

    top3 = rerank_impacts(persona_info, kpis, deduped)
    print(top3)
    return top3
