from __future__ import annotations

import math
from collections import Counter, defaultdict


def _tokenize(text: str) -> list[str]:
    return [t.strip(".,!?;:\"'()[]{}").lower() for t in text.split() if t.strip()]


def combine_text(news_rows: list[dict]) -> list[str]:
    return [f"{r.get('Title', '')} {r.get('Abstract', '')}".strip() for r in news_rows]


def build_tfidf_features(news_rows: list[dict]) -> tuple[dict, dict[str, dict[str, float]]]:
    docs = combine_text(news_rows)
    tokenized = [_tokenize(d) for d in docs]
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))

    n_docs = max(1, len(docs))
    idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    vectors: dict[str, dict[str, float]] = {}
    for row, toks in zip(news_rows, tokenized):
        tf = Counter(toks)
        norm = sum(tf.values()) or 1
        vec = {term: (count / norm) * idf[term] for term, count in tf.items()}
        vectors[row["NewsID"]] = vec
    return {"method": "tfidf", "idf": idf}, vectors


def mean_vector(news_ids: list[str], article_vectors: dict[str, dict[str, float]]) -> dict[str, float]:
    vecs = [article_vectors.get(nid, {}) for nid in news_ids]
    vecs = [v for v in vecs if v]
    if not vecs:
        return {}
    acc = Counter()
    for v in vecs:
        acc.update(v)
    denom = len(vecs)
    return {k: val / denom for k, val in acc.items()}


def build_user_subcategory_profile(
    user_history_ids: list[str],
    news_by_id: dict[str, dict],
    allowed_subcategories: set[str],
) -> dict:
    subcats = [
        news_by_id.get(nid, {}).get("SubCategory", "").strip().lower()
        for nid in user_history_ids
        if news_by_id.get(nid)
    ]
    filtered = [s for s in subcats if s in allowed_subcategories]
    counts = Counter(filtered)
    total = sum(counts.values())
    dominant = None
    if counts:
        dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    freq = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {
        "dominant_subcategory": dominant,
        "subcategory_frequency": freq,
    }


def build_user_vectors(histories_rows: list[dict], article_vectors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    user_terms: dict[str, list[dict[str, float]]] = defaultdict(list)
    for h in histories_rows:
        vec = article_vectors.get(h["NewsID"])
        if vec:
            user_terms[h["UserID"]].append(vec)

    out: dict[str, dict[str, float]] = {}
    for uid, vecs in user_terms.items():
        acc = Counter()
        for v in vecs:
            acc.update(v)
        denom = len(vecs)
        out[uid] = {k: v / denom for k, v in acc.items()}
    return out


def build_sentence_transformer_features(news_rows: list[dict], model_name: str = "all-MiniLM-L6-v2") -> tuple[dict, dict[str, dict[str, float]]]:
    docs = combine_text(news_rows)
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)
        embs = encoder.encode(docs, convert_to_numpy=True)
        vectors: dict[str, dict[str, float]] = {}
        for row, emb in zip(news_rows, embs):
            vectors[row["NewsID"]] = {f"d{i}": float(v) for i, v in enumerate(emb.tolist())}
        return {"method": "sentence-transformer", "model_name": model_name}, vectors
    except Exception as exc:
        raise RuntimeError(
            "Sentence-transformer embedding requested but unavailable. "
            "Install sentence-transformers and model weights or use --embedding-method tfidf."
        ) from exc
