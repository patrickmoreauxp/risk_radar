import re
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional at runtime
    SentenceTransformer = None

logger = logging.getLogger(__name__)


def _select_cluster_count(sample_count, max_clusters=40):
    if sample_count <= 3:
        return 1
    # Over-cluster first, then merge similar clusters downstream.
    heuristic = max(2, int(sample_count ** 0.60))
    return min(max_clusters, heuristic, sample_count)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return []
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    return sentences if sentences else [cleaned]


def _extract_context_window(text, mention_text=None, sentence_window=1, max_chars=1200):
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    mention = str(mention_text or "").strip().lower()
    hit_idx = None
    if mention:
        for idx, sentence in enumerate(sentences):
            if mention in sentence.lower():
                hit_idx = idx
                break

    if hit_idx is None:
        # Fallback: use first 2-3 sentences when no explicit mention location found.
        selected = sentences[: min(3, len(sentences))]
    else:
        start = max(0, hit_idx - sentence_window)
        end = min(len(sentences), hit_idx + sentence_window + 1)
        selected = sentences[start:end]

    context = " ".join(selected).strip()
    if len(context) > max_chars:
        context = context[: max_chars - 3].rstrip() + "..."
    return context


def _embed_texts(texts):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers unavailable for strict clustering mode")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        logger.info("CLUSTER_MODE=sentence_embeddings model=all-MiniLM-L6-v2")
        return embeddings, "sentence_embeddings"
    except Exception as exc:
        raise RuntimeError(f"Sentence embedding failed: {exc}") from exc


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _merge_similar_clusters(matrix, labels, similarity_threshold=0.90):
    """
    Merge only when two clusters are mutual nearest neighbors above threshold.
    This avoids transitive chain-merging that can collapse many narratives into one.
    """
    labels = np.asarray(labels).astype(int).copy()
    dense = _to_dense(matrix)
    if dense.ndim != 2 or dense.shape[0] == 0:
        return labels

    while True:
        unique_labels = sorted(np.unique(labels).tolist())
        if len(unique_labels) <= 1:
            break

        centroids = []
        sizes = []
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            centroid = dense[idx].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
            sizes.append(len(idx))

        centroid_matrix = np.vstack(centroids)
        sim = centroid_matrix @ centroid_matrix.T
        np.fill_diagonal(sim, -1.0)
        nearest = sim.argmax(axis=1)

        merges = []
        used = set()
        for i in range(len(unique_labels)):
            if i in used:
                continue
            j = int(nearest[i])
            if j < 0 or j == i or j in used:
                continue
            # Mutual nearest-neighbor constraint.
            if int(nearest[j]) != i:
                continue
            pair_sim = float(sim[i, j])
            if pair_sim < similarity_threshold:
                continue

            # Keep tiny clusters available to merge; avoid merging two already-large clusters too eagerly.
            if sizes[i] >= 20 and sizes[j] >= 20 and pair_sim < (similarity_threshold + 0.03):
                continue

            left = unique_labels[min(i, j)]
            right = unique_labels[max(i, j)]
            merges.append((left, right))
            used.add(i)
            used.add(j)

        if not merges:
            break

        for keep_label, drop_label in merges:
            labels[labels == drop_label] = keep_label

    # Reindex to contiguous labels.
    final_unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(final_unique)}
    return np.array([remap[int(v)] for v in labels.tolist()], dtype=int)


def _cluster_subset(subset, text_col, max_clusters):
    chunk = subset.copy()
    if "risk_embedding" in chunk.columns and len(chunk):
        emb_values = chunk["risk_embedding"].tolist()
        valid = (
            isinstance(emb_values, list)
            and len(emb_values) == len(chunk)
            and all(isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 for v in emb_values)
        )
    else:
        valid = False

    if valid:
        matrix = np.array([np.asarray(v, dtype=float) for v in emb_values], dtype=float)
        # normalize for cosine-style behavior
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        logger.info("CLUSTER_EXECUTION_MODE=openai_summary_embeddings rows=%d", len(chunk))
        n_clusters = _select_cluster_count(len(chunk), max_clusters=max_clusters)
        if n_clusters <= 1:
            chunk["cluster_context"] = chunk.get("risk_summary", "")
            chunk["_local_narrative_id"] = 0
            return chunk
        unique_vectors = max(1, matrix.shape[0])
        n_clusters = min(n_clusters, unique_vectors)
        if n_clusters <= 1:
            chunk["cluster_context"] = chunk.get("risk_summary", "")
            chunk["_local_narrative_id"] = 0
            return chunk
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        initial_labels = model.fit_predict(matrix)
        merged_labels = _merge_similar_clusters(matrix, initial_labels, similarity_threshold=0.95)
        chunk["cluster_context"] = chunk.get("risk_summary", "")
        chunk["_local_narrative_id"] = merged_labels.astype(int)
        return chunk

    text_series = chunk[text_col].fillna("").astype(str)
    if text_series.str.strip().eq("").all():
        chunk["cluster_context"] = ""
        chunk["_local_narrative_id"] = 0
        return chunk

    mention_series = (
        chunk["mention_text"].fillna("").astype(str)
        if "mention_text" in chunk.columns
        else ["" for _ in range(len(chunk))]
    )
    chunk["cluster_context"] = [
        _extract_context_window(text, mention)
        for text, mention in zip(text_series.tolist(), list(mention_series))
    ]
    context_series = chunk["cluster_context"].fillna("").astype(str)
    if context_series.str.strip().eq("").all():
        raise RuntimeError("Context extraction produced empty text for all rows")

    n_clusters = _select_cluster_count(len(chunk), max_clusters=max_clusters)
    if n_clusters <= 1:
        chunk["_local_narrative_id"] = 0
        return chunk

    matrix, cluster_mode = _embed_texts(context_series.tolist())
    logger.info("CLUSTER_EXECUTION_MODE=%s rows=%d", cluster_mode, len(chunk))
    unique_vectors = max(1, matrix.shape[0] if hasattr(matrix, "shape") else len(context_series))
    n_clusters = min(n_clusters, unique_vectors)
    if n_clusters <= 1:
        chunk["_local_narrative_id"] = 0
        return chunk

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    initial_labels = model.fit_predict(matrix)
    merged_labels = _merge_similar_clusters(matrix, initial_labels, similarity_threshold=0.90)
    chunk["_local_narrative_id"] = merged_labels.astype(int)
    return chunk


def cluster_narratives(posts, max_clusters=40, progress_callback=None):
    clustered = posts.copy()
    if clustered.empty:
        clustered["narrative_id"] = clustered.get("narrative_id", 0)
        clustered["cluster_context"] = clustered.get("cluster_context", "")
        if callable(progress_callback):
            progress_callback({"current": 0, "total": 0, "rows": 0})
        return clustered

    text_col = "text_altered" if "text_altered" in clustered.columns else "text"
    if text_col not in clustered.columns:
        raise ValueError("Missing text column required for clustering")

    # Enforce entity-scoped clustering so narratives never cross entities.
    entity_col = "entity_id" if "entity_id" in clustered.columns else ("entity" if "entity" in clustered.columns else None)
    if entity_col is None:
        clustered = _cluster_subset(clustered, text_col=text_col, max_clusters=max_clusters)
        clustered["narrative_id"] = clustered["_local_narrative_id"].astype(int) + 1
        if callable(progress_callback):
            progress_callback({"current": 1, "total": 1, "rows": int(len(clustered))})
        return clustered.drop(columns=["_local_narrative_id"])

    parts = []
    next_global_id = 1
    grouped_items = list(clustered.groupby(entity_col, dropna=False, sort=False))
    total_groups = len(grouped_items)
    for idx, (_, group) in enumerate(grouped_items, start=1):
        chunk = _cluster_subset(group, text_col=text_col, max_clusters=max_clusters)
        local_ids = chunk["_local_narrative_id"].astype(int)
        unique_local_ids = sorted(local_ids.unique().tolist())
        local_to_global = {
            local_id: next_global_id + idx
            for idx, local_id in enumerate(unique_local_ids)
        }
        next_global_id += len(unique_local_ids)
        chunk["narrative_id"] = local_ids.map(local_to_global).astype(int)
        parts.append(chunk.drop(columns=["_local_narrative_id"]))
        if callable(progress_callback):
            progress_callback(
                {
                    "current": idx,
                    "total": total_groups,
                    "rows": int(len(group)),
                    "entity": str(group[entity_col].iloc[0]) if len(group) else "-",
                }
            )

    # Keep stable row order from original index after groupwise processing.
    return pd.concat(parts).sort_index()
