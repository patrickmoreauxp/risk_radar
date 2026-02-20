import json
from pathlib import Path
import re
import logging

import pandas as pd
import numpy as np

from core.clustering import cluster_narratives
from core.entity_resolution import resolve_entities
from core.scoring import score_narratives

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    nltk = None
    SentimentIntensityAnalyzer = None

logger = logging.getLogger(__name__)


def _expand_resolved_entities(resolved_posts, entities_df):
    id_to_name = (
        entities_df[["entity_id", "canonical_name"]]
        .dropna(subset=["entity_id"])
        .assign(entity_id=lambda d: d["entity_id"].astype(str))
        .set_index("entity_id")["canonical_name"]
        .to_dict()
    )

    expanded_rows = []
    for _, row in resolved_posts.iterrows():
        base = row.to_dict()
        matches = base.get("resolved_entities", [])
        if not isinstance(matches, list) or not matches:
            continue

        # Keep at most one expanded row per (post, entity_id).
        # If the same entity matched multiple aliases/methods, keep the highest-confidence hit.
        best_by_entity = {}
        for match in matches:
            entity_id = str(match.get("entity_id", "")).strip()
            if not entity_id:
                continue
            candidate_conf = float(match.get("confidence", 0.0) or 0.0)
            current = best_by_entity.get(entity_id)
            if current is None or candidate_conf > current["confidence"]:
                best_by_entity[entity_id] = {
                    "mention_text": str(match.get("mention_text", "")).strip(),
                    "confidence": candidate_conf,
                }

        for entity_id, best in best_by_entity.items():
            expanded = dict(base)
            expanded["entity_id"] = entity_id
            expanded["entity"] = id_to_name.get(entity_id, base.get("entity"))
            expanded["mention_text"] = best["mention_text"]
            expanded["resolution_confidence"] = best["confidence"]
            expanded_rows.append(expanded)
    if not expanded_rows:
        return pd.DataFrame(columns=list(resolved_posts.columns) + ["entity_id", "mention_text"])
    return pd.DataFrame(expanded_rows)


TOKEN_RE = re.compile(r"[A-Za-z0-9#@$&'\-]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPACY_NLP = None
_SPACY_MODE = None
_VADER_ANALYZER = None
_VADER_MODE_LOGGED = False


def _signal_text(row):
    if "text_altered" in row and pd.notna(row.get("text_altered")) and str(row.get("text_altered")).strip():
        return str(row.get("text_altered"))
    return str(row.get("text", ""))


def _prefilter_posts(posts):
    evaluated = posts.copy()
    evaluated["_signal_text"] = evaluated.apply(_signal_text, axis=1)
    evaluated["_word_count"] = evaluated["_signal_text"].apply(lambda t: len(TOKEN_RE.findall(str(t))))
    evaluated["_drop_reason"] = evaluated["_word_count"].apply(
        lambda w: "too_long_over_2000_words" if int(w) > 2000 else None
    )

    analyzer = _get_vader_analyzer()
    if analyzer is not None:
        evaluated["_vader_compound"] = evaluated["_signal_text"].apply(
            lambda t: float(analyzer.polarity_scores(str(t)).get("compound", 0.0))
        )
        evaluated.loc[
            evaluated["_drop_reason"].isna() & (evaluated["_vader_compound"] >= 0.0),
            "_drop_reason",
        ] = "non_negative_vader_compound"
    else:
        # If VADER is unavailable, fail open but keep visibility.
        evaluated["_vader_compound"] = np.nan

    dropped = evaluated[evaluated["_drop_reason"].notna()].copy()
    kept = evaluated[evaluated["_drop_reason"].isna()].copy()
    dropped = dropped.drop(columns=["_signal_text"])
    kept = kept.drop(columns=["_signal_text", "_drop_reason", "_word_count", "_vader_compound"])
    return kept, dropped


HIGH_SEVERITY_RISK_TERMS = [
    "lawsuit", "investigation", "fraud", "recall", "adverse event", "safety concern",
    "side effect", "data breach", "cyberattack", "regulator", "regulatory action",
    "penalty", "fine", "misconduct", "corruption", "boycott",
]

MEDIUM_SEVERITY_RISK_TERMS = [
    "risk", "risky", "warning", "complaint", "concern", "issue", "outage", "shortage",
    "supply disruption", "price hike", "quality issue", "non-compliance", "violation",
    "negative reaction", "backlash", "downgrade", "missed guidance", "decline",
]


def _get_vader_analyzer():
    global _VADER_ANALYZER, _VADER_MODE_LOGGED
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER
    if SentimentIntensityAnalyzer is None:
        raise RuntimeError("VADER package unavailable for strict sentiment prefiltering")
    try:
        if nltk is not None:
            nltk.download("vader_lexicon", quiet=True)
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        if not _VADER_MODE_LOGGED:
            logger.info("VADER_MODE=enabled_filter_negative_compound_only")
            _VADER_MODE_LOGGED = True
        return _VADER_ANALYZER
    except Exception as exc:
        raise RuntimeError(f"VADER initialization failed: {exc}") from exc


def _get_spacy_nlp():
    global _SPACY_NLP, _SPACY_MODE
    if _SPACY_MODE is not None:
        return _SPACY_NLP

    if _SPACY_NLP is not None:
        if _SPACY_MODE is None:
            _SPACY_MODE = "spacy_cached"
        return _SPACY_NLP
    if spacy is None:
        raise RuntimeError("spaCy package unavailable for strict context extraction")
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        _SPACY_MODE = "spacy_en_core_web_sm"
        return _SPACY_NLP
    except Exception as exc:
        raise RuntimeError(f"spaCy model load failed (en_core_web_sm): {exc}") from exc


def _extract_entity_context(text, mention_text, sentence_window=2, max_words=100, nlp=None):
    raw = " ".join(str(text).split())
    if not raw:
        return ""
    mention = str(mention_text or "").strip().lower()
    if nlp is None:
        nlp = _get_spacy_nlp()
    if nlp is not None:
        doc = nlp(raw)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(raw) if s.strip()]
    if not sentences:
        sentences = [raw]

    hit_idx = None
    if mention:
        for idx, sent in enumerate(sentences):
            if mention in sent.lower():
                hit_idx = idx
                break
    if hit_idx is None:
        selected = sentences[: min(2 * sentence_window + 1, len(sentences))]
    else:
        start = max(0, hit_idx - sentence_window)
        end = min(len(sentences), hit_idx + sentence_window + 1)
        selected = sentences[start:end]

    context = " ".join(selected).strip()
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words]).strip()
    return context


def _is_generic_post_summary(summary):
    text = " ".join(str(summary).lower().split())
    if not text:
        return True
    generic_prefixes = [
        "posts mentioning",
        "this context indicates risk",
        "discussion focused on potential",
        "this post indicates risk signals",
    ]
    return any(text.startswith(prefix) for prefix in generic_prefixes)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i: i + size]


def _clean_json_text(raw_text):
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


def _escape_unescaped_summary_quotes(raw_json):
    """
    Deterministically escape stray quote characters inside summary values.
    This keeps strict schema parsing while handling model outputs like:
    ..."summary":"text with "quoted phrase" inside"...
    """
    text = str(raw_json or "")
    marker = '"summary":"'
    i = 0
    out = []
    text_len = len(text)
    while i < text_len:
        pos = text.find(marker, i)
        if pos < 0:
            out.append(text[i:])
            break
        out.append(text[i: pos + len(marker)])
        j = pos + len(marker)
        while j < text_len:
            ch = text[j]
            if ch == "\\":
                if j + 1 < text_len:
                    out.append(text[j: j + 2])
                    j += 2
                else:
                    out.append(ch)
                    j += 1
                continue
            if ch == '"':
                k = j + 1
                while k < text_len and text[k].isspace():
                    k += 1
                if k < text_len and text[k] in {",", "}"}:
                    out.append('"')
                    j += 1
                    break
                out.append('\\"')
                j += 1
                continue
            out.append(ch)
            j += 1
        i = j
    return "".join(out)


def _llm_entity_risk_batch(client, items, model):
    """
    items: list of {"id": str, "entity": str, "context": str}
    returns: dict[id] -> (is_risk: bool|None, summary: str|None)
    """
    if not items:
        return {}

    expected_ids = [str(item["id"]) for item in items]
    items_payload = [
        {
            "id": str(item["id"]),
            "entity": str(item["entity"]),
            "context": str(item["context"]),
        }
        for item in items
    ]
    prompt = (
        "You will evaluate multiple SINGLE-POST contexts for entity-linked risk.\n"
        "Return STRICT JSON only in this shape:\n"
        '{"results":[{"id":"<id>","is_risk":true|false,"summary":"<one sentence>"}]}\n'
        "Rules:\n"
        "- classify as risk when content is potentially inflammatory with regard to the entity in question\n"
        "- if the text is vague, unclear, or lacks a concrete entity-linked claim, set is_risk=false\n"
        "- summary <= 35 words\n"
        "- summary must describe the exact claim in that single post about the entity\n"
        "- summary must include at least one concrete allegation/signal from the context (event, action, harm, accusation, or impact)\n"
        "- if is_risk=false, summary MUST be an empty string\n"
        "- summary must be specific, not generic\n"
        "- NEVER use generic boilerplate such as: 'this context indicates risk', 'discussion focused on potential', 'this post indicates risk signals', or 'posts mentioning ...'\n"
        "- output valid, minified JSON only (no markdown, no prose, no trailing text)\n"
        "- do not include unescaped quote characters inside summary text\n"
        "- no quotes, usernames, or copied phrases\n"
        "- include every required id exactly once; NEVER omit an id\n"
        "- when unsure, set is_risk=false for that id instead of omitting it\n\n"
        f"Required IDs:\n{json.dumps(expected_ids, ensure_ascii=True)}\n\n"
        f"Items JSON:\n{json.dumps(items_payload, ensure_ascii=True)}"
    )
    max_output_tokens = min(3500, max(1000, 220 * len(items)))
    attempts = 3
    last_error = None
    last_preview = ""
    for _attempt in range(1, attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            raw = getattr(response, "output_text", "") or ""
            cleaned = _clean_json_text(raw)
            if not cleaned:
                raise RuntimeError("LLM risk batch returned empty output_text")

            out = {}
            try:
                parsed = json.loads(cleaned)
            except Exception:
                repaired = _escape_unescaped_summary_quotes(cleaned)
                if repaired != cleaned:
                    parsed = json.loads(repaired)
                    cleaned = repaired
                else:
                    raise

            if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
                raise RuntimeError("LLM risk batch returned invalid schema (expected object with results list)")

            results = parsed.get("results", [])
            expected_ids_set = set(expected_ids)
            seen_ids = set()
            for row in results:
                rid = str(row.get("id", "")).strip()
                if not rid:
                    raise RuntimeError("LLM risk batch row missing id")
                if rid not in expected_ids_set:
                    raise RuntimeError(f"LLM risk batch returned unexpected id: {rid}")
                if rid in seen_ids:
                    raise RuntimeError(f"LLM risk batch returned duplicate id: {rid}")
                seen_ids.add(rid)

                is_risk = row.get("is_risk")
                if not isinstance(is_risk, bool):
                    raise RuntimeError(f"LLM risk batch row has non-boolean is_risk for id={rid}")
                summary = " ".join(str(row.get("summary", "")).replace('"', "").replace("'", "").split())
                if is_risk:
                    if _is_generic_post_summary(summary):
                        raise RuntimeError(f"LLM risk batch returned generic/empty summary for id={rid}")
                    out[rid] = (True, summary)
                else:
                    out[rid] = (False, "")

            missing = expected_ids_set - seen_ids
            if missing:
                raise RuntimeError(f"LLM risk batch missing ids: {sorted(missing)}")
            return out
        except Exception as exc:
            last_error = exc
            last_preview = _clean_json_text(getattr(locals().get("response", None), "output_text", "") or "")[:700]
            continue
    raise RuntimeError(
        f"LLM risk batch validation failed after {attempts} attempts: {last_error} | raw_preview={last_preview}"
    ) from last_error


def _enrich_entity_rows_with_risk(
    expanded_posts,
    openai_api_key=None,
    llm_model="gpt-4.1-nano",
    embedding_model="text-embedding-3-small",
    progress_callback=None,
):
    if expanded_posts is None or len(expanded_posts) == 0:
        return expanded_posts, expanded_posts

    df = expanded_posts.copy()
    text_col = "text_altered" if "text_altered" in df.columns else "text"
    if text_col not in df.columns:
        raise ValueError("Missing text column required for risk check")

    if OpenAI is None:
        raise RuntimeError("OpenAI package unavailable for risk enrichment")

    if not str(openai_api_key or "").strip():
        raise RuntimeError("OpenAI API key missing for risk enrichment")
    client = OpenAI(api_key=openai_api_key)

    nlp = _get_spacy_nlp()
    logger.info("RISK_CONTEXT_MODE=%s", _SPACY_MODE)
    llm_cache = {}
    emb_cache = {}
    kept_rows = []
    dropped_rows = []
    total = int(len(df))
    kept_count = 0
    dropped_count = 0

    # Build contexts first and deduplicate LLM calls by (entity, context).
    row_items = []
    unique_items = {}
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        entity_name = str(row.get("entity", "") or row.get("entity_id", "")).strip() or "Entity"
        mention_text = str(row.get("mention_text", "")).strip()
        text_value = str(row.get(text_col, "")).strip()
        context = _extract_entity_context(
            text_value,
            mention_text,
            sentence_window=2,
            max_words=100,
            nlp=nlp,
        )
        key = (entity_name, context)
        row_items.append({"row": row, "entity": entity_name, "context": context, "key": key, "row_idx": idx})
        if key not in unique_items:
            uid = f"u{len(unique_items)}"
            unique_items[key] = {"id": uid, "entity": entity_name, "context": context}
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "context_progress",
                    "current": idx,
                    "total": total,
                }
            )

    # Batched LLM calls for risk decision + per-post claim summary.
    unique_list = list(unique_items.values())
    id_to_result = {}
    # Smaller batches improve strict-schema reliability (all IDs returned) for nano models.
    llm_batch_size = 2
    llm_batches_total = (len(unique_list) + llm_batch_size - 1) // llm_batch_size if unique_list else 0
    if callable(progress_callback):
        progress_callback(
            {
                "phase": "risk_stage_start",
                "total_rows": total,
                "unique_contexts": len(unique_list),
                "llm_batches_total": llm_batches_total,
            }
        )

    llm_batches_done = 0
    llm_unique_processed = 0
    for batch_idx, batch in enumerate(_chunks(unique_list, llm_batch_size), start=1):
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "llm_batch_start",
                    "batch_index": batch_idx,
                    "batch_total": llm_batches_total,
                    "batch_size": len(batch),
                    "unique_processed": llm_unique_processed,
                    "unique_total": len(unique_list),
                }
            )
        batch_result = _llm_entity_risk_batch(client, batch, llm_model)
        id_to_result.update(batch_result)
        llm_batches_done = batch_idx
        llm_unique_processed += len(batch)
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "llm_batch_done",
                    "batch_index": batch_idx,
                    "batch_total": llm_batches_total,
                    "batch_size": len(batch),
                    "unique_processed": llm_unique_processed,
                    "unique_total": len(unique_list),
                }
            )

    for key, item in unique_items.items():
        llm_cache[key] = id_to_result.get(item["id"], (None, None))

    # Batched embedding calls for unique accepted summaries.
    accepted_summaries = []
    for result in llm_cache.values():
        is_risk, summary = result
        if is_risk and summary:
            accepted_summaries.append(summary)
    unique_summaries = list(dict.fromkeys(accepted_summaries))
    embed_batch_size = 64
    embed_batches_total = (len(unique_summaries) + embed_batch_size - 1) // embed_batch_size if unique_summaries else 0
    embed_batches_done = 0
    embed_processed = 0
    for batch_idx, batch in enumerate(_chunks(unique_summaries, embed_batch_size), start=1):
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "embedding_batch_start",
                    "batch_index": batch_idx,
                    "batch_total": embed_batches_total,
                    "batch_size": len(batch),
                    "embedded_processed": embed_processed,
                    "embedded_total": len(unique_summaries),
                }
            )
        emb = client.embeddings.create(model=embedding_model, input=batch)
        if not emb or not emb.data or len(emb.data) != len(batch):
            raise RuntimeError("Embedding response mismatch for risk summaries")
        for i, datum in enumerate(emb.data):
            emb_cache[batch[i]] = datum.embedding
        embed_batches_done = batch_idx
        embed_processed += len(batch)
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "embedding_batch_done",
                    "batch_index": batch_idx,
                    "batch_total": embed_batches_total,
                    "batch_size": len(batch),
                    "embedded_processed": embed_processed,
                    "embedded_total": len(unique_summaries),
                }
            )

    for item in row_items:
        row = item["row"]
        entity_name = item["entity"]
        context = item["context"]
        is_risk, summary = llm_cache.get(item["key"], (None, None))

        if is_risk and summary:
            embedding = emb_cache.get(summary)
            if embedding is None:
                raise RuntimeError("Missing embedding for accepted risk summary")
            else:
                enriched = row.to_dict()
                enriched["risk_context"] = context
                enriched["risk_summary"] = summary
                enriched["risk_embedding"] = embedding
                kept_rows.append(enriched)
                kept_count += 1
        else:
            rejected = row.to_dict()
            rejected["risk_context"] = context
            if is_risk is False:
                rejected["_drop_reason"] = "llm_marked_not_risk"
            else:
                raise RuntimeError("Unexpected LLM risk state encountered")
            dropped_rows.append(rejected)
            dropped_count += 1

        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "row_progress",
                    "current": int(item["row_idx"]),
                    "total": total,
                    "kept": kept_count,
                    "dropped": dropped_count,
                    "entity": entity_name,
                    "llm_batches_done": llm_batches_done,
                    "llm_batches_total": llm_batches_total,
                    "embed_batches_done": embed_batches_done,
                    "embed_batches_total": embed_batches_total,
                }
            )

    kept_df = pd.DataFrame(kept_rows) if kept_rows else df.iloc[0:0].copy()
    dropped_df = pd.DataFrame(dropped_rows) if dropped_rows else df.iloc[0:0].copy()
    if callable(progress_callback):
        progress_callback(
            {
                "phase": "risk_stage_done",
                "total": total,
                "kept": kept_count,
                "dropped": dropped_count,
                "llm_batches_done": llm_batches_done,
                "llm_batches_total": llm_batches_total,
                "embed_batches_done": embed_batches_done,
                "embed_batches_total": embed_batches_total,
            }
        )
    return kept_df, dropped_df


def _save_artifacts(
    output_dir,
    resolved_posts,
    clustered_posts,
    narratives,
    filtered_out_posts=None,
    filtered_out_entity_rows=None,
    progress_callback=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    resolved_export = resolved_posts.copy()
    if "resolved_entities" in resolved_export.columns:
        resolved_export["resolved_entities"] = resolved_export["resolved_entities"].apply(
            lambda x: json.dumps(x, ensure_ascii=True)
        )
    total_files = 5
    if filtered_out_posts is not None:
        total_files += 1
    if filtered_out_entity_rows is not None:
        total_files += 1

    written = 0
    resolved_export.to_json(output / "resolved_posts.jsonl", orient="records", lines=True, date_format="iso")
    written += 1
    if callable(progress_callback):
        progress_callback({"current": written, "total": total_files, "file": "resolved_posts.jsonl"})

    clustered_export = clustered_posts.copy()
    # Keep runtime artifacts lean; embeddings are only needed during clustering.
    if "risk_embedding" in clustered_export.columns:
        clustered_export = clustered_export.drop(columns=["risk_embedding"])
    if "resolved_entities" in clustered_export.columns:
        clustered_export["resolved_entities"] = clustered_export["resolved_entities"].apply(
            lambda x: json.dumps(x, ensure_ascii=True)
        )
    clustered_export.to_json(output / "clustered_posts.jsonl", orient="records", lines=True, date_format="iso")
    written += 1
    if callable(progress_callback):
        progress_callback({"current": written, "total": total_files, "file": "clustered_posts.jsonl"})

    if filtered_out_posts is not None:
        dropped_export = filtered_out_posts.copy()
        dropped_export.to_json(output / "filtered_out_posts.jsonl", orient="records", lines=True, date_format="iso")
        written += 1
        if callable(progress_callback):
            progress_callback({"current": written, "total": total_files, "file": "filtered_out_posts.jsonl"})
    if filtered_out_entity_rows is not None:
        dropped_entity_export = filtered_out_entity_rows.copy()
        dropped_entity_export.to_json(
            output / "filtered_out_entity_rows.jsonl",
            orient="records",
            lines=True,
            date_format="iso",
        )
        written += 1
        if callable(progress_callback):
            progress_callback({"current": written, "total": total_files, "file": "filtered_out_entity_rows.jsonl"})
    narratives_export = []
    for n in narratives:
        narratives_export.append(
            {
                "narrative_id": int(n.get("narrative_id", 0)),
                "title": str(n.get("title", "")),
                "summary": str(n.get("summary", "")),
                "score": float(n.get("score", 0.0)),
                "risk_score_confidence": str(n.get("risk_score_confidence", "")),
                "volume": int(n.get("volume", 0)),
                "engagement": int(n.get("engagement", 0)),
                "post_ids": [str(p) for p in n.get("post_ids", [])],
                "taxonomy_labels": [str(t) for t in n.get("taxonomy_labels", [])],
            }
        )

    with (output / "narratives.json").open("w", encoding="utf-8") as handle:
        json.dump(narratives_export, handle, ensure_ascii=False, indent=2)
    written += 1
    if callable(progress_callback):
        progress_callback({"current": written, "total": total_files, "file": "narratives.json"})


def _attach_narrative_metadata(clustered_posts, narratives):
    if clustered_posts is None or len(clustered_posts) == 0:
        return clustered_posts
    enriched = clustered_posts.copy()
    meta = {
        int(n.get("narrative_id", 0)): {
            "narrative_title": str(n.get("title", "")),
            "narrative_summary": str(n.get("summary", "")),
        }
        for n in narratives
    }
    enriched["narrative_title"] = enriched["narrative_id"].map(
        lambda nid: meta.get(int(nid), {}).get("narrative_title", "")
    )
    enriched["narrative_summary"] = enriched["narrative_id"].map(
        lambda nid: meta.get(int(nid), {}).get("narrative_summary", "")
    )
    return enriched


def build_preprocessed_pipeline(
    posts,
    authors,
    entities,
    output_dir="data/processed",
    stage_callback=None,
    entity_progress_callback=None,
    risk_progress_callback=None,
    clustering_progress_callback=None,
    scoring_progress_callback=None,
    save_progress_callback=None,
    openai_api_key=None,
    llm_model="gpt-4.1-nano",
):
    if callable(stage_callback):
        stage_callback("preprocess_start")
    filtered_posts, dropped_posts = _prefilter_posts(posts)
    if callable(stage_callback):
        stage_callback("preprocess_done")

    if callable(stage_callback):
        stage_callback("entity_resolution_start")
    resolved_posts = resolve_entities(
        filtered_posts,
        entities,
        progress_callback=entity_progress_callback,
    )
    if callable(stage_callback):
        stage_callback("entity_resolution_done")

    if callable(stage_callback):
        stage_callback("expand_entities_start")
    expanded_posts = _expand_resolved_entities(resolved_posts, entities)
    if callable(stage_callback):
        stage_callback("expand_entities_done")

    if callable(stage_callback):
        stage_callback("risk_enrichment_start")
    risk_relevant_posts, dropped_entity_rows = _enrich_entity_rows_with_risk(
        expanded_posts,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        progress_callback=risk_progress_callback,
    )
    if callable(stage_callback):
        stage_callback("risk_enrichment_done")

    if callable(stage_callback):
        stage_callback("clustering_start")
    clustered_posts = cluster_narratives(
        risk_relevant_posts,
        progress_callback=clustering_progress_callback,
    )
    if callable(stage_callback):
        stage_callback("clustering_done")

    if callable(stage_callback):
        stage_callback("scoring_start")
    narratives = score_narratives(
        clustered_posts,
        authors,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        progress_callback=scoring_progress_callback,
    )
    clustered_posts = _attach_narrative_metadata(clustered_posts, narratives)
    if callable(stage_callback):
        stage_callback("scoring_done")

    if callable(stage_callback):
        stage_callback("save_artifacts_start")
    _save_artifacts(
        output_dir,
        resolved_posts,
        clustered_posts,
        narratives,
        filtered_out_posts=dropped_posts,
        filtered_out_entity_rows=dropped_entity_rows,
        progress_callback=save_progress_callback,
    )
    if callable(stage_callback):
        stage_callback("save_artifacts_done")

    return {
        "raw_posts": posts,
        "filtered_posts": filtered_posts,
        "dropped_posts": dropped_posts,
        "risk_relevant_posts": risk_relevant_posts,
        "dropped_entity_rows": dropped_entity_rows,
        "preprocessing": {
            "input_posts": int(len(posts)),
            "kept_posts": int(len(filtered_posts)),
            "dropped_posts": int(len(dropped_posts)),
            "entity_rows_after_resolution": int(len(expanded_posts)),
            "entity_rows_risk_relevant": int(len(risk_relevant_posts)),
            "entity_rows_dropped_no_risk": int(len(dropped_entity_rows)),
        },
        "resolved_posts": resolved_posts,
        "clustered_posts": clustered_posts,
        "narratives": narratives,
        "entities": entities,
        "authors": authors,
    }
