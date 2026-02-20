import numpy as np
import os
import re
import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None

_LLM_SUMMARY_CACHE = {}
_LLM_TAXONOMY_CACHE = {}
GENERIC_TITLE_PATTERNS = [
    re.compile(r"^\s*company narrative\s*$", re.IGNORECASE),
    re.compile(r"^\s*entity narrative\s*$", re.IGNORECASE),
    re.compile(r"^\s*general narrative\s*$", re.IGNORECASE),
    re.compile(r"^\s*risk narrative\s*$", re.IGNORECASE),
    re.compile(r"^\s*narrative\s*$", re.IGNORECASE),
    re.compile(r"^\s*\w+\s+narrative\s*$", re.IGNORECASE),
]

RISK_TAXONOMY_KEYWORDS = {
    "Regulatory / Compliance": [
        "regulator", "regulatory", "compliance", "breach", "fine", "investigation",
        "misconduct claim", "authority", "sanction", "lawsuit",
    ],
    "Financial Integrity": [
        "fraud", "money laundering", "market manipulation", "mis-selling", "embezzlement",
        "ponzi", "insider trading",
    ],
    "Customer Harm": [
        "poor treatment", "unfair", "discrimination", "complaint", "customer harm",
        "mistreatment", "abuse", "harm claim",
    ],
    "Data / Cyber": [
        "breach claim", "breach", "leak", "ransomware", "insecure", "data theft", "phishing",
    ],
    "Operational Resilience": [
        "outage", "service failure", "disruption", "downtime", "incident", "degraded service",
        "systemic disruption",
    ],
    "Executive / Employee Misconduct": [
        "leadership scandal", "harassment", "unethical", "executive", "employee misconduct",
        "workplace", "toxic culture",
    ],
    "Misinformation / Manipulation": [
        "misinformation", "disinformation", "coordinated", "bot", "astroturf", "manipulation",
        "influence campaign",
    ],
}

RISK_TAXONOMY_WEIGHTS = {
    "Regulatory / Compliance": 9.0,
    "Financial Integrity": 4.0,
    "Customer Harm": 10.0,
    "Data / Cyber": 5.0,
    "Operational Resilience": 7.0,
    "Executive / Employee Misconduct": 9.0,
    "Misinformation / Manipulation": 5.0,
}

RISK_TAXONOMY_LABELS = list(RISK_TAXONOMY_KEYWORDS.keys())


def _safe_numeric(series_like):
    return pd.to_numeric(series_like, errors="coerce").fillna(0.0)


def _compact(text, max_len=180):
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _normalize_summary_text(text, max_len=1200):
    cleaned = " ".join(str(text).split()).strip()
    cleaned = _strip_quotes(cleaned)
    if not cleaned:
        return ""
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def _clean_json_text(raw_text):
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object_text(raw_text):
    text = str(raw_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _extract_response_text(response):
    text = getattr(response, "output_text", "") or ""
    if text.strip():
        return text
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return ""
    parts = []
    for item in output_items:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for chunk in content:
            chunk_type = getattr(chunk, "type", "")
            if chunk_type in {"output_text", "text"}:
                value = getattr(chunk, "text", "") or ""
                if value:
                    parts.append(str(value))
    return "\n".join(parts).strip()


def _strip_quotes(text):
    return (
        str(text)
        .replace('"', "")
        .replace("'", "")
        .replace("“", "")
        .replace("”", "")
        .replace("‘", "")
        .replace("’", "")
    ).strip()


def _has_quote_chars(text):
    quote_chars = ['"', "'", "“", "”", "‘", "’"]
    return any(ch in str(text) for ch in quote_chars)


def _is_generic_title(title):
    normalized = " ".join(str(title).split()).strip()
    if not normalized:
        return True
    for pattern in GENERIC_TITLE_PATTERNS:
        if pattern.match(normalized):
            return True
    return False


def _taxonomy_language_contributions(texts, post_count):
    corpus = " ".join(str(t).lower() for t in texts if str(t).strip())
    if not corpus:
        empty = {k: 0.0 for k in RISK_TAXONOMY_KEYWORDS}
        return empty, []

    scale_denom = np.log1p(max(3, int(post_count) * 2))
    contributions = {}
    hits_by_label = {}
    for label, keywords in RISK_TAXONOMY_KEYWORDS.items():
        hits = sum(corpus.count(term) for term in keywords)
        hits_by_label[label] = int(hits)
        normalized = min(1.0, float(np.log1p(max(0, hits))) / scale_denom) if scale_denom > 0 else 0.0
        contributions[label] = round(float(RISK_TAXONOMY_WEIGHTS.get(label, 0.0) * normalized), 2)

    ranked_labels = [
        label for label, _ in sorted(hits_by_label.items(), key=lambda item: item[1], reverse=True) if hits_by_label[label] > 0
    ]
    return contributions, ranked_labels[:3]


def _taxonomy_label_contributions(taxonomy_labels):
    contributions = {k: 0.0 for k in RISK_TAXONOMY_KEYWORDS}
    if not isinstance(taxonomy_labels, list):
        return contributions

    # Primary taxonomy label gets full weight; additional labels are discounted.
    rank_multipliers = [1.0, 0.75, 0.5]
    for idx, raw_label in enumerate(taxonomy_labels[:3]):
        label = str(raw_label).strip()
        if label not in contributions:
            continue
        base_weight = float(RISK_TAXONOMY_WEIGHTS.get(label, 0.0))
        multiplier = rank_multipliers[idx] if idx < len(rank_multipliers) else rank_multipliers[-1]
        contributions[label] = round(base_weight * multiplier, 2)
    return contributions


def _risk_score_confidence_from_matching(group):
    if "resolution_confidence" not in group.columns:
        return "Low"
    conf = _safe_numeric(group["resolution_confidence"])
    if len(conf) == 0:
        return "Low"
    mean_conf = float(conf.mean())
    post_count = max(1, int(len(group)))

    # Add a bounded volume signal so larger evidence sets modestly increase confidence.
    volume_signal = float(min(1.0, np.log1p(post_count) / np.log1p(12)))
    combined_conf = (0.5 * mean_conf) + (0.5 * volume_signal)

    if combined_conf >= 0.75:
        return "High"
    if combined_conf >= 0.5:
        return "Medium"
    return "Low"


def _build_trend(group, points=12):
    if "created_at" not in group.columns:
        return np.linspace(0, max(1, len(group)), points).tolist()

    ts = pd.to_datetime(group["created_at"], errors="coerce").dropna()
    if ts.empty:
        return np.linspace(0, max(1, len(group)), points).tolist()

    daily = ts.dt.normalize().value_counts().sort_index()
    if len(daily) == 1:
        base = float(daily.iloc[0])
        return [base] * points

    x_old = np.linspace(0, 1, len(daily))
    x_new = np.linspace(0, 1, points)
    y = np.interp(x_new, x_old, daily.values.astype(float))
    return [round(float(v), 2) for v in y.tolist()]


def _build_llm_input(group):
    # Prefer per-post risk summaries (already entity-scoped and concise), and include all posts.
    source_col = None
    if "risk_summary" in group.columns:
        source_col = "risk_summary"
    elif "cluster_context" in group.columns:
        source_col = "cluster_context"
    text_col = "text_altered" if "text_altered" in group.columns else "text"
    if source_col is None:
        source_col = text_col
    if source_col not in group.columns:
        return []
    texts = group[source_col].fillna("").astype(str).tolist()
    compact = []
    seen = set()
    for t in texts:
        t2 = _compact(t, max_len=260).strip()
        if t2 and t2 not in seen:
            compact.append(t2)
            seen.add(t2)
    return compact


def _llm_title_summary(group, openai_api_key=None, model="gpt-4.1-nano"):
    key = str(openai_api_key or "").strip()
    if not key:
        raise RuntimeError("OpenAI API key missing for narrative title/summary generation")
    if OpenAI is None:
        raise RuntimeError("OpenAI package unavailable for narrative title/summary generation")

    snippets = _build_llm_input(group)
    if not snippets:
        raise RuntimeError("No LLM input snippets available for narrative title/summary generation")

    entity_names = []
    if "entity" in group.columns:
        entity_names = sorted(
            {
                str(v).strip()
                for v in group["entity"].dropna().tolist()
                if str(v).strip()
            }
        )
    entity_context = ", ".join(entity_names) if entity_names else "Unknown entity context"

    cache_key = (model, tuple(snippets))
    if cache_key in _LLM_SUMMARY_CACHE:
        return _LLM_SUMMARY_CACHE[cache_key]

    snippets_block = "\n".join([f"- post_{idx + 1}: {s}" for idx, s in enumerate(snippets)])
    prompt = (
        "You are labeling ONE entity-scoped risk narrative cluster from social posts.\n"
        "All items below belong to the SAME narrative and SAME entity context.\n"
        "Task: synthesize the shared/common risk narrative across ALL provided posts.\n"
        "Return STRICT JSON with keys: title, summary.\n"
        'Schema: {"title":"<text>","summary":"<text>"}\n\n'
        "Output requirements:\n"
        "- treat risk as including signals that are potentially inflammatory with regard to the entity in question.\n"
        "- title: 5-12 words, specific to this entity-linked risk pattern.\n"
        "- title must not be generic and must not include any quotation marks or apostrophes.\n"
        "- summary: 2-4 sentences, 70-130 words.\n"
        "- summary must describe the common risk mechanism and recurring allegations/signals across the cluster.\n"
        "- summary must be explicitly relative to the entity context.\n"
        "- no quotes, no usernames, no copied phrases, no direct post excerpts.\n"
        "- output valid minified JSON only; no markdown code fences and no prose.\n"
        "- neutral analyst tone; concrete and specific, not vague.\n\n"
        f"Entity context: {entity_context}\n"
        f"Total posts in this narrative cluster: {len(group)}\n"
        f"Post-level risk signals (covering all posts):\n{snippets_block}"
    )

    client = OpenAI(api_key=key)
    import json
    attempts = 3
    last_error = None
    last_preview = ""
    for _attempt in range(1, attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=220,
            )
            raw = _compact(_extract_response_text(response), max_len=5000)
            if not raw.strip():
                raise RuntimeError("Narrative title/summary response was empty")

            cleaned = _clean_json_text(raw)
            if not cleaned:
                raise RuntimeError("Narrative title/summary response cleaned to empty text")
            data = json.loads(cleaned)

            title = _strip_quotes(str(data.get("title", "")).strip())
            summary = _normalize_summary_text(str(data.get("summary", "")).strip(), max_len=1200)
            if not title or not summary:
                raise RuntimeError("Narrative title/summary missing title or summary")
            if _is_generic_title(title) or _has_quote_chars(title + summary):
                raise RuntimeError("Narrative title/summary failed quality validation")
            _LLM_SUMMARY_CACHE[cache_key] = (title, summary)
            return title, summary
        except Exception as exc:
            last_error = exc
            last_preview = _clean_json_text(locals().get("raw", ""))[:700]
            continue
    raise RuntimeError(
        f"Narrative title/summary validation failed after {attempts} attempts: {last_error} | raw_preview={last_preview}"
    ) from last_error


def _llm_taxonomy_labels(group, openai_api_key=None, model="gpt-4.1-nano"):
    key = str(openai_api_key or "").strip()
    if not key:
        raise RuntimeError("OpenAI API key missing for taxonomy classification")
    if OpenAI is None:
        raise RuntimeError("OpenAI package unavailable for taxonomy classification")

    snippets = _build_llm_input(group)
    if not snippets:
        raise RuntimeError("No LLM input snippets available for taxonomy classification")

    cache_key = (model, "taxonomy", tuple(snippets))
    if cache_key in _LLM_TAXONOMY_CACHE:
        return _LLM_TAXONOMY_CACHE[cache_key]

    entity_names = []
    if "entity" in group.columns:
        entity_names = sorted(
            {
                str(v).strip()
                for v in group["entity"].dropna().tolist()
                if str(v).strip()
            }
        )
    entity_context = ", ".join(entity_names) if entity_names else "Unknown entity context"
    snippets_block = "\n".join([f"- post_{idx + 1}: {s}" for idx, s in enumerate(snippets)])
    allowed = ", ".join(RISK_TAXONOMY_LABELS)
    prompt = (
        "You are labeling ONE entity-scoped risk narrative cluster from social posts.\n"
        "Assign one or more taxonomy labels that best fit the shared risk narrative.\n"
        "Return STRICT JSON only with key: labels.\n"
        'Schema: {"labels":["<label1>","<label2>", ...]}\n'
        "Rules:\n"
        f"- labels must be selected ONLY from: {allowed}\n"
        "- return 1 to 3 labels ordered by relevance\n"
        "- include multiple labels when distinct risk types are present\n"
        "- no extra keys, no markdown, no prose\n\n"
        f"Entity context: {entity_context}\n"
        f"Post-level risk signals:\n{snippets_block}"
    )
    client = OpenAI(api_key=key)
    import json
    attempts = 3
    last_error = None
    last_preview = ""
    current_prompt = prompt
    for _attempt in range(1, attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                input=current_prompt,
                max_output_tokens=120,
            )
            raw = _extract_response_text(response)
            if not raw.strip():
                raise RuntimeError("Taxonomy label response was empty")

            cleaned_text = _clean_json_text(raw)
            if not cleaned_text:
                raise RuntimeError("Taxonomy label response cleaned to empty text")
            try:
                parsed = json.loads(cleaned_text)
            except Exception:
                extracted = _extract_json_object_text(cleaned_text)
                parsed = json.loads(extracted)

            labels = parsed.get("labels", [])
            if not isinstance(labels, list):
                raise RuntimeError("Taxonomy label response missing labels list")
            cleaned = []
            seen = set()
            for label in labels:
                name = str(label).strip()
                if name in RISK_TAXONOMY_LABELS and name not in seen:
                    cleaned.append(name)
                    seen.add(name)
            if not cleaned:
                raise RuntimeError("Taxonomy label response contained no allowed labels")
            _LLM_TAXONOMY_CACHE[cache_key] = cleaned[:3]
            return cleaned[:3]
        except Exception as exc:
            last_error = exc
            last_preview = _clean_json_text(locals().get("raw", ""))[:700]
            current_prompt = (
                prompt
                + "\n\nValidation failed for your previous response. "
                "Retry and return ONLY strict JSON with labels from the allowed list exactly as written. "
                "No abbreviations, no synonyms, no extra text."
            )
            continue
    raise RuntimeError(
        f"Taxonomy label validation failed after {attempts} attempts: {last_error} | raw_preview={last_preview}"
    ) from last_error


def score_narratives(posts, authors, openai_api_key=None, llm_model="gpt-4.1-nano", progress_callback=None):
    if posts is None or len(posts) == 0:
        return []
    if "narrative_id" not in posts.columns:
        return []

    narratives = []
    if "entity_id" in posts.columns:
        grouped_items = list(posts.groupby(["entity_id", "narrative_id"], dropna=False))
    else:
        grouped_items = list(posts.groupby("narrative_id", dropna=False))

    total_groups = len(grouped_items)

    for idx, (group_key, group) in enumerate(grouped_items, start=1):
        group = group.copy().reset_index(drop=True)
        if isinstance(group_key, tuple):
            entity_id = str(group_key[0]) if group_key[0] is not None else ""
            nid_raw = group_key[1]
        else:
            entity_id = (
                str(group["entity_id"].iloc[0]).strip()
                if "entity_id" in group.columns and len(group)
                else ""
            )
            nid_raw = group_key

        try:
            narrative_id_value = int(nid_raw)
        except (TypeError, ValueError):
            narrative_id_value = 0

        entity_name = (
            str(group["entity"].iloc[0]).strip()
            if "entity" in group.columns and len(group)
            else ""
        )
        volume = int(len(group))
        likes = _safe_numeric(group["likes"]) if "likes" in group.columns else 0.0
        shares = _safe_numeric(group["shares"]) if "shares" in group.columns else 0.0
        comments = _safe_numeric(group["comments"]) if "comments" in group.columns else 0.0
        views = _safe_numeric(group["views"]) if "views" in group.columns else 0.0
        engagement = float((likes + shares + comments + views).sum())
        if "followers" in group.columns:
            followers = _safe_numeric(group["followers"])
        elif "author_influence" in group.columns:
            followers = _safe_numeric(group["author_influence"])
        elif (
            authors is not None
            and len(authors)
            and "author_id" in group.columns
            and "author_id" in authors.columns
            and "followers" in authors.columns
        ):
            follower_lookup = (
                authors[["author_id", "followers"]]
                .copy()
                .assign(author_id=lambda d: d["author_id"].astype(str).str.strip())
                .drop_duplicates(subset=["author_id"], keep="first")
                .set_index("author_id")["followers"]
            )
            followers = _safe_numeric(group["author_id"].astype(str).str.strip().map(follower_lookup))
        else:
            followers = pd.Series([0.0] * len(group), index=group.index)

        author_influence = float(np.log1p(max(0.0, float(followers.sum()))))
        text_values = (
            group["text"].fillna("").astype(str).tolist()
            if "text" in group.columns
            else []
        )
        base_components = {
            "Volume pressure": round(volume * 5.0, 2),
            "Engagement pressure": round(engagement * 0.015, 2),
            "Author influence pressure": round(min(20.0, author_influence * 2.5), 2),
        }
        risk_score_confidence = _risk_score_confidence_from_matching(group)

        # Fast path on app reloads: reuse seeded metadata from preprocessing artifacts.
        seeded_title = str(group["narrative_title"].iloc[0]).strip() if "narrative_title" in group.columns else ""
        seeded_summary = str(group["narrative_summary"].iloc[0]).strip() if "narrative_summary" in group.columns else ""
        use_seeded_title = bool(seeded_title)
        # Ignore legacy/truncated seeded titles so frontend can show full text.
        if seeded_title.endswith("...") and len(seeded_title) <= 90:
            use_seeded_title = False
        if use_seeded_title:
            title = _strip_quotes(seeded_title)
            summary = _normalize_summary_text(seeded_summary, max_len=1200) if seeded_summary else "No narrative summary available."
        else:
            title, summary = _llm_title_summary(
                group,
                openai_api_key=openai_api_key,
                model=llm_model,
            )
        seeded_taxonomy_labels = None
        if "narrative_taxonomy_labels" in group.columns and len(group):
            raw_labels = group["narrative_taxonomy_labels"].iloc[0]
            if isinstance(raw_labels, list):
                cleaned = [str(v).strip() for v in raw_labels if str(v).strip()]
                if cleaned:
                    seeded_taxonomy_labels = cleaned[:3]
        if seeded_taxonomy_labels:
            taxonomy_labels = seeded_taxonomy_labels
        else:
            taxonomy_labels = _llm_taxonomy_labels(
                group,
                openai_api_key=openai_api_key,
                model=llm_model,
            )
        if not taxonomy_labels:
            raise RuntimeError("No taxonomy labels available for narrative")
        taxonomy_components = _taxonomy_label_contributions(taxonomy_labels)
        raw_score = sum(base_components.values()) + sum(taxonomy_components.values())
        score = round(float(min(100.0, max(0.0, raw_score))), 1)

        narratives.append(
            {
                "narrative_id": narrative_id_value,
                "entity_id": entity_id,
                "entity": entity_name,
                "title": title,
                "summary": summary,
                "score": score,
                "risk_score_confidence": risk_score_confidence,
                "volume": volume,
                "engagement": int(round(engagement)),
                "trend": _build_trend(group),
                "taxonomy_labels": taxonomy_labels,
                "score_components": {
                    **base_components,
                    **{f"Language signal: {label}": value for label, value in taxonomy_components.items()},
                },
                "posts": group,
                "post_ids": (
                    group["post_id"].astype(str).dropna().tolist()
                    if "post_id" in group.columns
                    else []
                ),
            }
        )
        if callable(progress_callback):
            progress_callback(
                {
                    "current": idx,
                    "total": total_groups,
                    "narrative_id": narrative_id_value,
                    "entity": entity_name or entity_id or "-",
                }
            )

    narratives.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return narratives
