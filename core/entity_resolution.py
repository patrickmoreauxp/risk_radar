import re
from difflib import SequenceMatcher

import pandas as pd


ADDITIONAL_ALIASES = {
    "astrazeneca": [
        "AZN", "$AZN", "#AZN",
        "Astra Zeneca",
        "Astrazenica", "Astra Zenica",
    ],
    "pfizer": [
        "PFE", "$PFE", "#PFE",
        "Phizer", "Pizer",
    ],
    "eliquis": [],
    "johnson_johnson": [
        "J&J", "JNJ", "$JNJ", "#JNJ",
        "Johnson and Johnson",
        "J and J",
    ],
    "merck": [
        "MRK", "$MRK", "#MRK",
        "Merck & Co", "Merk",
    ],
    "keytruda": [],
    "bristol_myers_squibb": [
        "BMS", "$BMS", "#BMS",
        "Bristol Myers",
        "Bristol-Myers",
        "Bristol Meyers Squibb",
        "Bristol Myers Squib",
    ],
    "moderna": [
        "MRNA", "$MRNA", "#MRNA",
        "Modern a",
    ],
    "revlimid": [],
    "gardasil": [
        "Gardasil 9",
        "Gardasil9",
        "Gardasil-9",
        "Gardisil",
    ],
    "eli_lilly": [
        "EliLilly",
        "Eli Liliy",
        "Lilly",
    ],
    "gsk": [
        "GlaxoSmithKline",
        "Glaxo Smith Kline",
        "Glaxo",
    ],
    "amgen": [
        "AMGN", "$AMGN", "#AMGN",
    ],
    "opdivo": [],
    "hpv": [
        "Human Papillomavirus",
        "Human Papilloma Virus",
    ],
    "novo_nordisk": [
        "NVO", "$NVO", "#NVO",
        "NovoNordisk",
        "Novo Nordik",
    ],
    "xarelto": [
        "Xeralto",
    ],
    "biontech": [
        "BNTX", "$BNTX", "#BNTX",
        "Bio Tech",
    ],
    "cidara_therapeutics": [
        "Cidara",
        "CDTX", "$CDTX", "#CDTX",
        "Cidarra",
    ],
}


TOKEN_RE = re.compile(r"[A-Za-z0-9#&$@\-]+")


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _split_aliases(raw_aliases):
    if pd.isna(raw_aliases):
        return []
    raw = str(raw_aliases).strip()
    if not raw:
        return []
    return [alias.strip() for alias in raw.split("|") if alias.strip()]


def _mention_from_text(text, term):
    if not term:
        return None
    pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
    match = pattern.search(str(text))
    if not match:
        return None
    return str(text)[match.start(): match.end()]


def _fuzzy_match(text, candidate_terms):
    text_tokens = TOKEN_RE.findall(_normalize_text(text))
    if not text_tokens:
        return None
    best = None
    for term in candidate_terms:
        norm_term = _normalize_text(term)
        if not norm_term:
            continue
        term_tokens = TOKEN_RE.findall(norm_term)
        if not term_tokens:
            continue
        window = len(term_tokens)
        if window == 0 or window > len(text_tokens):
            continue
        for start in range(0, len(text_tokens) - window + 1):
            window_text = " ".join(text_tokens[start: start + window])
            ratio = SequenceMatcher(None, norm_term, window_text).ratio()
            if ratio >= 0.9 and (best is None or ratio > best["ratio"]):
                best = {"ratio": ratio, "mention": window_text}
    return best


def _build_entity_index(entities_df):
    entities = entities_df.copy()
    if "search_list" not in entities.columns:
        entities["search_list"] = [[] for _ in range(len(entities))]

    rows = []
    for _, row in entities.iterrows():
        entity_id = str(row.get("entity_id", "")).strip()
        canonical_name = str(row.get("canonical_name", "")).strip()
        base_aliases = _split_aliases(row.get("aliases", ""))
        extra_aliases = ADDITIONAL_ALIASES.get(entity_id, [])
        search_list = set()
        search_list.update(base_aliases)
        search_list.update(extra_aliases)
        row_search_list = row.get("search_list", [])
        if isinstance(row_search_list, list):
            search_list.update(str(a).strip() for a in row_search_list if str(a).strip())

        rows.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "base_aliases": [a for a in base_aliases if a],
                "additional_aliases": [a for a in extra_aliases if a],
                "search_list": sorted(a for a in search_list if a),
            }
        )
    return pd.DataFrame(rows)


def resolve_entities(posts, entities, progress_callback=None):
    resolved = posts.copy()
    entity_index = _build_entity_index(entities)
    id_to_name = {
        str(row["entity_id"]): str(row["canonical_name"])
        for _, row in entity_index.iterrows()
    }

    resolved_entities_col = []
    resolution_method_col = []
    primary_entity_col = []
    primary_conf_col = []

    total_posts = int(len(resolved))
    for idx, (_, row) in enumerate(resolved.iterrows(), start=1):
        text = str(row.get("text", "") or "")
        text_norm = _normalize_text(text)
        matches = []
        seen = set()
        found_exact_any = False

        for _, ent in entity_index.iterrows():
            entity_id = ent["entity_id"]
            canonical_name = ent["canonical_name"]

            # 1) Direct canonical/entity-id match.
            direct_candidates = [canonical_name, entity_id]
            for candidate in direct_candidates:
                cand_norm = _normalize_text(candidate)
                if not cand_norm:
                    continue
                if cand_norm in text_norm:
                    mention = _mention_from_text(text, candidate) or candidate
                    sig = (entity_id, mention.lower())
                    if sig not in seen:
                        matches.append(
                            {
                                "entity_id": entity_id,
                                "mention_text": mention,
                                "confidence": 0.97,
                                "method": "exact",
                            }
                        )
                        seen.add(sig)
                        found_exact_any = True

            # 2) Seed aliases from entity list.
            for alias in ent["base_aliases"]:
                alias_norm = _normalize_text(alias)
                if not alias_norm:
                    continue
                if alias_norm in text_norm:
                    mention = _mention_from_text(text, alias) or alias
                    sig = (entity_id, mention.lower())
                    if sig not in seen:
                        matches.append(
                            {
                                "entity_id": entity_id,
                                "mention_text": mention,
                                "confidence": 0.92,
                                "method": "exact alias",
                            }
                        )
                        seen.add(sig)
                        found_exact_any = True

            # 3) Additional aliases provided by user.
            for alias in ent["additional_aliases"]:
                alias_norm = _normalize_text(alias)
                if not alias_norm:
                    continue
                if alias_norm in text_norm:
                    mention = _mention_from_text(text, alias) or alias
                    sig = (entity_id, mention.lower())
                    if sig not in seen:
                        matches.append(
                            {
                                "entity_id": entity_id,
                                "mention_text": mention,
                                "confidence": 0.88,
                                "method": "exact alias",
                            }
                        )
                        seen.add(sig)
                        found_exact_any = True

        # 4) Fuzzy fallback (only when no exact/alias hits were found).
        if not found_exact_any:
            for _, ent in entity_index.iterrows():
                entity_id = ent["entity_id"]
                canonical_name = ent["canonical_name"]
                fuzzy = _fuzzy_match(text, [canonical_name] + ent["search_list"])
                if fuzzy is not None:
                    sig = (entity_id, str(fuzzy["mention"]).lower())
                    if sig not in seen:
                        matches.append(
                            {
                                "entity_id": entity_id,
                                "mention_text": str(fuzzy["mention"]),
                                "confidence": round(min(0.85, max(0.75, fuzzy["ratio"] * 0.9)), 2),
                                "method": "fuzzy",
                            }
                        )
                        seen.add(sig)

        if matches:
            matches = sorted(matches, key=lambda x: float(x["confidence"]), reverse=True)
            methods = sorted({m["method"] for m in matches})
            top = matches[0]
            primary_entity = id_to_name.get(top["entity_id"])
            primary_conf = float(top["confidence"])
            clean_matches = [
                {
                    "entity_id": m["entity_id"],
                    "mention_text": m["mention_text"],
                    "confidence": float(m["confidence"]),
                }
                for m in matches
            ]
            resolved_entities_col.append(clean_matches)
            resolution_method_col.append(", ".join(methods))
            primary_entity_col.append(primary_entity)
            primary_conf_col.append(primary_conf)
        else:
            resolved_entities_col.append([])
            resolution_method_col.append("none")
            primary_entity_col.append(None)
            primary_conf_col.append(0.0)

        if callable(progress_callback):
            methods = "none"
            if matches:
                methods = ", ".join(sorted({m["method"] for m in matches}))
            progress_callback(
                {
                    "current": idx,
                    "total": total_posts,
                    "post_id": str(row.get("post_id", idx)),
                    "matched_entities": int(len(matches)),
                    "methods": methods,
                }
            )

    resolved["resolved_entities"] = resolved_entities_col
    resolved["resolution_method"] = resolution_method_col
    resolved["entity"] = primary_entity_col
    resolved["resolution_confidence"] = primary_conf_col

    return resolved
