import streamlit as st
import pandas as pd
import os
import json
import time
from html import escape
from dotenv import load_dotenv
from services.data_loader import load_data
from core.scoring import score_narratives
from services.pipeline import build_preprocessed_pipeline
from frontend.narrative_list import render_narrative_list
from frontend.narrative_detail import (
    render_narrative_detail,
    build_narrative_sentiment_summary,
    build_narrative_quotes,
    build_narrative_pill_groups,
)

st.set_page_config(
    page_title="Risk Radar",
    page_icon="assets/favicon-red-dot.svg",
    layout="wide",
)

load_dotenv()

_run_perf_logs = []
_run_started_at = time.perf_counter()


def _perf_log(label, started_at):
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    _run_perf_logs.append(f"{label}: {elapsed_ms:.1f} ms")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    html, body, [class*="css"] {
        color: #0B1733;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #0B1733;
    }
    [data-testid="stHeader"] {
        background: #FFFFFF;
    }
    [data-testid="stToolbar"] {
        background: #FFFFFF;
    }
    .stMainBlockContainer {
        padding-top: 2rem !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.05rem !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0B1733;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.45rem;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        color: #0B1733;
        border-color: #D1D5DB;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #0B1733 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #0B1733 !important;
        -webkit-text-fill-color: #0B1733 !important;
    }
    [data-testid="stDateInput"] label,
    [data-testid="stDateInput"] label p {
        font-weight: 700 !important;
    }
    [data-testid="stButton"] {
        margin-bottom: 0 !important;
    }
    [data-testid="stButton"] > button {
        margin-bottom: 0 !important;
    }
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="columnheader"] * {
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] canvas:hover,
    [data-testid="stDataFrame"] [role="grid"]:hover,
    [data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"] *:hover {
        cursor: pointer !important;
    }
    .sidebar-brand {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        line-height: 1.1;
        margin: 0 0 0.45rem 0;
        text-transform: lowercase;
    }
    .sidebar-brand-dot {
        color: #DC2626;
    }
    .sidebar-separator {
        width: 100%;
        border-top: 1px solid #334155;
        margin: 0.35rem 0 0.85rem 0;
    }
    .st-key-rebuild_pipeline_btn button {
        background-color: #DC2626 !important;
        border: 1px solid #DC2626 !important;
        color: #FFFFFF !important;
    }
    .st-key-rebuild_pipeline_btn button:hover {
        background-color: #B91C1C !important;
        border-color: #B91C1C !important;
        color: #FFFFFF !important;
    }
    .entity-title {
        margin: 10px 0 0.55rem 0 !important;
        padding: 0 !important;
        color: #0B1733;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
    }
    .entity-title-highlight {
        color: #DC2626;
        text-decoration: underline;
        text-underline-offset: 3px;
        font-size: inherit;
        font-weight: inherit;
        line-height: inherit;
    }
    .detail-back-wrap {
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .narrative-subtitle {
        margin: 0.25rem 0 0.65rem 0;
        color: #0B1733;
        font-size: 1.2rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .ai-summary-tile {
        margin-top: 14px;
        margin-bottom: 4px;
        border: 1px solid #DDD6FE;
        border-radius: 8px;
        background: #F5F3FF;
        padding: 10px 12px;
    }
    .ai-summary-label {
        color: #5B21B6;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .ai-summary-text {
        color: #1E293B;
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .st-key-detail_mark_false_positive_btn button {
        background-color: #F87171 !important;
        border-color: #F87171 !important;
        color: #FFFFFF !important;
    }
    .st-key-detail_mark_false_positive_btn button:hover {
        background-color: #EF4444 !important;
        border-color: #EF4444 !important;
        color: #FFFFFF !important;
    }
    .quotes-tile {
        margin-top: 10px;
        margin-bottom: 4px;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        background: #EFF6FF;
        padding: 10px 12px;
    }
    .quotes-label {
        color: #1D4ED8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .quotes-line {
        margin-top: 6px;
        padding-left: 8px;
        border-left: 2px solid #93C5FD;
        color: #1E3A8A;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .detail-summary-tile {
        margin-top: 14px;
        margin-bottom: 4px;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        background: #F8FAFC;
        padding: 10px 12px 30px 12px;
    }
    .detail-summary-label {
        color: #0F172A;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .detail-summary-row {
        margin-top: 6px;
        font-size: 0.9rem;
        line-height: 1.35;
        color: #1E293B;
    }
    .detail-summary-row b {
        color: #0F172A;
    }
    .detail-pill-wrap {
        display: inline;
        margin-top: 0;
    }
    .detail-pill {
        display: inline-block;
        background: #E2E8F0;
        color: #0F172A;
        border-radius: 999px;
        padding: 0.16rem 0.5rem;
        margin: 0 0.3rem 0.28rem 0;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _run_preprocessed_pipeline_with_progress(openai_api_key=None):
    posts, authors, entities = load_data()
    pipeline_steps = [
        ("load_raw", "Load raw posts/authors/entities"),
        ("preprocess", "Prefilter long + non-negative sentiment posts"),
        ("entity_resolution", "Resolve entities per post"),
        ("expand_entities", "Expand to post-entity rows"),
        ("risk_enrichment", "LLM risk extraction + summaries + embeddings"),
        ("clustering", "Cluster risk summaries into narratives"),
        ("scoring", "Score narratives + generate titles/summaries"),
        ("save_artifacts", "Save processed artifacts"),
    ]
    step_order = {name: i for i, (name, _) in enumerate(pipeline_steps)}
    step_labels = {name: label for name, label in pipeline_steps}
    completed_steps = set()
    current_step = "load_raw"
    step_started_at = {}

    def _render_pipeline_timeline():
        done_count = len(completed_steps)
        total_count = len(pipeline_steps)
        timeline_lines = []
        for idx, (step_name, label) in enumerate(pipeline_steps, start=1):
            if step_name in completed_steps:
                marker = "✅"
            elif step_name == current_step:
                marker = "🔄"
            else:
                marker = "⬜"
            timeline_lines.append(f"{marker} {idx}/{total_count} {label}")
        return "\n".join(timeline_lines)

    def _step_elapsed_text(step_name):
        started = step_started_at.get(step_name)
        if started is None:
            return ""
        return f" | {time.perf_counter() - started:.1f}s"

    with st.status("Preparing data pipeline...", expanded=True) as pipeline_status:
        step_progress = st.progress(0, text="Waiting to start")
        timeline_placeholder = st.empty()
        timeline_placeholder.code(_render_pipeline_timeline())

        step_progress.progress(100, text=f"{step_labels['load_raw']}: complete")
        completed_steps.add("load_raw")
        current_step = None
        timeline_placeholder.code(_render_pipeline_timeline())

        def _stage_callback(stage_name):
            nonlocal current_step
            if stage_name.endswith("_start"):
                step_name = stage_name.replace("_start", "")
                if step_name in step_order:
                    current_step = step_name
                    step_started_at[step_name] = time.perf_counter()
                    step_progress.progress(0, text=f"{step_labels[step_name]}: 0%")
            elif stage_name.endswith("_done"):
                step_name = stage_name.replace("_done", "")
                if step_name in step_order:
                    completed_steps.add(step_name)
                    elapsed = _step_elapsed_text(step_name)
                    step_progress.progress(100, text=f"{step_labels[step_name]}: complete{elapsed}")
                    current_step = None
            timeline_placeholder.code(_render_pipeline_timeline())

        def _set_step_progress(step_name, current, total, suffix=""):
            if current_step != step_name:
                return
            total = max(1, int(total))
            current = min(total, max(0, int(current)))
            pct = min(100, int(round((current / total) * 100)))
            elapsed = _step_elapsed_text(step_name)
            suffix_text = f" | {suffix}" if suffix else ""
            step_progress.progress(
                pct,
                text=f"{step_labels[step_name]}: {pct}% ({current}/{total}){suffix_text}{elapsed}",
            )

        def _entity_progress_callback(payload):
            current = int(payload.get("current", 0))
            total = max(1, int(payload.get("total", 1)))
            _set_step_progress("entity_resolution", current, total)

        def _risk_progress_callback(payload):
            phase = str(payload.get("phase", "row_progress"))
            # Keep risk-enrichment progress monotonic while surfacing sub-stage work.
            if "_risk_last_pct" not in st.session_state:
                st.session_state["_risk_last_pct"] = 0

            def _set_risk_progress(pct, detail=""):
                if current_step != "risk_enrichment":
                    return
                pct = max(int(st.session_state.get("_risk_last_pct", 0)), int(pct))
                pct = max(0, min(100, pct))
                st.session_state["_risk_last_pct"] = pct
                elapsed = _step_elapsed_text("risk_enrichment")
                detail_text = f" | {detail}" if detail else ""
                step_progress.progress(
                    pct,
                    text=f"{step_labels['risk_enrichment']}: {pct}%{detail_text}{elapsed}",
                )

            if phase == "risk_stage_start":
                st.session_state["_risk_last_pct"] = 0
                total_rows = max(1, int(payload.get("total_rows", 1)))
                unique_contexts = max(0, int(payload.get("unique_contexts", 0)))
                llm_total = max(0, int(payload.get("llm_batches_total", 0)))
                _set_risk_progress(
                    1,
                    detail=f"context 0/{total_rows} | llm 0/{llm_total if llm_total else '?'} | unique {unique_contexts}",
                )
                return
            if phase == "context_progress":
                current = int(payload.get("current", 0))
                total = max(1, int(payload.get("total", 1)))
                pct = int(round((current / total) * 20))
                _set_risk_progress(pct, detail=f"context {current}/{total}")
                return
            if phase in {"llm_batch_start", "llm_batch_done"}:
                done = int(payload.get("batch_index", 0))
                total = max(1, int(payload.get("batch_total", 1)))
                if phase == "llm_batch_start":
                    done = max(0, done - 1)
                pct = 20 + int(round((done / total) * 50))
                _set_risk_progress(pct, detail=f"llm batches {done}/{total}")
                return
            if phase in {"embedding_batch_start", "embedding_batch_done"}:
                done = int(payload.get("batch_index", 0))
                total = max(1, int(payload.get("batch_total", 1)))
                if phase == "embedding_batch_start":
                    done = max(0, done - 1)
                pct = 70 + int(round((done / total) * 20))
                _set_risk_progress(pct, detail=f"emb batches {done}/{total}")
                return
            if phase == "row_progress":
                current = int(payload.get("current", 0))
                total = max(1, int(payload.get("total", 1)))
                llm_done = int(payload.get("llm_batches_done", 0))
                llm_total = int(payload.get("llm_batches_total", 0))
                embed_done = int(payload.get("embed_batches_done", 0))
                embed_total = int(payload.get("embed_batches_total", 0))
                pct = 90 + int(round((current / total) * 10))
                suffix = (
                    f"llm {llm_done}/{llm_total if llm_total else '?'} | "
                    f"emb {embed_done}/{embed_total if embed_total else '?'} | rows {current}/{total}"
                )
                _set_risk_progress(pct, detail=suffix)
                return
            if phase == "risk_stage_done":
                kept = int(payload.get("kept", 0))
                dropped = int(payload.get("dropped", 0))
                _set_risk_progress(100, detail=f"kept {kept} | dropped {dropped}")

        def _clustering_progress_callback(payload):
            _set_step_progress(
                "clustering",
                int(payload.get("current", 0)),
                max(1, int(payload.get("total", 1))),
            )

        def _scoring_progress_callback(payload):
            _set_step_progress(
                "scoring",
                int(payload.get("current", 0)),
                max(1, int(payload.get("total", 1))),
            )

        def _save_progress_callback(payload):
            file_name = str(payload.get("file", ""))
            _set_step_progress(
                "save_artifacts",
                int(payload.get("current", 0)),
                max(1, int(payload.get("total", 1))),
                suffix=file_name,
            )

        try:
            artifacts = build_preprocessed_pipeline(
                posts,
                authors,
                entities,
                stage_callback=_stage_callback,
                entity_progress_callback=_entity_progress_callback,
                risk_progress_callback=_risk_progress_callback,
                clustering_progress_callback=_clustering_progress_callback,
                scoring_progress_callback=_scoring_progress_callback,
                save_progress_callback=_save_progress_callback,
                openai_api_key=openai_api_key,
            )
        except TypeError as exc:
            # Defensive fallback for stale hot-reload states where pipeline kwargs are unavailable.
            if "unexpected keyword argument" not in str(exc):
                raise
            artifacts = build_preprocessed_pipeline(
                posts,
                authors,
                entities,
                openai_api_key=openai_api_key,
            )

        completed_steps.update([name for name, _ in pipeline_steps])
        current_step = None
        timeline_placeholder.code(_render_pipeline_timeline())
        step_progress.progress(100, text="Pipeline complete")
        pipeline_status.update(label="Data pipeline ready", state="complete", expanded=False)

    return artifacts

# OpenAI key is loaded from environment only.
active_openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()


@st.cache_data(show_spinner=False)
def _load_pipeline_artifacts_from_disk(_artifact_signature=None):
    processed_dir = "data/processed"
    clustered_path = os.path.join(processed_dir, "clustered_posts.jsonl")
    narratives_path = os.path.join(processed_dir, "narratives.json")
    if not (os.path.exists(clustered_path) and os.path.exists(narratives_path)):
        return None
    if os.path.getsize(clustered_path) == 0 or os.path.getsize(narratives_path) == 0:
        return None

    clustered_posts = pd.read_json(clustered_path, lines=True)
    with open(narratives_path, "r", encoding="utf-8") as handle:
        narratives_meta = json.load(handle)
    # Keep runtime memory lean if legacy files still include embeddings.
    if "risk_embedding" in clustered_posts.columns:
        clustered_posts = clustered_posts.drop(columns=["risk_embedding"])
    if isinstance(narratives_meta, list) and len(narratives_meta):
        taxonomy_map = {}
        for row in narratives_meta:
            try:
                nid = int(row.get("narrative_id", 0))
            except (TypeError, ValueError):
                continue
            labels = row.get("taxonomy_labels", [])
            if isinstance(labels, list):
                taxonomy_map[nid] = [str(v).strip() for v in labels if str(v).strip()]
        if taxonomy_map:
            clustered_posts["narrative_taxonomy_labels"] = clustered_posts["narrative_id"].map(
                lambda nid: taxonomy_map.get(int(nid), [])
            )
    if clustered_posts.empty:
        return None
    authors = pd.read_csv("data/authors.csv")
    entities = pd.read_csv("data/entities_seed.csv")
    return {
        "clustered_posts": clustered_posts,
        "authors": authors,
        "entities": entities,
    }


def _processed_artifacts_exist():
    processed_dir = "data/processed"
    clustered_path = os.path.join(processed_dir, "clustered_posts.jsonl")
    narratives_path = os.path.join(processed_dir, "narratives.json")
    return (
        os.path.exists(clustered_path)
        and os.path.exists(narratives_path)
        and os.path.getsize(clustered_path) > 0
        and os.path.getsize(narratives_path) > 0
    )


def _processed_artifacts_signature():
    processed_dir = "data/processed"
    clustered_path = os.path.join(processed_dir, "clustered_posts.jsonl")
    narratives_path = os.path.join(processed_dir, "narratives.json")
    signature = []
    for path in [clustered_path, narratives_path]:
        if os.path.exists(path):
            stat = os.stat(path)
            signature.append((path, stat.st_size, int(stat.st_mtime)))
        else:
            signature.append((path, 0, 0))
    return tuple(signature)


force_rebuild = bool(st.session_state.get("_force_pipeline_rebuild", False))
if force_rebuild:
    # One-shot behavior: clear immediately so failed runs don't force-loop on refresh.
    st.session_state["_force_pipeline_rebuild"] = False
    st.session_state.pop("_pipeline_artifacts", None)
    st.session_state.pop("_scored_narratives_cache", None)
    _load_pipeline_artifacts_from_disk.clear()

# Load artifacts once. Prefer disk cache; run pipeline only when missing or forced.
_artifact_load_t0 = time.perf_counter()
if "_pipeline_artifacts" not in st.session_state:
    if force_rebuild:
        _run_perf_logs.append("artifacts.rebuild_forced: true")
        _artifact_pipeline_t0 = time.perf_counter()
        st.session_state["_pipeline_artifacts"] = _run_preprocessed_pipeline_with_progress(
            openai_api_key=active_openai_api_key,
        )
        _perf_log("artifacts.build_pipeline", _artifact_pipeline_t0)
    else:
        _artifact_disk_t0 = time.perf_counter()
        artifacts_present = _processed_artifacts_exist()
        disk_artifacts = _load_pipeline_artifacts_from_disk(_processed_artifacts_signature())
        _perf_log("artifacts.read_from_disk", _artifact_disk_t0)
        if disk_artifacts is not None:
            st.session_state["_pipeline_artifacts"] = disk_artifacts
        elif artifacts_present:
            # Do not auto-rebuild when files exist: force explicit user action instead.
            raise RuntimeError(
                "Processed artifacts exist but failed to load. "
                "Please click 'Rebuild data pipeline' to regenerate them."
            )
        else:
            _artifact_pipeline_t0 = time.perf_counter()
            st.session_state["_pipeline_artifacts"] = _run_preprocessed_pipeline_with_progress(
                openai_api_key=active_openai_api_key,
            )
            _perf_log("artifacts.build_pipeline", _artifact_pipeline_t0)
else:
    _run_perf_logs.append("artifacts.session_cache_hit: 0.0 ms")
_perf_log("artifacts.total", _artifact_load_t0)

pipeline_artifacts = st.session_state["_pipeline_artifacts"]
authors = pipeline_artifacts["authors"]
entities = pipeline_artifacts["entities"]
clustered_posts_all = pipeline_artifacts["clustered_posts"]

# Sidebar controls
st.sidebar.markdown(
    '<div class="sidebar-brand">risk radar<span class="sidebar-brand-dot">.</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

st.sidebar.header("Select Entity")
entity_options = ["All"] + entities["canonical_name"].dropna().tolist()
selected_entity = st.sidebar.selectbox(
    "Entity",
    entity_options,
    index=0,
    label_visibility="collapsed",
)

if st.sidebar.button("Rebuild data pipeline", key="rebuild_pipeline_btn"):
    st.session_state["_force_pipeline_rebuild"] = True
    st.rerun()

posts_for_view = clustered_posts_all

if "selected_narrative" not in st.session_state:
    st.session_state.selected_narrative = None

query_narrative = st.query_params.get("narrative")
if query_narrative and st.session_state.selected_narrative is None:
    try:
        st.session_state.selected_narrative = int(query_narrative)
    except ValueError:
        st.session_state.selected_narrative = None

if selected_entity == "All":
    entity_posts = posts_for_view.copy()
else:
    entity_posts = posts_for_view[posts_for_view["entity"] == selected_entity].copy()
_run_perf_logs.append(f"entity_posts.rows: {int(len(entity_posts))}")

# Ensure follower counts are present for detail-table "Author Followers".
_followers_t0 = time.perf_counter()
if (
    isinstance(authors, pd.DataFrame)
    and "author_id" in entity_posts.columns
    and "author_id" in authors.columns
    and "followers" in authors.columns
):
    author_lookup = authors[["author_id", "followers"]].copy()
    author_lookup = author_lookup.dropna(subset=["author_id"])
    author_lookup["author_id"] = author_lookup["author_id"].astype(str)
    author_lookup = author_lookup.drop_duplicates(subset=["author_id"], keep="last")
    follower_map = author_lookup.set_index("author_id")["followers"]
    mapped_followers = entity_posts["author_id"].astype(str).map(follower_map)
    mapped_followers = pd.to_numeric(mapped_followers, errors="coerce")
    if "followers" in entity_posts.columns:
        existing_followers = pd.to_numeric(entity_posts["followers"], errors="coerce")
        entity_posts["followers"] = existing_followers.fillna(mapped_followers).fillna(0.0)
    else:
        entity_posts["followers"] = mapped_followers.fillna(0.0)
_perf_log("followers_enrichment", _followers_t0)

if "_scored_narratives_cache" not in st.session_state:
    st.session_state["_scored_narratives_cache"] = {}

cache_rows = int(len(entity_posts))
cache_post_min = str(entity_posts["post_id"].min()) if cache_rows and "post_id" in entity_posts.columns else ""
cache_post_max = str(entity_posts["post_id"].max()) if cache_rows and "post_id" in entity_posts.columns else ""
cache_key = (selected_entity, cache_rows, cache_post_min, cache_post_max)

if cache_key in st.session_state["_scored_narratives_cache"]:
    _score_t0 = time.perf_counter()
    scored_narratives = st.session_state["_scored_narratives_cache"][cache_key]
    _perf_log("score_narratives.cache_hit", _score_t0)
else:
    _score_t0 = time.perf_counter()
    scored_narratives = score_narratives(
        entity_posts,
        authors,
        openai_api_key=active_openai_api_key,
        llm_model="gpt-4.1-nano",
    )
    _perf_log("score_narratives.compute", _score_t0)
    # Keep cache bounded so it does not grow unbounded with date tweaks.
    if len(st.session_state["_scored_narratives_cache"]) >= 12:
        first_key = next(iter(st.session_state["_scored_narratives_cache"]))
        st.session_state["_scored_narratives_cache"].pop(first_key, None)
    st.session_state["_scored_narratives_cache"][cache_key] = scored_narratives
_run_perf_logs.append(f"narratives.count: {len(scored_narratives)}")

if st.session_state.get("selected_narrative") is not None:
    available_ids = {int(n["narrative_id"]) for n in scored_narratives}
    if int(st.session_state.selected_narrative) not in available_ids:
        st.session_state.selected_narrative = None
        st.session_state.pop("selected_narrative_title", None)
        st.query_params.clear()

if (
    st.session_state.selected_narrative is not None
    and "selected_narrative_title" not in st.session_state
):
    selected = next(
        (n for n in scored_narratives if n["narrative_id"] == st.session_state.selected_narrative),
        None,
    )
    if selected is not None:
        explicit_title = str(selected.get("title", "")).strip()
        if explicit_title:
            st.session_state.selected_narrative_title = explicit_title
        else:
            st.session_state.selected_narrative_title = f"Narrative {st.session_state.selected_narrative}"
    else:
        st.session_state.selected_narrative_title = f"Narrative {st.session_state.selected_narrative}"


def _render_detail_false_positive_modal(entity_choices):
    narrative_id = st.session_state.get("selected_narrative")
    narrative_title = st.session_state.get("selected_narrative_title", "Title of Narrative")
    if narrative_id is None:
        st.session_state["_detail_header_false_positive_open"] = False
        return

    match_options = ["None"] + [str(e).strip() for e in entity_choices if str(e).strip()]

    if hasattr(st, "dialog"):
        @st.dialog("Mark as False Positive")
        def _detail_header_false_positive_dialog():
            st.markdown("Are you sure you want to mark this narrative as a false positive match?")
            st.markdown(f"**Narrative:** {int(narrative_id)} - {escape(str(narrative_title))}")
            st.selectbox(
                "Change Entity Match",
                match_options,
                index=0,
                key=f"detail_header_fp_entity_match_{narrative_id}",
            )
            st.text_area(
                "Reasoning",
                placeholder="Explain why this should be marked as a false positive.",
                key=f"detail_header_fp_reason_{narrative_id}",
                height=100,
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm", key=f"detail_header_fp_confirm_{narrative_id}", use_container_width=True):
                    st.session_state["_detail_header_false_positive_open"] = False
                    if hasattr(st, "toast"):
                        st.toast("Done.")
                    else:
                        st.success("Done.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"detail_header_fp_cancel_{narrative_id}", use_container_width=True):
                    st.session_state["_detail_header_false_positive_open"] = False
                    st.rerun()

        _detail_header_false_positive_dialog()
    else:
        with st.expander("Mark as False Positive", expanded=True):
            st.markdown("Are you sure you want to mark this narrative as a false positive match?")
            st.markdown(f"**Narrative:** {int(narrative_id)} - {escape(str(narrative_title))}")
            st.selectbox(
                "Change Entity Match",
                match_options,
                index=0,
                key=f"detail_header_fp_entity_match_inline_{narrative_id}",
            )
            st.text_area(
                "Reasoning",
                placeholder="Explain why this should be marked as a false positive.",
                key=f"detail_header_fp_reason_inline_{narrative_id}",
                height=100,
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm", key=f"detail_header_fp_confirm_inline_{narrative_id}", use_container_width=True):
                    st.session_state["_detail_header_false_positive_open"] = False
                    st.success("Done.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"detail_header_fp_cancel_inline_{narrative_id}", use_container_width=True):
                    st.session_state["_detail_header_false_positive_open"] = False
                    st.rerun()


# UI routing
entity_title = selected_entity if selected_entity != "All" else "All Entities"
entity_title_safe = escape(entity_title)
is_detail_view = st.session_state.selected_narrative is not None

if is_detail_view:
    back_col, _ = st.columns([2.2, 7.8], vertical_alignment="center")
    with back_col:
        st.markdown('<div class="detail-back-wrap"></div>', unsafe_allow_html=True)
        nav_btn_col, fp_btn_col = st.columns([1, 1])
        with nav_btn_col:
            if st.button("← Back to Dashboard", key="detail_back_btn"):
                st.session_state.selected_narrative = None
                st.session_state.pop("selected_narrative_title", None)
                st.query_params.clear()
                st.rerun()
        with fp_btn_col:
            if st.button("Mark as False Positive", key="detail_mark_false_positive_btn"):
                st.session_state["_show_risk_score_help_modal"] = False
                st.session_state["_show_risk_adjust_modal"] = False
                st.session_state["_detail_inspect_open"] = False
                st.session_state["_detail_false_positive_open"] = False
                st.session_state["_detail_header_false_positive_open"] = True

if st.session_state.get("_detail_header_false_positive_open", False):
    _render_detail_false_positive_modal(entity_options[1:])

if is_detail_view:
    narrative_title = escape(
        st.session_state.get(
            "selected_narrative_title",
            "Title of Narrative",
        )
    )
    st.markdown(
        f'<div class="entity-title">Narrative {st.session_state.selected_narrative}: {narrative_title}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="entity-title">Narrative Risk Dashboard for <span class="entity-title-highlight">{entity_title_safe}</span></div>',
        unsafe_allow_html=True,
    )

if st.session_state.selected_narrative is not None:
    _detail_prep_t0 = time.perf_counter()
    selected_narrative_obj = next(
        (n for n in scored_narratives if n["narrative_id"] == st.session_state.selected_narrative),
        None,
    )
    summary_text = build_narrative_sentiment_summary(
        st.session_state.selected_narrative,
        scored_narratives,
    )
    quotes = build_narrative_quotes(
        st.session_state.selected_narrative,
        scored_narratives,
        limit=3,
    )
    first_seen_at = "-"
    total_posts = 0
    total_engagement = 0
    total_unique_authors = 0
    platforms = "-"
    risk_types = []
    if selected_narrative_obj is not None and "posts" in selected_narrative_obj:
        detail_posts = selected_narrative_obj["posts"]
        if detail_posts is not None and len(detail_posts):
            total_posts = int(len(detail_posts))
            time_col = next(
                (c for c in ["created_at", "timestamp", "datetime", "date", "published_at", "posted_at", "time"] if c in detail_posts.columns),
                None,
            )
            if time_col is not None:
                parsed = pd.to_datetime(detail_posts[time_col], errors="coerce").dropna()
                if len(parsed):
                    first_seen_at = parsed.min().strftime("%Y-%m-%d %H:%M")

            likes = pd.to_numeric(detail_posts["likes"], errors="coerce").fillna(0) if "likes" in detail_posts.columns else 0
            shares = pd.to_numeric(detail_posts["shares"], errors="coerce").fillna(0) if "shares" in detail_posts.columns else 0
            comments = pd.to_numeric(detail_posts["comments"], errors="coerce").fillna(0) if "comments" in detail_posts.columns else 0
            views = pd.to_numeric(detail_posts["views"], errors="coerce").fillna(0) if "views" in detail_posts.columns else 0
            total_engagement = int((likes + shares + comments + views).sum())

            author_col = next((c for c in ["author_id", "author", "username", "user_id"] if c in detail_posts.columns), None)
            if author_col is not None:
                total_unique_authors = int(detail_posts[author_col].astype(str).replace("nan", pd.NA).dropna().nunique())
            else:
                total_unique_authors = total_posts

            if "platform" in detail_posts.columns:
                unique_platforms = sorted(
                    {str(p).strip() for p in detail_posts["platform"].dropna().tolist() if str(p).strip()}
                )
                if unique_platforms:
                    platforms = ", ".join(unique_platforms)
    if selected_narrative_obj is not None:
        try:
            risk_types, _drivers = build_narrative_pill_groups(
                st.session_state.selected_narrative,
                scored_narratives,
            )
        except Exception:
            risk_types = []
    if risk_types:
        risk_types_html = "".join(
            f'<span class="detail-pill">{escape(str(risk_type))}</span>'
            for risk_type in risk_types
        )
    else:
        risk_types_html = '<span class="detail-pill">Operational Resilience</span>'

    platform_pill_palette = [
        ("#DBEAFE", "#1E3A8A"),
        ("#EDE9FE", "#5B21B6"),
        ("#D1FAE5", "#065F46"),
        ("#FCE7F3", "#9D174D"),
        ("#FEF3C7", "#92400E"),
    ]
    platform_values = [p.strip() for p in str(platforms).split(",") if p.strip()] if platforms else []
    if platform_values:
        platform_pills_html = "".join(
            (
                f'<span class="detail-pill" style="background:{platform_pill_palette[idx % len(platform_pill_palette)][0]};'
                f'color:{platform_pill_palette[idx % len(platform_pill_palette)][1]};">'
                f"{escape(platform_name)}</span>"
            )
            for idx, platform_name in enumerate(platform_values)
        )
    else:
        platform_pills_html = '<span class="detail-pill">Unknown</span>'

    main_col, side_col = st.columns([2.1, 1.2], vertical_alignment="top")
    with main_col:
        st.markdown(
            f"""
            <div class="ai-summary-tile">
                <div class="ai-summary-label">Narrative Description</div>
                <div class="ai-summary-text">{escape(summary_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if quotes:
            rendered_quotes = "".join(
                f'<div class="quotes-line">"{escape(q)}"</div>'
                for q in quotes
            )
            quotes_body = rendered_quotes
        else:
            quotes_body = '<div class="ai-summary-text">No matched quotes available.</div>'

        st.markdown(
            f"""
            <div class="quotes-tile">
                <div class="quotes-label">Quotes</div>
                {quotes_body}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with side_col:
        st.markdown(
            f"""
            <div class="detail-summary-tile">
                <div class="detail-summary-label">Summary</div>
                <div class="detail-summary-row"><b>First Seen At:</b> {escape(str(first_seen_at))}</div>
                <div class="detail-summary-row"><b>Risk Types:</b> <span class="detail-pill-wrap">{risk_types_html}</span></div>
                <div class="detail-summary-row"><b>Total Posts:</b> {total_posts:,}</div>
                <div class="detail-summary-row"><b>Total Engagement:</b> {total_engagement:,}</div>
                <div class="detail-summary-row"><b>Total Unique Authors:</b> {total_unique_authors:,}</div>
                <div class="detail-summary-row"><b>Platforms:</b> <span class="detail-pill-wrap">{platform_pills_html}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    _perf_log("detail_header_summary_and_quotes", _detail_prep_t0)

st.markdown("---")

_render_t0 = time.perf_counter()
if st.session_state.selected_narrative is None:
    render_narrative_list(
        scored_narratives,
        entity_posts,
        include_entity_column=(selected_entity == "All"),
    )
else:
    render_narrative_detail(
        st.session_state.selected_narrative,
        scored_narratives,
        entity_posts,
        entity_options=entity_options[1:],
    )
_perf_log("page_render", _render_t0)

_perf_log("rerun_total", _run_started_at)
st.session_state["_last_perf_logs"] = _run_perf_logs[-80:]
