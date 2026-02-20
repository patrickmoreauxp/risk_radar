import streamlit as st
import pandas as pd
import numpy as np
import re


RISK_THEME_KEYWORDS = {
    "Regulatory / Compliance": [
        "regulator", "compliance", "fine", "investigation", "breach", "sanction",
    ],
    "Financial Integrity": [
        "fraud", "money laundering", "market manipulation", "mis-selling", "loss",
    ],
    "Customer Harm": [
        "complaint", "unfair", "discrimination", "harm", "mistreatment",
    ],
    "Data / Cyber": [
        "cyber", "breach", "leak", "ransomware", "phishing",
    ],
    "Operational Resilience": [
        "outage", "service failure", "downtime", "disruption", "incident",
    ],
    "Executive / Employee Misconduct": [
        "scandal", "harassment", "unethical", "misconduct", "leadership",
    ],
    "Misinformation / Manipulation": [
        "misinformation", "disinformation", "bot", "coordinated", "manipulation",
    ],
}


THEME_IMPACT = {
    "Regulatory / Compliance": "higher regulatory scrutiny, legal exposure, and potential penalties.",
    "Financial Integrity": "investor confidence erosion and possible legal or enforcement action.",
    "Customer Harm": "brand trust decline and sustained customer attrition risk.",
    "Data / Cyber": "reputation damage, customer trust loss, and remediation cost escalation.",
    "Operational Resilience": "service reliability concerns that can weaken customer confidence.",
    "Executive / Employee Misconduct": "governance credibility pressure and leadership risk.",
    "Misinformation / Manipulation": "narrative control risk and amplification of reputational damage.",
}

RISK_TYPE_ACRONYMS = {
    "Regulatory / Compliance": "RC",
    "Financial Integrity": "FI",
    "Customer Harm": "CH",
    "Data / Cyber": "DC",
    "Operational Resilience": "OR",
    "Executive / Employee Misconduct": "EEM",
    "Misinformation / Manipulation": "MM",
}


def _shorten(text, max_len=150):
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _sanitize_narrative_description(text):
    clean = re.sub(r"\s+", " ", str(text)).strip()
    # Remove quote-like characters to keep narrative text clean and non-quoted.
    for ch in ['"', "'", "“", "”", "‘", "’"]:
        clean = clean.replace(ch, "")
    return clean


def _build_shap_like_contributors(narrative):
    component_map = narrative.get("score_components", {})
    if isinstance(component_map, dict) and component_map:
        display_name_map = {
            "Volume pressure": "High Volume",
            "Engagement pressure": "High Engagement",
            "Author influence pressure": "Influential Authors",
        }

        def _display_label(raw_label):
            label = str(raw_label).strip()
            if label in display_name_map:
                return display_name_map[label]
            if label.startswith("Language signal:"):
                suffix = label.split(":", 1)[1].strip()
                return f"Language Risk: {suffix}"
            return label

        rows = []
        for feature, value in component_map.items():
            try:
                rows.append(
                    {
                        "Feature": _display_label(feature),
                        "Contribution": float(value),
                    }
                )
            except (TypeError, ValueError):
                continue
        if rows:
            return (
                pd.DataFrame(rows)
                .sort_values("Contribution", ascending=False)
                .reset_index(drop=True)
            )

    # Fallback only when score components are missing in legacy artifacts.
    return pd.DataFrame([{"Feature": "No score components available", "Contribution": 0.0}])


def _risk_tile_palette(score):
    if score >= 80:
        return {"bg": "#FEE2E2", "border": "#FCA5A5", "value": "#991B1B", "label": "#7F1D1D"}
    if score >= 60:
        return {"bg": "#FFEDD5", "border": "#FDBA74", "value": "#9A3412", "label": "#7C2D12"}
    if score >= 40:
        return {"bg": "#FEF9C3", "border": "#FDE68A", "value": "#854D0E", "label": "#713F12"}
    return {"bg": "#DCFCE7", "border": "#86EFAC", "value": "#166534", "label": "#14532D"}


def _risk_score_confidence(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _narrative_risk_score_confidence(narrative):
    explicit = str(narrative.get("risk_score_confidence", "")).strip()
    if explicit:
        return explicit
    return _risk_score_confidence(round(float(narrative.get("score", 0)), 1))


def _risk_score_color(score):
    clamped = max(0.0, min(100.0, float(score)))
    t = clamped / 100.0
    # Bright green (#22C55E) -> bright red (#EF4444)
    start = (34, 197, 94)
    end = (239, 68, 68)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _render_risk_score_help_modal():
    help_text = (
        "The Current Risk Score is a normalized 0-100 indicator of narrative risk. "
        "Lower values indicate weaker risk signals, while higher values indicate stronger risk signals. "
        "The score is driven by the risk drivers shown in the section below."
    )

    if hasattr(st, "dialog"):
        @st.dialog("Current Risk Score - Help")
        def _risk_score_help_dialog():
            st.markdown(help_text)
            st.markdown(
                "- `0-39`: Lower risk confidence\n"
                "- `40-69`: Medium risk confidence\n"
                "- `70-100`: High risk confidence"
            )
            if st.button("Close", key="risk_score_help_close_btn"):
                st.session_state["_show_risk_score_help_modal"] = False
                st.rerun()

        _risk_score_help_dialog()
    else:
        # Fallback for older Streamlit versions without modal dialogs.
        st.info(help_text)
        st.caption("Risk bands: 0-39 (Low), 40-69 (Medium), 70-100 (High).")


def _render_adjust_score_confidence_modal(narrative):
    narrative_id = int(narrative.get("narrative_id", 0))
    default_score = int(round(float(narrative.get("score", 0))))
    default_conf = _narrative_risk_score_confidence(narrative)
    adjustment_store = st.session_state.get("_risk_adjustments", {})
    current = adjustment_store.get(
        narrative_id,
        {"score": default_score, "confidence": default_conf, "reasoning": ""},
    )

    if hasattr(st, "dialog"):
        @st.dialog("Adjust Score or Confidence")
        def _adjust_dialog():
            score_value = st.slider(
                "Risk Score",
                min_value=0,
                max_value=100,
                value=int(current.get("score", default_score)),
                step=1,
                key=f"risk_adjust_score_{narrative_id}",
            )
            confidence_value = st.selectbox(
                "Risk Score Confidence",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(str(current.get("confidence", default_conf))),
                key=f"risk_adjust_conf_{narrative_id}",
            )
            reasoning_value = st.text_area(
                "Reasoning",
                value=str(current.get("reasoning", "")),
                placeholder="Optional: explain why you adjusted score/confidence.",
                key=f"risk_adjust_reason_{narrative_id}",
                height=90,
            )
            btn_save, btn_cancel = st.columns(2)
            with btn_save:
                if st.button("Save", key=f"risk_adjust_save_{narrative_id}", use_container_width=True):
                    adjustment_store[narrative_id] = {
                        "score": int(score_value),
                        "confidence": str(confidence_value),
                        "reasoning": str(reasoning_value).strip(),
                    }
                    st.session_state["_risk_adjustments"] = adjustment_store
                    st.session_state["_show_risk_adjust_modal"] = False
                    st.rerun()
            with btn_cancel:
                if st.button("Cancel", key=f"risk_adjust_cancel_{narrative_id}", use_container_width=True):
                    st.session_state["_show_risk_adjust_modal"] = False
                    st.rerun()

        _adjust_dialog()
    else:
        # Fallback for older Streamlit versions without modal dialogs.
        st.markdown("**Adjust Score or Confidence**")
        score_value = st.slider(
            "Risk Score",
            min_value=0,
            max_value=100,
            value=int(current.get("score", default_score)),
            step=1,
            key=f"risk_adjust_inline_score_{narrative_id}",
        )
        confidence_value = st.selectbox(
            "Risk Score Confidence",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(str(current.get("confidence", default_conf))),
            key=f"risk_adjust_inline_conf_{narrative_id}",
        )
        reasoning_value = st.text_area(
            "Reasoning",
            value=str(current.get("reasoning", "")),
            placeholder="Optional: explain why you adjusted score/confidence.",
            key=f"risk_adjust_inline_reason_{narrative_id}",
            height=90,
        )
        if st.button("Save Adjustment", key=f"risk_adjust_inline_save_{narrative_id}"):
            adjustment_store[narrative_id] = {
                "score": int(score_value),
                "confidence": str(confidence_value),
                "reasoning": str(reasoning_value).strip(),
            }
            st.session_state["_risk_adjustments"] = adjustment_store
            st.success("Adjustment saved.")


def render_current_risk_score_tile(narrative, panel_height=220):
    score = float(narrative.get("score", 0))
    score_int = int(round(score))
    score_color = _risk_score_color(score)
    confidence_label = _narrative_risk_score_confidence(narrative)
    narrative_id = narrative.get("narrative_id", "na")

    with st.container(border=True):
        header_left, header_right = st.columns([0.88, 0.12], vertical_alignment="center")
        with header_left:
            st.caption("Current Risk Score")
        with header_right:
            if hasattr(st, "popover"):
                with st.popover("⋯", help="More actions"):
                    if st.button("Help", key=f"risk_score_help_btn_{narrative_id}", use_container_width=True):
                        st.session_state["_show_risk_adjust_modal"] = False
                        st.session_state["_show_risk_score_help_modal"] = True
                    if st.button("Adjust Score or Confidence", key=f"risk_score_adjust_btn_{narrative_id}", use_container_width=True):
                        st.session_state["_show_risk_score_help_modal"] = False
                        st.session_state["_show_risk_adjust_modal"] = True
            else:
                if st.button("Help", key=f"risk_score_help_btn_{narrative_id}"):
                    st.session_state["_show_risk_adjust_modal"] = False
                    st.session_state["_show_risk_score_help_modal"] = True

        if st.session_state.get("_show_risk_score_help_modal", False):
            _render_risk_score_help_modal()
        elif st.session_state.get("_show_risk_adjust_modal", False):
            _render_adjust_score_confidence_modal(narrative)

        st.markdown(
            f"""
            <div style="height:{panel_height}px;display:flex;flex-direction:column;justify-content:space-between;">
                <div style="flex:1.05;display:flex;align-items:center;justify-content:center;">
                    <div style="display:flex;align-items:flex-end;line-height:1;">
                        <span style="font-size:4rem;font-weight:800;color:{score_color};">{score_int}</span>
                        <span style="font-size:1.25rem;font-weight:600;color:#64748B;margin-left:6px;">/100</span>
                    </div>
                </div>
                <div style="height:1px;background:#E2E8F0;margin:2px 0 6px 0;"></div>
                <div style="flex:0.55;display:flex;align-items:center;justify-content:center;text-align:center;">
                    <div>
                        <div style="font-size:0.78rem;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:0.02em;">Risk Score Confidence</div>
                        <div style="margin-top:4px;font-size:1.15rem;font-weight:700;color:{score_color};">{confidence_label}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_confidence_tile(narrative):
    score = round(float(narrative.get("score", 0)), 1)
    palette = _risk_tile_palette(score)
    confidence_label = _narrative_risk_score_confidence(narrative)
    confidence_levels = pd.DataFrame(
        [
            {"Level": "Low", "Value": 1, "Selected": confidence_label == "Low"},
            {"Level": "Medium", "Value": 1, "Selected": confidence_label == "Medium"},
            {"Level": "High", "Value": 1, "Selected": confidence_label == "High"},
        ]
    )

    with st.container(border=True):
        st.caption("Risk Score Confidence")
        st.vega_lite_chart(
            confidence_levels,
            {
                "layer": [
                    {
                        "mark": {"type": "bar", "cornerRadiusTopLeft": 8, "cornerRadiusTopRight": 8},
                        "encoding": {
                            "x": {
                                "field": "Level",
                                "type": "nominal",
                                "sort": ["Low", "Medium", "High"],
                                "axis": {"title": None, "labelFontSize": 10},
                            },
                            "y": {
                                "field": "Value",
                                "type": "quantitative",
                                "axis": None,
                                "scale": {"domain": [0, 1.2]},
                            },
                            "color": {
                                "condition": {"test": "datum.Selected === true", "value": palette["value"]},
                                "value": "#E5E7EB",
                            },
                            "tooltip": [
                                {"field": "Level", "type": "nominal"},
                                {"field": "Selected", "type": "nominal", "title": "Selected"},
                            ],
                        },
                    },
                    {
                        "mark": {"type": "text", "dy": -8, "fontWeight": 700, "fontSize": 11},
                        "encoding": {
                            "x": {"field": "Level", "type": "nominal", "sort": ["Low", "Medium", "High"]},
                            "y": {"field": "Value", "type": "quantitative"},
                            "text": {"field": "Level", "type": "nominal"},
                            "color": {
                                "condition": {"test": "datum.Selected === true", "value": palette["value"]},
                                "value": "#94A3B8",
                            },
                        },
                    },
                ],
                "height": 180,
                "width": "container",
                "autosize": {"type": "fit", "contains": "padding"},
                "padding": {"left": 6, "right": 10, "top": 8, "bottom": 8},
                "config": {"view": {"stroke": None}},
            },
            use_container_width=True,
        )
        st.markdown(
            f'<div style="text-align:center;font-size:0.86rem;font-weight:700;color:{palette["value"]};margin-top:2px;">Selected: {confidence_label}</div>',
            unsafe_allow_html=True,
        )


def _classify_narrative_type(narrative):
    posts = narrative.get("posts")
    if posts is None or "text" not in posts or len(posts) == 0:
        return "Operational Resilience"
    text = " ".join(posts["text"].astype(str).tolist()).lower()
    scores = {
        theme: sum(1 for kw in keywords if kw in text)
        for theme, keywords in RISK_THEME_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Operational Resilience"


def _area_line_chart_spec(x_field, x_type, y_field, x_axis, y_axis, color="#2563EB", height=180):
    base_encoding = {
        "x": {"field": x_field, "type": x_type, "axis": x_axis},
        "y": {"field": y_field, "type": "quantitative", "axis": y_axis},
    }
    return {
        "layer": [
            {
                "mark": {"type": "area", "color": color, "opacity": 0.2, "interpolate": "monotone"},
                "encoding": base_encoding,
            },
            {
                "mark": {
                    "type": "line",
                    "color": color,
                    "interpolate": "monotone",
                    "strokeWidth": 2,
                    "point": True,
                },
                "encoding": base_encoding,
            },
        ],
        "height": height,
        "width": "container",
        "autosize": {"type": "fit", "contains": "padding"},
        "padding": {"left": 6, "right": 16, "top": 4, "bottom": 4},
        "config": {"view": {"stroke": None}},
    }


def _render_risk_score_over_time_chart(narrative):
    posts = narrative.get("posts")
    if posts is None or len(posts) == 0:
        end_day = pd.Timestamp.now().normalize()
        last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
        risk_hourly = pd.DataFrame({"date": last_7_days, "RiskScore": [0.0] * 7})
    else:
        time_col = next(
            (c for c in ["created_at", "timestamp", "datetime", "date", "published_at", "posted_at", "time"] if c in posts.columns),
            None,
        )
        if time_col is None:
            end_day = pd.Timestamp.now().normalize()
            last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
            risk_hourly = pd.DataFrame({"date": last_7_days, "RiskScore": [0.0] * 7})
        else:
            this = posts[[time_col]].copy()
            this["date"] = pd.to_datetime(this[time_col], errors="coerce").dt.normalize()
            this = this.dropna(subset=["date"])
            if this.empty:
                end_day = pd.Timestamp.now().normalize()
                last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
                risk_hourly = pd.DataFrame({"date": last_7_days, "RiskScore": [0.0] * 7})
            else:
                end_day = this["date"].max()
                last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
                base_risk = float(narrative.get("score", 0))
                if "resolution_confidence" in posts.columns:
                    conf = pd.to_numeric(posts["resolution_confidence"], errors="coerce").fillna(0.5)
                    this["risk_value"] = (base_risk * conf.values).astype(float)
                else:
                    this["risk_value"] = float(base_risk)

                risk_hourly = (
                    this.groupby("date")["risk_value"]
                    .mean()
                    .sort_index()
                    .rename("RiskScore")
                    .reset_index()
                )
                risk_hourly = (
                    pd.DataFrame({"date": last_7_days})
                    .merge(risk_hourly, on="date", how="left")
                    .fillna({"RiskScore": 0.0})
                )
                risk_hourly["RiskScore"] = risk_hourly["RiskScore"].astype(float)
    spec = _area_line_chart_spec(
        x_field="date",
        x_type="temporal",
        y_field="RiskScore",
        x_axis={"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
        y_axis={"title": None, "labelFontSize": 9, "titleFontSize": 10},
        height=190,
    )
    spec["layer"].append(
        {
            "transform": [{"regression": "RiskScore", "on": "date"}],
            "mark": {
                "type": "line",
                "color": "#94A3B8",
                "strokeDash": [4, 4],
                "strokeWidth": 1.5,
                "opacity": 0.45,
            },
            "encoding": {
                "x": {"field": "date", "type": "temporal", "axis": {"title": "Date", "labelFontSize": 9, "titleFontSize": 10}},
                "y": {"field": "RiskScore", "type": "quantitative", "axis": {"title": None, "labelFontSize": 9, "titleFontSize": 10}},
            },
        }
    )
    spec["encoding"] = {
        "tooltip": [
            {"field": "date", "type": "temporal"},
            {"field": "RiskScore", "type": "quantitative", "format": ".1f"},
        ]
    }
    st.vega_lite_chart(risk_hourly, spec, use_container_width=True)


def _render_matched_posts_charts(narrative):
    posts = narrative.get("posts")
    if posts is None or len(posts) == 0:
        return

    c1 = st.container()
    time_col = next(
        (c for c in ["created_at", "timestamp", "datetime", "date", "published_at", "posted_at", "time"] if c in posts.columns),
        None,
    )

    with c1:
        with st.container(border=True):
            st.caption("Matched Posts Info by Date")
            if time_col is not None:
                ts = posts.copy()
                ts["date"] = pd.to_datetime(ts[time_col], errors="coerce").dt.normalize()
                ts = ts.dropna(subset=["date"])
                if len(ts):
                    end_day = ts["date"].max()
                else:
                    end_day = pd.Timestamp.now().normalize()

                last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
                ts_counts = (
                    ts.groupby("date").size().rename("Posts").reset_index()
                    if len(ts)
                    else pd.DataFrame({"date": [], "Posts": []})
                )
                ts_counts = (
                    pd.DataFrame({"date": last_7_days})
                    .merge(ts_counts, on="date", how="left")
                    .fillna({"Posts": 0})
                )
                ts_counts["Posts"] = ts_counts["Posts"].astype(float)

                author_col = next((c for c in ["author_id", "author", "username", "user_id"] if c in ts.columns), None)
                author_counts = (
                    ts.groupby("date")[author_col].nunique().rename("UniqueAuthors").reset_index()
                    if len(ts) and author_col is not None
                    else ts_counts[["date"]].assign(UniqueAuthors=ts_counts["Posts"])
                )

                engagement_parts = []
                for col in ["likes", "shares", "comments", "views"]:
                    if col in ts.columns:
                        engagement_parts.append(pd.to_numeric(ts[col], errors="coerce").fillna(0))
                ts["EstimatedEngagement"] = sum(engagement_parts) if engagement_parts else 0.0
                engagement_counts = (
                    ts.groupby("date")["EstimatedEngagement"].sum().rename("EstimatedEngagement").reset_index()
                    if len(ts)
                    else pd.DataFrame({"date": [], "EstimatedEngagement": []})
                )

                ts_counts = (
                    ts_counts
                    .merge(author_counts, on="date", how="left")
                    .merge(engagement_counts, on="date", how="left")
                    .fillna({"UniqueAuthors": 0, "EstimatedEngagement": 0})
                )
                ts_counts["UniqueAuthors"] = ts_counts["UniqueAuthors"].astype(float)
                ts_counts["EstimatedEngagement"] = ts_counts["EstimatedEngagement"].astype(float)

                metric_labels = ["Posts", "Unique Authors", "Estimated Engagement"]
                metric_col, _metric_spacer = st.columns([0.75, 3.25])
                with metric_col:
                    selected_metric = st.selectbox(
                        "Metric",
                        metric_labels,
                        index=0,
                        label_visibility="collapsed",
                        key=f"matched_posts_metric_{narrative.get('narrative_id', 'na')}",
                    )
                metric_field = {
                    "Posts": "Posts",
                    "Unique Authors": "UniqueAuthors",
                    "Estimated Engagement": "EstimatedEngagement",
                }[selected_metric]
                metric_color = {
                    "Posts": "#2563EB",
                    "Unique Authors": "#0F766E",
                    "Estimated Engagement": "#F59E0B",
                }[selected_metric]

                spec = _area_line_chart_spec(
                    x_field="date",
                    x_type="temporal",
                    y_field=metric_field,
                    x_axis={"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
                    y_axis={"title": None, "labelFontSize": 9, "titleFontSize": 10},
                    color=metric_color,
                )
                spec["encoding"] = {
                    "tooltip": [
                        {"field": "date", "type": "temporal"},
                        {"field": metric_field, "type": "quantitative", "title": selected_metric},
                    ]
                }
                st.vega_lite_chart(ts_counts, spec, use_container_width=True)
            else:
                st.info("No timestamp field found.")


def build_narrative_pill_groups(narrative_id, narratives):
    narrative = next(n for n in narratives if n["narrative_id"] == narrative_id)
    risk_types = narrative.get("taxonomy_labels", [])
    if not isinstance(risk_types, list) or not risk_types:
        risk_types = ["Operational Resilience"]
    else:
        risk_types = [str(v).strip() for v in risk_types if str(v).strip()]
        if not risk_types:
            risk_types = ["Operational Resilience"]

    contributors = _build_shap_like_contributors(narrative)
    drivers = (
        contributors.assign(abs_contribution=contributors["Contribution"].abs())
        .sort_values("abs_contribution", ascending=False)["Feature"]
        .head(3)
        .tolist()
    )
    return risk_types, drivers


def build_narrative_sentiment_summary(narrative_id, narratives):
    narrative = next(n for n in narratives if n["narrative_id"] == narrative_id)
    explicit_summary = _sanitize_narrative_description(narrative.get("summary", ""))
    if explicit_summary:
        return explicit_summary

    posts = narrative.get("posts")
    if posts is None or "text" not in posts or len(posts) == 0:
        return "No narrative text available to estimate sentiment."

    post_count = len(posts)
    risk_score = float(narrative.get("score", 0))
    entity_name = str(narrative.get("entity", "this entity")).strip() or "this entity"
    labels = narrative.get("taxonomy_labels", [])
    if isinstance(labels, list):
        clean_labels = [str(v).strip() for v in labels if str(v).strip()]
    else:
        clean_labels = []
    primary_theme = clean_labels[0] if clean_labels else "Operational Resilience"
    secondary_theme = clean_labels[1] if len(clean_labels) > 1 else None

    if risk_score >= 80:
        severity = "high"
    elif risk_score >= 60:
        severity = "elevated"
    elif risk_score >= 40:
        severity = "moderate"
    else:
        severity = "emerging"

    theme_text = primary_theme if secondary_theme is None else f"{primary_theme} with overlap into {secondary_theme}"
    impact_text = THEME_IMPACT.get(primary_theme, "reputational and operational pressure.")

    fallback = (
        f"Across {post_count} matched posts, this {entity_name}-linked narrative shows {severity} risk centered on "
        f"{theme_text}. The common pattern suggests {impact_text}"
    )
    return _sanitize_narrative_description(fallback)


def build_narrative_quotes(narrative_id, narratives, limit=3):
    narrative = next(n for n in narratives if n["narrative_id"] == narrative_id)
    posts = narrative.get("posts")
    if posts is None or "text" not in posts or len(posts) == 0:
        return []

    def _valid_snippet(value):
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return re.sub(r"\s+", " ", text)

    # Prefer entity-centered context extracted during risk enrichment.
    source_col = "risk_context" if "risk_context" in posts.columns else "text"
    source_texts = []
    for t in posts[source_col].astype(str).tolist():
        cleaned = _valid_snippet(t)
        if cleaned:
            source_texts.append(cleaned)

    # Fallback for older artifacts that may not include usable risk_context.
    if not source_texts:
        for t in posts["text"].astype(str).tolist():
            cleaned = _valid_snippet(t)
            if cleaned:
                source_texts.append(cleaned)

    quotes = []
    seen = set()

    for text in source_texts:
        q = _shorten(text, max_len=120)
        if q and q.lower() not in seen:
            quotes.append(q)
            seen.add(q.lower())
        if len(quotes) >= limit:
            return quotes

    # If fewer than requested, split context into additional snippets.
    for text in source_texts:
        words = text.split()
        chunk_size = 18
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size]).strip()
            if not chunk:
                continue
            q = _shorten(chunk, max_len=120)
            if q.lower() in seen:
                continue
            quotes.append(q)
            seen.add(q.lower())
            if len(quotes) >= limit:
                return quotes

    return quotes[:limit]


def render_narrative_detail(narrative_id, narratives, entity_posts, entity_options=None):
    narrative = next(n for n in narratives if n["narrative_id"] == narrative_id)
    raw_entity_options = entity_options or []
    cleaned_entity_options = []
    for entity_name in raw_entity_options:
        name = str(entity_name).strip()
        if not name or name.lower() == "all":
            continue
        if name not in cleaned_entity_options:
            cleaned_entity_options.append(name)
    entity_match_options = ["None"] + cleaned_entity_options

    st.markdown("##### Risk Analysis")
    risk_summary_col, risk_daily_col = st.columns([0.9, 2.1], vertical_alignment="top")
    with risk_summary_col:
        render_current_risk_score_tile(narrative, panel_height=220)
    with risk_daily_col:
        with st.container(border=True):
            st.caption("Risk Score Over Time")
            _render_risk_score_over_time_chart(narrative)
            st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True)

    risk_left_col = st.container()
    shap_like = _build_shap_like_contributors(narrative)
    shap_plot_df = shap_like.copy()
    shap_plot_df["AbsContribution"] = pd.to_numeric(
        shap_plot_df["Contribution"], errors="coerce"
    ).fillna(0.0).abs()
    max_abs_contrib = float(shap_plot_df["AbsContribution"].max()) if len(shap_plot_df) else 0.0
    max_abs_contrib = max(1.0, max_abs_contrib)
    mid_abs_contrib = max_abs_contrib / 2.0
    with risk_left_col:
        with st.container(border=True):
            st.caption("Risk Score Drivers")
            st.vega_lite_chart(
                shap_plot_df,
                {
                    "mark": {"type": "bar"},
                    "encoding": {
                        "y": {
                            "field": "Feature",
                            "type": "nominal",
                            "sort": "-x",
                            "axis": {"title": None, "labelLimit": 280, "labelFontSize": 9},
                        },
                        "x": {
                            "field": "AbsContribution",
                            "type": "quantitative",
                            "scale": {"domain": [0, max_abs_contrib]},
                            "axis": {
                                "title": None,
                                "values": [0, mid_abs_contrib, max_abs_contrib],
                                "labelExpr": (
                                    f"datum.value <= 0.000001 ? 'Low' : "
                                    f"(datum.value >= {max_abs_contrib * 0.999} ? 'High' : 'Medium')"
                                ),
                            },
                        },
                        "color": {
                            "condition": {"test": "datum.Contribution >= 0", "value": "#DC2626"},
                            "value": "#16A34A",
                        },
                        "tooltip": [
                            {"field": "Feature", "type": "nominal"},
                            {"field": "Contribution", "type": "quantitative", "format": ".2f"},
                            {"field": "AbsContribution", "type": "quantitative", "format": ".2f", "title": "Relative strength"},
                        ],
                    },
                    "height": 220,
                },
                use_container_width=True,
            )

    st.markdown("---")
    posts = narrative.get("posts")
    matched_post_count = len(posts) if posts is not None else 0
    st.markdown(f"##### Matched Posts (Total: {matched_post_count:,})")

    if posts is None or len(posts) == 0:
        st.info("No posts available for this narrative.")
        return

    posts_df = posts.copy()
    if "created_at" in posts_df.columns:
        created_series = pd.to_datetime(posts_df["created_at"], errors="coerce")
        created_out = created_series.dt.strftime("%Y-%m-%d %H:%M").fillna(posts_df["created_at"].astype(str))
    else:
        created_out = pd.Series(["-"] * len(posts_df), index=posts_df.index)

    platform_out = posts_df["platform"].astype(str) if "platform" in posts_df.columns else "-"
    text_out = posts_df["text"].astype(str) if "text" in posts_df.columns else "-"

    if "followers" in posts_df.columns:
        author_followers = pd.to_numeric(posts_df["followers"], errors="coerce").fillna(0)
    elif "author_influence" in posts_df.columns:
        author_followers = pd.to_numeric(posts_df["author_influence"], errors="coerce").fillna(0)
    else:
        author_followers = pd.Series([0] * len(posts_df), index=posts_df.index)

    likes = pd.to_numeric(posts_df["likes"], errors="coerce").fillna(0) if "likes" in posts_df.columns else 0
    shares = pd.to_numeric(posts_df["shares"], errors="coerce").fillna(0) if "shares" in posts_df.columns else 0
    comments = pd.to_numeric(posts_df["comments"], errors="coerce").fillna(0) if "comments" in posts_df.columns else 0
    views = pd.to_numeric(posts_df["views"], errors="coerce").fillna(0) if "views" in posts_df.columns else 0
    viewer_engagement = likes + shares + comments + views

    if "resolution_confidence" in posts_df.columns:
        match_confidence = pd.to_numeric(posts_df["resolution_confidence"], errors="coerce").fillna(0)
    else:
        match_confidence = pd.Series([0.0] * len(posts_df), index=posts_df.index)

    view_df = pd.DataFrame(
        {
            "Created At": created_out,
            "Platform": platform_out,
            "Post Summary": text_out,
            "Author Followers": author_followers.astype(float),
            "Viewer Engagement": viewer_engagement.astype(float),
            "Match Confidence": match_confidence.astype(float),
        }
    )
    display_df = view_df.copy()
    display_df["Inspect Raw Data"] = "Inspect Raw Data"
    display_df["Mark as False Positive"] = "Mark as False Positive"

    def _match_conf_style(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "color: #64748B;"
        if v >= 0.7:
            return "color: #16A34A;"
        if v >= 0.4:
            return "color: #B45309;"
        return "color: #DC2626;"

    def _action_style(_value):
        return (
            "color: #2563EB; "
            "font-weight: 600; "
            "text-decoration: underline; "
            "text-underline-offset: 2px; "
        )

    styled_display_df = (
        display_df.style
        .map(_match_conf_style, subset=["Match Confidence"])
        .map(_action_style, subset=["Inspect Raw Data", "Mark as False Positive"])
    )

    event = st.dataframe(
        styled_display_df,
        width="stretch",
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-cell",
        column_config={
            "Created At": st.column_config.TextColumn("Created At", width="medium"),
            "Platform": st.column_config.TextColumn("Platform", width="small"),
            "Post Summary": st.column_config.TextColumn("Post Summary", width="large"),
            "Author Followers": st.column_config.NumberColumn("Author Followers", format="%.0f", width="small"),
            "Viewer Engagement": st.column_config.NumberColumn("Viewer Engagement", format="%.0f", width="small"),
            "Match Confidence": st.column_config.NumberColumn("Match Confidence", format="%.2f", width="small"),
            "Inspect Raw Data": st.column_config.TextColumn("Inspect Raw Data", width="medium"),
            "Mark as False Positive": st.column_config.TextColumn("Mark as False Positive", width="medium"),
        },
    )

    _render_matched_posts_charts(narrative)

    selection = event.get("selection", {}) if isinstance(event, dict) else {}
    selected_cells = selection.get("cells", [])

    selected_row_idx = None
    selected_col_name = None

    # Only trigger actions from explicit single-cell selections.
    # Row-level fallback payloads can become sticky and re-trigger old actions.
    if selected_cells:
        first_cell = selected_cells[0]
        if isinstance(first_cell, dict):
            selected_row_idx = first_cell.get("row")
            selected_col_name = first_cell.get("column")
        elif isinstance(first_cell, (tuple, list)) and len(first_cell) >= 2:
            selected_row_idx = first_cell[0]
            selected_col_name = first_cell[1]

    if isinstance(selected_col_name, int):
        selected_col_name = display_df.columns[selected_col_name] if 0 <= selected_col_name < len(display_df.columns) else None

    action_columns = {"Inspect Raw Data", "Mark as False Positive"}
    if selected_col_name not in action_columns:
        st.session_state["_detail_last_action_sig"] = None
        selected_row_idx = None
        selected_col_name = None

    current_action_sig = None
    if selected_row_idx is not None and selected_col_name is not None:
        current_action_sig = f"{int(selected_row_idx)}::{selected_col_name}"
    last_action_sig = st.session_state.get("_detail_last_action_sig")

    if (
        selected_row_idx is not None
        and 0 <= int(selected_row_idx) < len(posts_df)
        and current_action_sig is not None
        and current_action_sig != last_action_sig
    ):
        row_pos = int(selected_row_idx)
        raw_post = posts_df.iloc[row_pos]
        if selected_col_name == "Inspect Raw Data":
            st.session_state["_detail_false_positive_open"] = False
            author_name = next(
                (
                    str(raw_post.get(col))
                    for col in ["author_name", "author", "username", "author_id", "user_id"]
                    if col in raw_post.index and pd.notna(raw_post.get(col))
                ),
                "-",
            )
            st.session_state["_detail_inspect_payload"] = {
                "created_at": str(raw_post.get("created_at", display_df.iloc[row_pos].get("Created At", "-"))),
                "platform": str(raw_post.get("platform", display_df.iloc[row_pos].get("Platform", "-"))),
                "url": str(raw_post.get("url", "-")),
                "author_name": author_name,
                "text": str(raw_post.get("text", display_df.iloc[row_pos].get("Post Summary", ""))),
                "raw": raw_post.to_dict(),
            }
            st.session_state["_detail_inspect_open"] = True
        elif selected_col_name == "Mark as False Positive":
            st.session_state["_detail_inspect_open"] = False
            st.session_state["_detail_false_positive_payload"] = {
                "created_at": str(raw_post.get("created_at", display_df.iloc[row_pos].get("Created At", "-"))),
                "platform": str(raw_post.get("platform", display_df.iloc[row_pos].get("Platform", "-"))),
                "summary": str(raw_post.get("text", display_df.iloc[row_pos].get("Post Summary", ""))),
                "entity": str(raw_post.get("entity", "")).strip(),
            }
            st.session_state["_detail_false_positive_open"] = True

        st.session_state["_detail_last_action_sig"] = current_action_sig

    if st.session_state.get("_detail_inspect_open", False):
        payload = st.session_state.get("_detail_inspect_payload")
        if payload:
            if hasattr(st, "dialog"):
                @st.dialog("Inspect Raw Data")
                def _inspect_raw_data_dialog():
                    st.markdown(f"**Author:** {payload.get('author_name', '-')}")
                    st.markdown(f"**Platform:** {payload.get('platform', '-')}")
                    st.markdown(f"**Created At:** {payload.get('created_at', '-')}")
                    st.markdown(f"**Post URL:** {payload.get('url', '-')}")
                    st.markdown("**Full Text:**")
                    st.write(payload.get("text", ""))
                    with st.expander("Raw JSON"):
                        st.json(payload.get("raw", {}))
                    if st.button("Close", key=f"inspect_raw_close_{narrative_id}"):
                        st.session_state["_detail_inspect_open"] = False
                        st.rerun()

                _inspect_raw_data_dialog()
            else:
                with st.expander("Inspect Raw Data", expanded=True):
                    st.markdown(f"**Author:** {payload.get('author_name', '-')}")
                    st.markdown(f"**Platform:** {payload.get('platform', '-')}")
                    st.markdown(f"**Created At:** {payload.get('created_at', '-')}")
                    st.markdown(f"**Post URL:** {payload.get('url', '-')}")
                    st.markdown("**Full Text:**")
                    st.write(payload.get("text", ""))
                    with st.expander("Raw JSON"):
                        st.json(payload.get("raw", {}))

    elif st.session_state.get("_detail_false_positive_open", False):
        fp_payload = st.session_state.get("_detail_false_positive_payload", {})
        if hasattr(st, "dialog"):
            @st.dialog("Mark as False Positive")
            def _false_positive_dialog():
                st.markdown("Are you sure you want to mark this as a false positive match?")
                st.markdown(
                    f"**Created At:** {fp_payload.get('created_at', '-')}\n\n"
                    f"**Platform:** {fp_payload.get('platform', '-')}"
                )
                selected_entity = str(fp_payload.get("entity", "")).strip()
                if selected_entity not in entity_match_options:
                    selected_entity = "None"
                st.selectbox(
                    "Change Entity Match",
                    entity_match_options,
                    index=entity_match_options.index(selected_entity),
                    key=f"fp_entity_match_{narrative_id}",
                )
                st.text_area(
                    "Reasoning",
                    placeholder="Explain why this should be marked as a false positive.",
                    key=f"fp_reason_{narrative_id}",
                    height=100,
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Confirm", key=f"fp_confirm_{narrative_id}", use_container_width=True):
                        st.session_state["_detail_false_positive_open"] = False
                        if hasattr(st, "toast"):
                            st.toast("Done.")
                        else:
                            st.success("Done.")
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"fp_cancel_{narrative_id}", use_container_width=True):
                        st.session_state["_detail_false_positive_open"] = False
                        st.rerun()

            _false_positive_dialog()
        else:
            with st.expander("Mark as False Positive", expanded=True):
                st.markdown("Are you sure you want to mark this as a false positive match?")
                st.markdown(
                    f"**Created At:** {fp_payload.get('created_at', '-')}\n\n"
                    f"**Platform:** {fp_payload.get('platform', '-')}"
                )
                selected_entity = str(fp_payload.get("entity", "")).strip()
                if selected_entity not in entity_match_options:
                    selected_entity = "None"
                st.selectbox(
                    "Change Entity Match",
                    entity_match_options,
                    index=entity_match_options.index(selected_entity),
                    key=f"fp_entity_match_inline_{narrative_id}",
                )
                st.text_area(
                    "Reasoning",
                    placeholder="Explain why this should be marked as a false positive.",
                    key=f"fp_reason_inline_{narrative_id}",
                    height=100,
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Confirm", key=f"fp_confirm_inline_{narrative_id}", use_container_width=True):
                        st.session_state["_detail_false_positive_open"] = False
                        st.success("Done.")
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"fp_cancel_inline_{narrative_id}", use_container_width=True):
                        st.session_state["_detail_false_positive_open"] = False
                        st.rerun()
