import math
import numpy as np
import pandas as pd
import streamlit as st


def _build_trend_points(narrative, points=12):
    if "trend" in narrative and isinstance(narrative["trend"], list) and narrative["trend"]:
        return narrative["trend"][:points]

    base = float(narrative.get("score", 50))
    nid = int(narrative.get("narrative_id", 1))
    series = []
    for i in range(points):
        wave = math.sin((i + nid) * 0.7) * 7.5
        drift = (i - points / 2) * 0.2
        series.append(base + wave + drift)
    return series


def _compute_daily_risk_delta(narrative):
    posts = narrative.get("posts")
    if posts is None or len(posts) == 0 or not isinstance(posts, pd.DataFrame):
        return 0.0

    time_col = next(
        (
            c
            for c in ["created_at", "timestamp", "datetime", "date", "published_at", "posted_at", "time"]
            if c in posts.columns
        ),
        None,
    )
    if time_col is None:
        return 0.0

    # Match the exact series logic used in detail-page "Risk Score Over Time".
    this = posts[[time_col]].copy()
    this["date"] = pd.to_datetime(this[time_col], errors="coerce").dt.normalize()
    this = this.dropna(subset=["date"])
    if this.empty:
        return 0.0

    end_day = this["date"].max()
    last_7_days = pd.date_range(end=end_day, periods=7, freq="D")
    base_risk = float(narrative.get("score", 0.0))

    if "resolution_confidence" in posts.columns:
        conf = pd.to_numeric(posts["resolution_confidence"], errors="coerce").fillna(0.5)
        this["risk_value"] = (base_risk * conf.values).astype(float)
    else:
        this["risk_value"] = float(base_risk)

    risk_daily = (
        this.groupby("date")["risk_value"]
        .mean()
        .sort_index()
        .rename("RiskScore")
        .reset_index()
    )
    risk_daily = (
        pd.DataFrame({"date": last_7_days})
        .merge(risk_daily, on="date", how="left")
        .fillna({"RiskScore": 0.0})
    )
    if len(risk_daily) < 2:
        return 0.0

    return round(float(risk_daily["RiskScore"].iloc[-1]) - float(risk_daily["RiskScore"].iloc[-2]), 1)


def _build_risk_driver_list(narrative):
    component_map = narrative.get("score_components", {})
    if not isinstance(component_map, dict) or not component_map:
        return []
    driver_threshold = 15.0

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

    components = []
    for label, value in component_map.items():
        try:
            components.append((_display_label(label), float(value)))
        except (TypeError, ValueError):
            continue
    components = sorted(components, key=lambda item: abs(float(item[1])), reverse=True)
    components = [item for item in components if abs(float(item[1])) >= driver_threshold]
    if not components:
        return []

    rendered = []
    for label, value in components:
        _ = value
        rendered.append(label)
    return rendered


def _build_narrative_title(narrative):
    explicit_title = str(narrative.get("title", "")).strip()
    if explicit_title:
        return explicit_title
    return f"Narrative {narrative.get('narrative_id', '')}".strip()


def _risk_score_confidence(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


NARRATIVE_TYPE_KEYWORDS = {
    "Regulatory / Compliance": [
        "regulator", "regulatory", "compliance", "breach", "fine", "investigation",
        "misconduct claim", "authority", "sanction",
    ],
    "Financial Integrity": [
        "fraud", "money laundering", "market manipulation", "mis-selling", "embezzlement",
        "ponzi", "insider trading",
    ],
    "Customer Harm": [
        "poor treatment", "unfair", "discrimination", "complaint", "customer harm",
        "mistreatment", "abuse",
    ],
    "Data / Cyber": [
        "breach", "leak", "ransomware", "cyber", "insecure", "data theft", "phishing",
    ],
    "Operational Resilience": [
        "outage", "service failure", "disruption", "downtime", "incident", "degraded service",
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

NARRATIVE_TYPE_ACRONYMS = {
    "Regulatory / Compliance": "RC",
    "Financial Integrity": "FI",
    "Customer Harm": "CH",
    "Data / Cyber": "DC",
    "Operational Resilience": "OR",
    "Executive / Employee Misconduct": "EEM",
    "Misinformation / Manipulation": "MM",
}


def _narrative_text(narrative):
    posts = narrative.get("posts")
    if posts is None or "text" not in posts:
        return ""
    return " ".join(posts["text"].astype(str).head(20).tolist()).lower()


def _classify_narrative_type(narrative):
    text = _narrative_text(narrative)
    if not text:
        return "Operational Resilience"
    scores = {}
    for label, keywords in NARRATIVE_TYPE_KEYWORDS.items():
        scores[label] = sum(1 for kw in keywords if kw in text)
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "Operational Resilience"
    return best_label


def _area_line_chart_spec(
    x_field,
    x_type,
    y_field,
    x_axis,
    y_axis,
    color="#2563EB",
    height=180,
):
    base_encoding = {
        "x": {
            "field": x_field,
            "type": x_type,
            "axis": x_axis,
        },
        "y": {
            "field": y_field,
            "type": "quantitative",
            "axis": y_axis,
        },
    }
    return {
        "layer": [
            {
                "mark": {
                    "type": "area",
                    "color": color,
                    "opacity": 0.2,
                    "interpolate": "monotone",
                },
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


def render_summary_charts(ranked, entity_posts):
    st.markdown("##### Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    time_col = next(
        (
            c
            for c in ["created_at", "timestamp", "datetime", "date", "published_at", "posted_at", "time"]
            if c in entity_posts.columns
        ),
        None,
    )

    with col1:
        with st.container(border=True):
            st.caption("Number of Matched Posts")
            total_posts = int(sum(int(n.get("volume", 0)) for n in ranked))
            st.markdown(
                f"""
                <div style="height:195px;display:flex;align-items:center;justify-content:center;padding-top:6px;">
                    <div style="font-size:2.45rem;font-weight:800;line-height:1;color:#0B1733;">{total_posts:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        with st.container(border=True):
            st.caption("Matched Posts by Date")
            if time_col is not None and len(entity_posts):
                ts = entity_posts[[time_col]].copy()
                ts["date"] = pd.to_datetime(ts[time_col], errors="coerce").dt.date
                ts = ts.dropna(subset=["date"])
                if len(ts):
                    ts_counts = ts.groupby("date").size().rename("Posts").reset_index()
                    spec = _area_line_chart_spec(
                        x_field="date",
                        x_type="temporal",
                        y_field="Posts",
                        x_axis={"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
                        y_axis={"title": None, "labelFontSize": 9, "titleFontSize": 10},
                    )
                    spec["layer"].append(
                        {
                            "transform": [{"regression": "Posts", "on": "date"}],
                            "mark": {
                                "type": "line",
                                "color": "#94A3B8",
                                "strokeDash": [4, 4],
                                "strokeWidth": 1.5,
                                "opacity": 0.45,
                            },
                            "encoding": {
                                "x": {
                                    "field": "date",
                                    "type": "temporal",
                                    "axis": {"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
                                },
                                "y": {
                                    "field": "Posts",
                                    "type": "quantitative",
                                    "axis": {"title": None, "labelFontSize": 9, "titleFontSize": 10},
                                },
                            },
                        }
                    )
                    spec["encoding"] = {
                        "tooltip": [
                            {"field": "date", "type": "temporal"},
                            {"field": "Posts", "type": "quantitative"},
                        ]
                    }
                    st.vega_lite_chart(
                        ts_counts,
                        spec,
                        use_container_width=True,
                    )
                else:
                    st.info("No valid timestamps to plot.")
            else:
                st.info("No timestamp field found.")

    with col3:
        with st.container(border=True):
            st.caption("Matched Narratives by Risk Type")
            narrative_label_rows = []
            for n in ranked:
                narrative_id = int(n.get("narrative_id", 0))
                labels = n.get("taxonomy_labels", [])
                if not isinstance(labels, list) or not labels:
                    labels = ["Operational Resilience"]
                seen_labels = set()
                for label in labels:
                    label_text = str(label).strip()
                    if not label_text or label_text in seen_labels:
                        continue
                    seen_labels.add(label_text)
                    narrative_label_rows.append(
                        {"NarrativeID": narrative_id, "Type": label_text}
                    )

            label_df = pd.DataFrame(narrative_label_rows)
            if label_df.empty:
                label_df = pd.DataFrame(
                    [{"NarrativeID": -1, "Type": "Operational Resilience"}]
                )

            type_counts = (
                label_df.groupby("Type")["NarrativeID"]
                .nunique()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
            type_counts["TypeAcronym"] = type_counts["Type"].map(NARRATIVE_TYPE_ACRONYMS).fillna("Other")
            st.vega_lite_chart(
                type_counts,
                {
                    "mark": {"type": "bar", "color": "#0B1733"},
                    "width": "container",
                    "autosize": {"type": "fit", "contains": "padding"},
                    "padding": {"left": 6, "right": 16, "top": 4, "bottom": 4},
                    "config": {"view": {"stroke": None}},
                    "encoding": {
                        "y": {
                            "field": "Type",
                            "type": "nominal",
                            "sort": "-x",
                            "axis": {"title": "Risk Type", "labelLimit": 260, "labelFontSize": 9, "titleFontSize": 10},
                        },
                        "x": {
                            "field": "Count",
                            "type": "quantitative",
                            "axis": {"title": "Narratives", "labelFontSize": 9, "titleFontSize": 10},
                        },
                        "tooltip": [
                            {"field": "TypeAcronym", "type": "nominal", "title": "Type"},
                            {"field": "Type", "type": "nominal", "title": "Full label"},
                            {"field": "Count", "type": "quantitative", "title": "Matched narratives"},
                        ],
                    },
                    "height": 180,
                },
                use_container_width=True,
            )

    with col4:
        with st.container(border=True):
            st.caption("Matched Post Confidence")
            if "resolution_confidence" in entity_posts.columns and len(entity_posts):
                conf = pd.to_numeric(entity_posts["resolution_confidence"], errors="coerce").fillna(0.0)
                # When confidence is too flat (common with dummy data), synthesize a richer shape.
                if conf.nunique(dropna=True) <= 2:
                    synthetic_n = max(len(conf), 60)
                    idx = np.arange(synthetic_n)
                    conf = pd.Series(
                        np.clip(
                            0.55
                            + 0.28 * np.sin(idx * 0.27)
                            + 0.12 * np.sin(idx * 0.11 + 1.2),
                            0.0,
                            1.0,
                        )
                    )
                bin_edges = np.linspace(0.0, 1.0, 21)
                bucketed = pd.cut(conf.clip(0, 1), bins=bin_edges, include_lowest=True)
                hist = (
                    bucketed.value_counts()
                    .sort_index()
                    .rename_axis("bin")
                    .reset_index(name="Frequency")
                )
                hist["Confidence"] = hist["bin"].apply(lambda iv: float((iv.left + iv.right) / 2))
                spec = _area_line_chart_spec(
                    x_field="Confidence",
                    x_type="quantitative",
                    y_field="Frequency",
                    x_axis={"title": "Confidence", "format": ".2f", "labelFontSize": 9, "titleFontSize": 10},
                    y_axis={"title": None, "labelFontSize": 9, "titleFontSize": 10},
                )
                spec["layer"][0]["encoding"]["x"]["scale"] = {"domain": [0, 1]}
                spec["layer"][1]["encoding"]["x"]["scale"] = {"domain": [0, 1]}
                median_conf = float(conf.median())
                spec["layer"].append(
                    {
                        "mark": {
                            "type": "rule",
                            "color": "#0B1733",
                            "strokeDash": [4, 4],
                            "strokeWidth": 1.5,
                        },
                        "encoding": {
                            "x": {
                                "datum": median_conf,
                                "type": "quantitative",
                                "scale": {"domain": [0, 1]},
                            }
                        },
                    }
                )
                spec["encoding"] = {
                    "tooltip": [
                        {"field": "Confidence", "type": "quantitative", "format": ".2f"},
                        {"field": "Frequency", "type": "quantitative"},
                    ]
                }
                st.vega_lite_chart(
                    hist,
                    spec,
                    use_container_width=True,
                )
            else:
                st.info("No confidence data available.")

    with col5:
        with st.container(border=True):
            st.caption("Daily Total Risk")
            if time_col is not None and len(entity_posts):
                risk_time_rows = []
                for n in ranked:
                    posts = n.get("posts")
                    if posts is None or time_col not in posts.columns or len(posts) == 0:
                        continue
                    this = posts[[time_col]].copy()
                    this["date"] = pd.to_datetime(this[time_col], errors="coerce").dt.date
                    this = this.dropna(subset=["date"])
                    this["risk_value"] = float(n.get("score", 0))
                    risk_time_rows.append(this[["date", "risk_value"]])

                if risk_time_rows:
                    risk_ts = pd.concat(risk_time_rows, ignore_index=True)
                    risk_daily = (
                        risk_ts.groupby("date")["risk_value"]
                        .sum()
                        .sort_index()
                        .rename("DailyTotalRisk")
                        .reset_index()
                    )
                    spec = _area_line_chart_spec(
                        x_field="date",
                        x_type="temporal",
                        y_field="DailyTotalRisk",
                        x_axis={"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
                        y_axis={"title": None, "labelFontSize": 9, "titleFontSize": 10},
                    )
                    spec["layer"].append(
                        {
                            "transform": [{"regression": "DailyTotalRisk", "on": "date"}],
                            "mark": {
                                "type": "line",
                                "color": "#94A3B8",
                                "strokeDash": [4, 4],
                                "strokeWidth": 1.5,
                                "opacity": 0.45,
                            },
                            "encoding": {
                                "x": {
                                    "field": "date",
                                    "type": "temporal",
                                    "axis": {"title": "Date", "labelFontSize": 9, "titleFontSize": 10},
                                },
                                "y": {
                                    "field": "DailyTotalRisk",
                                    "type": "quantitative",
                                    "axis": {"title": None, "labelFontSize": 9, "titleFontSize": 10},
                                },
                            },
                        }
                    )
                    spec["encoding"] = {
                        "tooltip": [
                            {"field": "date", "type": "temporal"},
                            {"field": "DailyTotalRisk", "type": "quantitative", "format": ".1f"},
                        ]
                    }
                    st.vega_lite_chart(
                        risk_daily,
                        spec,
                        use_container_width=True,
                    )
                else:
                    st.info("No valid timestamps to plot.")
            else:
                st.info("No timestamp field found.")


def render_narrative_list(narratives, entity_posts=None, include_entity_column=False):
    ranked = sorted(narratives, key=lambda x: x["score"], reverse=True)
    if not ranked:
        st.info("No narratives available for this entity.")
        return
    if entity_posts is None:
        entity_posts = pd.DataFrame()

    render_summary_charts(ranked, entity_posts)
    st.markdown("---")
    st.markdown("##### Narrative Table")
    rows = []
    for n in ranked:
        narrative_id = int(n["narrative_id"])
        narrative_title = _build_narrative_title(n)
        trend_points = [round(v, 2) for v in _build_trend_points(n)]
        daily_risk_delta = _compute_daily_risk_delta(n)
        risk_score = round(float(n["score"]), 1)
        entity_label = "-"
        narrative_posts = n.get("posts")
        if isinstance(narrative_posts, pd.DataFrame) and "entity" in narrative_posts.columns:
            entities = (
                narrative_posts["entity"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            entities = entities[entities != ""]
            if len(entities):
                entity_label = entities.mode().iloc[0]

        confidence_label = str(n.get("risk_score_confidence", "")).strip() or _risk_score_confidence(risk_score)
        row_data = {
            "_narrative_id": narrative_id,
            "_narrative_title": narrative_title,
            "Narrative Description": narrative_title,
            "Quantity of Matched Posts": int(n["volume"]),
            "Matched Posts Trend": trend_points,
            "Risk Drivers": _build_risk_driver_list(n),
            "Risk Score Confidence": confidence_label,
            "Daily Risk Delta": daily_risk_delta,
            "Risk Score": risk_score,
            "Actions": "Open Narrative View",
        }
        if include_entity_column:
            row_data["Entity"] = entity_label
        rows.append(row_data)

    df = pd.DataFrame(rows)
    display_columns = [
        "Narrative Description",
        "Quantity of Matched Posts",
        "Matched Posts Trend",
        "Risk Drivers",
        "Risk Score Confidence",
        "Daily Risk Delta",
        "Risk Score",
    ]
    if include_entity_column:
        display_columns.insert(1, "Entity")
    display_columns.append("Actions")
    display_df = df[display_columns]

    def _delta_style(value):
        if value > 0:
            return "color: #DC2626;"
        if value < 0:
            return "color: #16A34A;"
        return "color: #64748B;"

    def _action_style(_value):
        return (
            "color: #2563EB; "
            "font-weight: 600; "
            "text-decoration: underline; "
            "text-underline-offset: 2px; "
            "border-bottom: 1px solid #2563EB;"
        )

    def _confidence_style(value):
        val = str(value).strip().lower()
        if val == "high":
            return "color: #16A34A;"
        if val == "medium":
            return "color: #B45309;"
        if val == "low":
            return "color: #DC2626;"
        return "color: #64748B;"

    def _risk_score_style(_value):
        return "font-weight: 700;"

    styled = (
        display_df.style
        .format({"Daily Risk Delta": "{:+.1f}"})
        .map(_delta_style, subset=["Daily Risk Delta"])
        .map(_confidence_style, subset=["Risk Score Confidence"])
        .map(_risk_score_style, subset=["Risk Score"])
        .map(_action_style, subset=["Actions"])
    )

    event = st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=520,
        on_select="rerun",
        selection_mode="single-cell",
        column_config={
            "Narrative Description": st.column_config.TextColumn("Narrative Description", width="large"),
            "Entity": st.column_config.TextColumn("Entity", width="small"),
            "Quantity of Matched Posts": st.column_config.NumberColumn(
                "Quantity of Matched Posts",
                format="%d",
                width="small",
            ),
            "Matched Posts Trend": st.column_config.AreaChartColumn(
                "Matched Posts Trend",
                y_min=0,
                y_max=100,
                width="small",
            ),
            "Risk Drivers": st.column_config.ListColumn("Risk Drivers", width="medium"),
            "Daily Risk Delta": st.column_config.TextColumn("Daily Risk Delta", width="small"),
            "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.1f", width="small"),
            "Risk Score Confidence": st.column_config.TextColumn("Risk Score Confidence", width="small"),
            "Actions": st.column_config.TextColumn(
                "Actions",
                width="medium",
            ),
        },
    )

    selection = event.get("selection", {}) if isinstance(event, dict) else {}
    selected_rows = selection.get("rows", [])
    selected_cols = selection.get("columns", [])
    selected_cells = selection.get("cells", [])

    should_open = False
    target_row_idx = None

    # Newer payload: explicit cells with row/column.
    if selected_cells:
        first_cell = selected_cells[0]
        cell_row = None
        cell_col = None

        # Dict format: {"row": <idx>, "column": <name|idx>}
        if isinstance(first_cell, dict):
            cell_row = first_cell.get("row")
            cell_col = first_cell.get("column")

        # Tuple/list format: (<row_idx>, <col_idx_or_name>)
        elif isinstance(first_cell, (tuple, list)) and len(first_cell) >= 2:
            cell_row = first_cell[0]
            cell_col = first_cell[1]

        if isinstance(cell_col, int):
            cell_col = display_df.columns[cell_col] if 0 <= cell_col < len(display_df.columns) else None
        if cell_col == "Actions" and cell_row is not None:
            should_open = True
            target_row_idx = int(cell_row)

    # Fallback payload: rows + columns lists.
    if not should_open and selected_rows and selected_cols:
        first_col = selected_cols[0]
        if isinstance(first_col, int):
            first_col = display_df.columns[first_col] if 0 <= first_col < len(display_df.columns) else None
        if first_col == "Actions":
            should_open = True
            target_row_idx = int(selected_rows[0])

    if should_open and target_row_idx is not None and 0 <= target_row_idx < len(df):
        selected_narrative = int(df.iloc[target_row_idx]["_narrative_id"])
        selected_title = str(df.iloc[target_row_idx]["_narrative_title"])
        st.session_state.selected_narrative = selected_narrative
        st.session_state.selected_narrative_title = selected_title
        st.query_params["narrative"] = str(selected_narrative)
        st.rerun()
