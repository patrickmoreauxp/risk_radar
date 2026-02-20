import streamlit as st

def render_entity_overview(entity_name, posts):
    st.subheader(f"Entity: {entity_name}")

    st.metric("Matched Posts", len(posts))

    avg_conf = posts["resolution_confidence"].mean() if len(posts) else 0.0
    st.metric("Avg Resolution Confidence", round(avg_conf, 2))
