import json
import streamlit as st

def save_feedback(data):
    with open("outputs/feedback.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

def render_feedback(narrative_id):
    rating = st.slider("Risk too low / high?", 1, 5, 3)

    if st.button("Submit Feedback"):
        save_feedback({
            "narrative_id": narrative_id,
            "rating": rating
        })
        st.success("Feedback saved")
