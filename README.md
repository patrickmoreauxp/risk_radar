# RiskRadar Prototype

This repository runs a Streamlit app that preprocesses social data into entity-linked risk narratives, then renders a dashboard + narrative detail views.

![RiskRadar app screenshot](screenshots/Screenshot%202026-02-20%20at%2009.36.58.webp)

## 1) First-time setup

From the repo root:

```bash
python3 -m venv storyful_env
source storyful_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional but recommended model/data setup:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon
```

## 2) Add your OpenAI key

Create/update `.env` in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

The app reads this key from environment only (no UI key input).

## 3) Place raw input data files

Put these files in the `data/` folder:

- `data/posts.jsonl` (JSONL, one post per line)
- `data/authors.csv`
- `data/entities_seed.csv`

Expected loading path is defined in `services/data_loader.py`.

## 4) Run the app

```bash
source storyful_env/bin/activate
streamlit run app.py
```

## 5) What happens when the app starts

- If processed artifacts are missing, the app runs the preprocessing pipeline and shows step-by-step progress.
- Artifacts are written to `data/processed/` (for example `clustered_posts.jsonl` and `narratives.json`).
- On normal refresh, the app loads existing processed artifacts and should not re-run the pipeline.
- Use the **Rebuild data pipeline** button in the sidebar when you explicitly want to regenerate artifacts.

## 6) Common workflow

1. Update raw files under `data/`
2. Press **Rebuild data pipeline** in the app
3. Review dashboard and narrative detail views
4. Refresh/reopen app (it should load from processed artifacts)

## 7) Screenshots

UI screenshots are stored in the `screenshots/` folder.

