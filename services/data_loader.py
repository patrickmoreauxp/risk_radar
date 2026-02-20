import pandas as pd
import json

def load_data():
    posts = pd.read_json("data/posts.jsonl", lines=True)
    authors = pd.read_csv("data/authors.csv")
    entities = pd.read_csv("data/entities_seed.csv")

    return posts, authors, entities
