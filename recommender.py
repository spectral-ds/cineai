"""
Two recommenders:
1. PopularityRecommender  — IMDB weighted rating
2. ContentRecommender     — TF-IDF + cosine similarity
"""

import ast
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# Helpers

def _names(val, n=None):
    """Extract name list from a JSON-like string column."""
    try:
        items = ast.literal_eval(val)
    except Exception:
        return []
    names = [i["name"] for i in items if isinstance(i, dict) and "name" in i]
    return names[:n] if n else names


def _director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
    except Exception:
        return ""
    for m in crew:
        if isinstance(m, dict) and m.get("job") == "Director":
            return m.get("name", "")
    return ""


def _load_data():
    base = os.path.dirname(__file__)
    movies  = pd.read_csv(os.path.join(base, "data", "movies.csv"))
    credits = pd.read_csv(os.path.join(base, "data", "credits.csv"))
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    movies["id"]  = movies["id"].astype(str)
    credits["id"] = credits["id"].astype(str)
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_c"))
    return df



# Popularity Recommender


class PopularityRecommender:
    """IMDB Weighted Rating: WR = (v/(v+m))*R + (m/(v+m))*C"""

    _PKL = os.path.join(CACHE_DIR, "pop.pkl")

    def fit(self):
        df = _load_data()
        df["genres_list"] = df["genres"].apply(_names)
        df["cast_list"]   = df["cast"].apply(lambda x: _names(x, 5))

        m = df["vote_count"].quantile(0.90)
        C = df["vote_average"].mean()

        q = df[df["vote_count"] >= m].copy()
        q["score"] = ((q["vote_count"] / (q["vote_count"] + m)) * q["vote_average"]
                    + (m / (q["vote_count"] + m)) * C)
        q.sort_values("score", ascending=False, inplace=True)

        cols = ["id", "title", "score", "vote_average", "vote_count",
                "genres_list", "cast_list", "overview", "release_date"]
        self.df = q[cols].reset_index(drop=True)
        self.genres = sorted({g for gl in self.df["genres_list"] for g in gl})

        with open(self._PKL, "wb") as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls):
        if os.path.exists(cls._PKL):
            with open(cls._PKL, "rb") as f:
                return pickle.load(f)
        print("Building PopularityRecommender...")
        return cls().fit()

    def recommend(self, n=12, genre=None):
        df = self.df
        if genre:
            df = df[df["genres_list"].apply(
                lambda g: genre.lower() in [x.lower() for x in g])]
        return df.head(n).to_dict("records")


#  Content Recommender

class ContentRecommender:
    """TF-IDF on overview+genres+cast+director, cosine similarity."""

    _PKL = os.path.join(CACHE_DIR, "cb.pkl")
    _NPY = os.path.join(CACHE_DIR, "sim.npy")

    def fit(self):
        df = _load_data()
        df["overview"]    = df["overview"].fillna("")
        df["genres_list"] = df["genres"].apply(_names)
        df["cast_list"]   = df["cast"].apply(lambda x: _names(x, 3))
        df["director"]    = df["crew"].apply(_director)
        df["cast_display"]= df["cast"].apply(lambda x: _names(x, 5))

        def soup(row):
            g = " ".join(row["genres_list"]) * 2          # genres weighted ×2
            c = " ".join(row["cast_list"])
            d = (row["director"] * 2).replace(" ", "")     # director weighted ×2
            o = row["overview"]
            return f"{o} {o} {g} {c} {d}"                 # overview ×2 too

        df["soup"] = df.apply(soup, axis=1)

        tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
        mat   = tfidf.fit_transform(df["soup"])
        sim   = cosine_similarity(mat, mat)
        np.save(self._NPY, sim)

        cols = ["id", "title", "overview", "genres_list",
                "cast_display", "director", "release_date", "vote_average"]
        self.df   = df[cols].reset_index(drop=True)
        self.idx  = pd.Series(df.index, index=df["title"].str.lower())

        # Don't store huge sim matrix in pickle
        self.sim  = None
        with open(self._PKL, "wb") as f:
            pickle.dump(self, f)
        self.sim = sim
        return self

    @classmethod
    def load(cls):
        if os.path.exists(cls._PKL) and os.path.exists(cls._NPY):
            with open(cls._PKL, "rb") as f:
                obj = pickle.load(f)
            obj.sim = np.load(cls._NPY)
            return obj
        print("Building ContentRecommender (may take ~15s)...")
        return cls().fit()

    def recommend(self, title, n=12):
        key = title.strip().lower()

        # Exact match first, then partial
        if key not in self.idx:
            matches = [t for t in self.idx.index if key in t]
            if not matches:
                raise ValueError(f"'{title}' not found. Try a partial title.")
            key = matches[0]

        i = self.idx[key]
        if isinstance(i, pd.Series):
            i = i.iloc[0]
        i = int(i)

        scores = sorted(enumerate(self.sim[i]), key=lambda x: x[1], reverse=True)
        top    = [s[0] for s in scores[1:n+1]]
        return self.df.iloc[top].to_dict("records")

    def titles(self):
        return sorted(self.df["title"].dropna().tolist())
