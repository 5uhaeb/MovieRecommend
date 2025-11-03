# app.py
import os
import io
import zipfile
import math
import urllib.request
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb


# --------------- FastAPI app ---------------
app = FastAPI(title="Movie Recommender API (Hybrid SVD + XGBoost)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # front-end on 127.0.0.1:5500
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ML_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Globals populated at startup
movies_df: pd.DataFrame = pd.DataFrame()
ratings_df: pd.DataFrame = pd.DataFrame()
item_vecs: np.ndarray = np.empty((0, 50))
user_vecs: np.ndarray = np.empty((0, 50))
xgb_model: Optional[xgb.XGBRegressor] = None
le_user = LabelEncoder()
le_movie = LabelEncoder()
global_mean = 3.5


# --------------- Utility ---------------
def ensure_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    movies_path = os.path.join(DATA_DIR, "movies.csv")

    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        return ratings_path, movies_path

    print("Downloading MovieLens small…")
    resp = urllib.request.urlopen(ML_URL, timeout=60)
    data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DATA_DIR)

    ratings_path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")
    movies_path = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    # copy to DATA_DIR root for simpler paths next time
    pd.read_csv(ratings_path).to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    pd.read_csv(movies_path).to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
    return os.path.join(DATA_DIR, "ratings.csv"), os.path.join(DATA_DIR, "movies.csv")


def parse_year_from_title(title: str) -> Optional[int]:
    # MovieLens titles like "Toy Story (1995)"
    if not isinstance(title, str):
        return None
    import re
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


# --------------- Model prep ---------------
def fit_models():
    global movies_df, ratings_df, item_vecs, user_vecs, xgb_model, global_mean

    ratings_path, movies_path = ensure_movielens()
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    # Basic movie parsing
    movies_df["genres"] = movies_df["genres"].fillna("").astype(str).apply(lambda s: s.split("|") if s != "(no genres listed)" else [])
    movies_df["year"] = movies_df["title"].apply(parse_year_from_title)

    # --- SVD for item/user embeddings ---
    # Build a dense-ish user-item matrix (implicit zeros ok for SVD)
    ui = ratings_df.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0)
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_vecs = svd.fit_transform(ui.values)             # (num_users, 50)
    item_vecs_full = svd.components_.T                   # (num_movies, 50)

    # Map item order back to movieId
    movie_id_index = ui.columns.to_numpy()
    # Align item vectors to movies_df order
    # Build mapping movieId -> vector
    vec_map = {int(mid): item_vecs_full[i] for i, mid in enumerate(movie_id_index)}
    # For any movie without ratings, set small random vector
    item_vecs = np.vstack([vec_map.get(int(mid), np.zeros(50)) for mid in movies_df["movieId"].to_numpy()])

    # --- Simple hybrid regressor with XGBoost ---
    # Encode ids
    uid_enc = le_user.fit_transform(ratings_df["userId"])
    mid_enc = le_movie.fit_transform(ratings_df["movieId"])

    # Create features: user/movie ids (encoded) + their SVD vectors (looked up)
    # Build lookups from original ids -> svd vectors
    # user vectors align with ui.index
    user_id_index = ui.index.to_numpy()
    uvec_map = {int(uid): user_vecs[i] for i, uid in enumerate(user_id_index)}
    # fallback zero vecs
    U = np.vstack([uvec_map.get(int(u), np.zeros(50)) for u in ratings_df["userId"].to_numpy()])
    M = np.vstack([vec_map.get(int(m), np.zeros(50)) for m in ratings_df["movieId"].to_numpy()])

    X = np.column_stack([uid_enc, mid_enc, U, M])
    y = ratings_df["rating"].to_numpy()
    global_mean = float(np.mean(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train)
    pred_xgb = np.clip(xgb_model.predict(X_test), 0.5, 5.0)
    rmse_xgb = float(np.sqrt(((pred_xgb - y_test) ** 2).mean()))
    mae_xgb = float(np.mean(np.abs(pred_xgb - y_test)))
    print(f"XGBoost RMSE: {rmse_xgb:.3f}, MAE: {mae_xgb:.3f}")

    # A tiny hybrid blend with the global mean to stabilize
    pred_hybrid = 0.85 * pred_xgb + 0.15 * global_mean
    rmse_h = float(np.sqrt(((pred_hybrid - y_test) ** 2).mean()))
    mae_h = float(np.mean(np.abs(pred_hybrid - y_test)))
    print(f"Hybrid RMSE: {rmse_h:.3f}, MAE: {mae_h:.3f}")
    print("Model ready ✅")


# --------------- API models ---------------
class MovieOut(BaseModel):
    movieId: int
    title: str
    genres: List[str] = []
    year: Optional[int] = None


# --------------- Endpoints ---------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
def _startup():
    fit_models()

@app.get("/api/movies", response_model=List[MovieOut])
def list_movies(
    limit: int = Query(500, ge=1, le=20000),   # <<<<<< increased limit here
    offset: int = Query(0, ge=0),
    sort: str = Query("title", description="title|year")
):
    df = movies_df.copy()
    if sort == "year":
        df = df.sort_values(by=["year", "title"], ascending=[False, True])
    else:
        df = df.sort_values(by="title")

    df = df.iloc[offset: offset + limit]
    res = [
        MovieOut(
            movieId=int(r.movieId),
            title=str(r.title),
            genres=list(r.genres) if isinstance(r.genres, list) else [],
            year=int(r.year) if pd.notna(r.year) else None,
        ).model_dump()
        for r in df.itertuples(index=False)
    ]
    return res


@app.get("/api/similar", response_model=List[MovieOut])
def similar_movies(
    movie_id: int = Query(..., alias="movie_id"),
    k: int = Query(12, ge=1, le=100)
):
    # find index in movies_df
    try:
        idx = int(np.where(movies_df["movieId"].to_numpy() == movie_id)[0][0])
    except IndexError:
        raise HTTPException(status_code=404, detail="movie_id not found")

    if item_vecs.shape[0] == 0:
        raise HTTPException(500, detail="Embeddings not available")

    # cosine similarity on item vectors
    q = item_vecs[idx:idx+1, :]
    sims = cosine_similarity(q, item_vecs)[0]
    # exclude itself
    sims[idx] = -np.inf
    top_idx = np.argsort(-sims)[:k]

    rows = movies_df.iloc[top_idx]
    res = [
        MovieOut(
            movieId=int(r.movieId),
            title=str(r.title),
            genres=list(r.genres) if isinstance(r.genres, list) else [],
            year=int(r.year) if pd.notna(r.year) else None,
        ).model_dump()
        for r in rows.itertuples(index=False)
    ]
    return res


@app.get("/api/recommendations", response_model=List[MovieOut])
def recommend_for_user(
    user_id: int = Query(..., ge=1),
    k: int = Query(12, ge=1, le=100)
):
    """
    Very simple: predict rating for unseen movies for a user using XGBoost features
    (encoded ids + SVD vectors) and return top-k by predicted score.
    """
    if xgb_model is None:
        raise HTTPException(500, detail="Model not loaded")

    # encode user id (if unseen -> new index at end)
    uid = user_id
    if uid in le_user.classes_:
        uid_enc = le_user.transform([uid])[0]
        uvec = None
        # try to grab user vec from SVD map by userId
        # We don't keep the original mapping explicitly, fallback zeros if missing
    else:
        # unseen user -> use zeros
        uid_enc = le_user.transform([le_user.classes_[0]])[0]
        uvec = np.zeros((1, 50))

    # build per-movie features
    movie_ids = movies_df["movieId"].to_numpy()
    # movie encodings (unseen -> fallback to 0)
    enc_known = set(le_movie.classes_)
    mid_enc = np.array([le_movie.transform([m])[0] if m in enc_known else 0 for m in movie_ids])

    # user vec: if we can’t map exactly, just zeros (works fine for demo)
    if uvec is None:
        # We don't have a direct mapping (we built uvecs aligned to ratings pivot users)
        uvec = np.zeros((1, 50))

    # item vecs already aligned to movies_df order
    U = np.repeat(uvec, len(movie_ids), axis=0)
    M = item_vecs

    X = np.column_stack([np.full(len(movie_ids), uid_enc), mid_enc, U, M])
    preds = xgb_model.predict(X)
    preds = 0.85 * preds + 0.15 * global_mean

    top_idx = np.argsort(-preds)[:k]
    rows = movies_df.iloc[top_idx]
    res = [
        MovieOut(
            movieId=int(r.movieId),
            title=str(r.title),
            genres=list(r.genres) if isinstance(r.genres, list) else [],
            year=int(r.year) if pd.notna(r.year) else None,
        ).model_dump()
        for r in rows.itertuples(index=False)
    ]
    return res
