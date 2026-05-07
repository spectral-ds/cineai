# CineAI — Flask App

import os, json, threading, requests
from flask import Flask, render_template, request, jsonify
from recommender import PopularityRecommender, ContentRecommender

BASE        = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(BASE, "cache")
POSTER_JSON = os.path.join(CACHE_DIR, "poster_paths.json")

TMDB_KEY  = os.getenv("TMDB_API_KEY", "33646a32600aa482e4046ad6f3d701aa")
TMDB_IMG  = "https://image.tmdb.org/t/p/w500"
TMDB_URL  = "https://api.themoviedb.org/3/movie/{id}"

os.makedirs(CACHE_DIR, exist_ok=True)
app = Flask(__name__)

# Recommenders

print("Loading recommenders...")
pop = PopularityRecommender.load()
cb  = ContentRecommender.load()
print(f"Ready. {len(cb.titles())} movies loaded.")

# Poster cache

_posters: dict = {}
if os.path.exists(POSTER_JSON):
    try:
        with open(POSTER_JSON) as f:
            _posters = json.load(f)
        n = sum(1 for v in _posters.values() if v)
        print(f"Poster cache: {n}/{len(_posters)} URLs loaded.")
    except Exception:
        _posters = {}

_lock = threading.Lock()

def _save():
    with _lock:
        try:
            with open(POSTER_JSON, "w") as f:
                json.dump(_posters, f)
        except Exception:
            pass

def _fetch_from_tmdb(mid: str) -> str:
    """Direct TMDB lookup by movie ID. Returns CDN URL or empty string."""
    try:
        r = requests.get(
            TMDB_URL.format(id=mid),
            params={"api_key": TMDB_KEY},
            timeout=8,
        )
        if r.status_code == 200:
            path = r.json().get("poster_path") or ""
            return (TMDB_IMG + path) if path else ""
    except Exception:
        pass
    return ""

def get_poster_url(mid: str) -> str:
    """
    Return cached poster URL for a TMDB movie ID.
    If not cached yet, fetch from TMDB and store.
    Returns empty string (not placeholder) so browser can decide the fallback.
    """
    mid = str(mid).strip()
    if not mid or mid == "nan":
        return ""
    if mid in _posters:
        return _posters[mid]          # "" is valid (means no poster on TMDB)
    url = _fetch_from_tmdb(mid)
    _posters[mid] = url
    threading.Thread(target=_save, daemon=True).start()
    return url

def with_posters(movies: list) -> list:
    """Used only by JSON API endpoints."""
    for m in movies:
        url = get_poster_url(str(m.get("id", "")))
        m["poster"] = url if url else "/static/img/no_poster.png"
    return movies

# ── Routes ────────────────────────────────────────────────────────────────── #

@app.route("/")
def index():
    return render_template("index.html",
                           genres=pop.genres,
                           titles=cb.titles()[:300])

@app.route("/popular")
def popular():
    genre  = request.args.get("genre") or None
    movies = pop.recommend(n=12, genre=genre)
    return render_template("popular.html",
                           movies=movies,
                           genres=pop.genres,
                           active_genre=genre or "All")

@app.route("/similar")
def similar():
    title  = request.args.get("title", "").strip()
    movies, error = [], None
    if title:
        try:
            movies = cb.recommend(title, n=12)
        except ValueError as e:
            error = str(e)
    return render_template("similar.html",
                           movies=movies, query=title, error=error)

# ── API ────────────────────────────────────────────────────────────────────── #

@app.route("/api/poster/<movie_id>")
def api_poster(movie_id):
    """
    Fallback poster endpoint — browser calls this if direct TMDB fetch fails.
    Returns {"url": "https://..."} or {"url": ""} if no poster available.
    """
    url = get_poster_url(movie_id)
    # Add CORS header so browser can call this cross-origin if needed
    resp = jsonify({"url": url})
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp

@app.route("/api/popular")
def api_popular():
    genre = request.args.get("genre") or None
    return jsonify(with_posters(pop.recommend(n=12, genre=genre)))

@app.route("/api/similar")
def api_similar():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "title required"}), 400
    try:
        return jsonify(with_posters(cb.recommend(title, n=12)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

@app.route("/api/titles")
def api_titles():
    return jsonify(cb.titles())

@app.route("/health")
def health():
    cached = sum(1 for v in _posters.values() if v)
    return jsonify({"status": "ok", "movies": len(cb.titles()),
                    "posters_cached": cached})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
