from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime
import os
import pandas as pd
import joblib
import time
import logging
import json
import uuid
from azure.storage.blob import BlobServiceClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from retriever import get_game_id, get_latest_features, process_game, FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("momentum-finder")

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        rid = str(uuid.uuid4())[:8]
        try:
            response = await call_next(request)
            logger.info(json.dumps({
                "event": "request",
                "service": "momentum-finder",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.time() - start) * 1000, 1),
                "request_id": rid,
            }))
            return response
        except Exception as exc:
            logger.error(json.dumps({
                "event": "error",
                "service": "momentum-finder",
                "method": request.method,
                "path": request.url.path,
                "error": str(exc),
                "duration_ms": round((time.time() - start) * 1000, 1),
                "request_id": rid,
            }))
            raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.quixotry.me", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)


def _download_models() -> bool:
    """Download model files from Azure Blob Storage. Returns True if successful."""
    conn_str = os.getenv("MODEL_STORAGE_CONNECTION_STRING")
    if not conn_str:
        logger.warning("MODEL_STORAGE_CONNECTION_STRING not set, skipping model download")
        return False
    try:
        container = BlobServiceClient.from_connection_string(conn_str).get_container_client("models")
        for blob_name in ("home_run_model.pkl", "away_run_model.pkl", "win_prob_model.pkl"):
            with open(blob_name, "wb") as f:
                f.write(container.download_blob(blob_name).readall())
        return True
    except Exception as e:
        logger.error(f"Could not download models from blob storage: {e}")
        return False


_download_models()

try:
    home_run_model = joblib.load("home_run_model.pkl")
    away_run_model = joblib.load("away_run_model.pkl")
    win_prob_model = joblib.load("win_prob_model.pkl")
    _models_ready = True
except FileNotFoundError:
    _models_ready = False
    logger.warning("Model files not found, momentum detection disabled until retrain job runs")

_momentum_cache: dict[str, tuple[float, str | None, float | None]] = {}
MOMENTUM_CACHE_TTL = 30  # seconds


class GameRequest(BaseModel):
    team1: str
    team2: str
    date: str  # Format: YYYY-MM-DD


def get_current_momentum(game_id: str, home_team: str, away_team: str) -> tuple[str | None, float | None]:
    """Returns (momentum_team, home_win_probability). Either value may be None on error."""
    if not _models_ready:
        return None, None

    now = time.time()
    if game_id in _momentum_cache:
        cached_at, momentum_team, home_win_prob = _momentum_cache[game_id]
        if now - cached_at < MOMENTUM_CACHE_TTL:
            return momentum_team, home_win_prob

    try:
        features = get_latest_features(game_id)
        if features is None:
            momentum_team, home_win_prob = None, None
        else:
            X = pd.DataFrame([features])
            home_run_prob = home_run_model.predict_proba(X)[0][1]
            away_run_prob = away_run_model.predict_proba(X)[0][1]
            home_win_prob = float(win_prob_model.predict_proba(X)[0][1])

            if home_run_prob >= 0.5 and home_run_prob >= away_run_prob:
                momentum_team = home_team
            elif away_run_prob >= 0.5:
                momentum_team = away_team
            else:
                momentum_team = None
    except Exception:
        momentum_team, home_win_prob = None, None

    _momentum_cache[game_id] = (now, momentum_team, home_win_prob)
    return momentum_team, home_win_prob


@app.post("/get-momentum")
async def get_momentum(request: GameRequest):
    game_id = get_game_id(request.team1, request.team2, request.date)
    if not game_id:
        raise HTTPException(status_code=404, detail="Game not found")

    try:
        df = process_game(game_id)
        shifts = df[(df['home_run_coming'] == 1) | (df['away_run_coming'] == 1)]
        return {"game_id": game_id, "momentum_shifts": shifts.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-current-games")
async def get_current_games():
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()

        game_list = []
        for g in games:
            home = g['homeTeam']
            away = g['awayTeam']
            is_live = g.get('gameStatus') == 2

            momentum_team = None
            home_win_prob = None
            if is_live:
                momentum_team, home_win_prob = get_current_momentum(g['gameId'], home['teamName'], away['teamName'])

            game_obj = {
                "gameId": g['gameId'],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "team1": away['teamName'],
                "team2": home['teamName'],
                "status": g['gameStatusText'],
                "score": {
                    away['teamName']: away.get('score', 0),
                    home['teamName']: home.get('score', 0)
                },
                "isLive": is_live,
                "momentumTeam": momentum_team,
                "winProbability": {
                    home['teamName']: round(home_win_prob, 2),
                    away['teamName']: round(1 - home_win_prob, 2),
                } if home_win_prob is not None else None,
            }
            game_list.append(game_obj)

        return {"games": game_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
