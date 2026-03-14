from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime
import pandas as pd
import joblib
import time
from retriever import get_game_id, get_latest_features, process_game, FEATURES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.quixotry.me", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    home_run_model = joblib.load("home_run_model.pkl")
    away_run_model = joblib.load("away_run_model.pkl")
    _models_ready = True
except FileNotFoundError:
    _models_ready = False
    print("Warning: model files not found, momentum detection disabled until retrain job runs")

_momentum_cache: dict[str, tuple[float, str | None]] = {}
MOMENTUM_CACHE_TTL = 30  # seconds


class GameRequest(BaseModel):
    team1: str
    team2: str
    date: str  # Format: YYYY-MM-DD


def get_current_momentum(game_id: str, home_team: str, away_team: str) -> str | None:
    """Returns the team name predicted to go on a run, or None."""
    if not _models_ready:
        return None

    now = time.time()
    if game_id in _momentum_cache:
        cached_at, result = _momentum_cache[game_id]
        if now - cached_at < MOMENTUM_CACHE_TTL:
            return result

    try:
        features = get_latest_features(game_id)
        if features is None:
            result = None
        else:
            X = pd.DataFrame([features])
            home_prob = home_run_model.predict_proba(X)[0][1]
            away_prob = away_run_model.predict_proba(X)[0][1]

            if home_prob >= 0.5 and home_prob >= away_prob:
                result = home_team
            elif away_prob >= 0.5:
                result = away_team
            else:
                result = None
    except Exception:
        result = None

    _momentum_cache[game_id] = (now, result)
    return result


@app.post("/get-momentum")
async def get_momentum(request: GameRequest):
    game_id = get_game_id(request.team1, request.team2, request.date)
    if not game_id:
        raise HTTPException(status_code=404, detail="Game not found")

    try:
        df = process_game(game_id)
        # Return plays where a run was detected (home or away)
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
            if is_live:
                momentum_team = get_current_momentum(g['gameId'], home['teamName'], away['teamName'])

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
                "momentumTeam": momentum_team,
            }
            game_list.append(game_obj)

        return {"games": game_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
