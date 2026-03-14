from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.stats.endpoints import playbyplay
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import time
from retriever import convert_time_to_seconds, get_game_id

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.quixotry.me", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = joblib.load("momentum_model.pkl")

# Cache momentum results per game_id to avoid re-running play-by-play on every poll.
# Entries are (timestamp, result) tuples; results older than MOMENTUM_CACHE_TTL are recomputed.
_momentum_cache: dict[str, tuple[float, str | None]] = {}
MOMENTUM_CACHE_TTL = 30  # seconds


class GameRequest(BaseModel):
    team1: str
    team2: str
    date: str  # Format: YYYY-MM-DD


def _build_momentum_features(game_id: str) -> pd.DataFrame:
    df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
    df_scores = df.dropna(subset=['SCORE']).copy()
    df_scores[['Visitor_Score', 'Home_Score']] = df_scores['SCORE'].str.split(' - ', expand=True).astype(int)
    df_scores['Home_Lead'] = df_scores['Home_Score'] - df_scores['Visitor_Score']
    df_scores['Lead_Change'] = df_scores['Home_Lead'].diff().fillna(0)
    df_scores['Score_Change'] = df_scores[['Home_Score', 'Visitor_Score']].diff().fillna(0).sum(axis=1)
    df_scores['Time_Since_Last_Score'] = df['PCTIMESTRING'].apply(convert_time_to_seconds).diff(-1).fillna(0).abs()
    return df_scores


def get_momentum_shifts(game_id):
    df_scores = _build_momentum_features(game_id)
    features = ['Home_Lead', 'Lead_Change', 'Score_Change', "Time_Since_Last_Score"]
    df_scores['Momentum_Shift'] = model.predict(df_scores[features].fillna(0))
    momentum_moments = df_scores[df_scores['Momentum_Shift'] == 1][['EVENTNUM', 'PCTIMESTRING', 'SCORE']]
    return momentum_moments.to_dict(orient='records')


def get_current_momentum(game_id: str, home_team: str, away_team: str) -> str | None:
    """Returns the team name that currently has momentum, or None."""
    now = time.time()
    if game_id in _momentum_cache:
        cached_at, result = _momentum_cache[game_id]
        if now - cached_at < MOMENTUM_CACHE_TTL:
            return result

    try:
        df_scores = _build_momentum_features(game_id)
        if df_scores.empty:
            _momentum_cache[game_id] = (now, None)
            return None

        features = ['Home_Lead', 'Lead_Change', 'Score_Change', 'Time_Since_Last_Score']
        df_scores['Momentum_Shift'] = model.predict(df_scores[features].fillna(0))

        shifts = df_scores[df_scores['Momentum_Shift'] == 1]
        if shifts.empty:
            _momentum_cache[game_id] = (now, None)
            return None

        latest = shifts.iloc[-1]
        result = home_team if latest['Home_Lead'] > 0 else away_team
    except Exception:
        result = None

    _momentum_cache[game_id] = (now, result)
    return result


@app.post("/get-momentum")
async def get_momentum(request: GameRequest):
    game_id = get_game_id(request.team1, request.team2, request.date)
    if not game_id:
        raise HTTPException(status_code=404, detail="Game not found")

    momentum_shifts = get_momentum_shifts(game_id)
    return {"game_id": game_id, "momentum_shifts": momentum_shifts}


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
