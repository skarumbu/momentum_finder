from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nba_api.stats.endpoints import playbyplay
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from retriever import convert_time_to_seconds, get_game_id

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.quixotry.me", "http://localhost:3000"],  # Add your domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = joblib.load("momentum_model.pkl")

class GameRequest(BaseModel):
    team1: str
    team2: str
    date: str  # Format: YYYY-MM-DD

def get_momentum_shifts(game_id):
    df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
    
    df_scores = df.dropna(subset=['SCORE']).copy()
    df_scores[['Visitor_Score', 'Home_Score']] = df_scores['SCORE'].str.split(' - ', expand=True).astype(int)
    
    df_scores['Home_Lead'] = df_scores['Home_Score'] - df_scores['Visitor_Score']
    df_scores['Lead_Change'] = df_scores['Home_Lead'].diff().fillna(0)
    df_scores['Score_Change'] = df_scores[['Home_Score', 'Visitor_Score']].diff().fillna(0).sum(axis=1)
    df_scores['Time_Since_Last_Score'] = df['PCTIMESTRING'].apply(convert_time_to_seconds).diff(-1).fillna(0).abs()

    features = ['Home_Lead', 'Lead_Change', 'Score_Change', "Time_Since_Last_Score"]
    
    df_scores['Momentum_Shift'] = model.predict(df_scores[features].fillna(0))

    momentum_moments = df_scores[df_scores['Momentum_Shift'] == 1][['EVENTNUM', 'PCTIMESTRING', 'SCORE']]
    return momentum_moments.to_dict(orient='records')

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
            game_obj = {
                "gameId": g['gameId'],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "team1": away['teamName'],
                "team2": home['teamName'],
                "status": g['gameStatusText'],
                "score": {
                    away['teamName']: away.get('score', 0),
                    home['teamName']: home.get('score', 0)
                }
            }
            game_list.append(game_obj)

        return {"games": game_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}