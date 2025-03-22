from nba_api.stats.endpoints import playbyplay
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def convert_time_to_seconds(time_str):
    """Convert time from MM:SS format to seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def fetch_and_process_data(season='2023-24', season_type='Regular Season', games_to_process=10):
    """Fetch play-by-play data for multiple games, process it, and return a DataFrame."""
    
    # Fetch game data
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=None, 
        season_nullable=season,
        season_type_nullable=season_type
    )  

    games_dict = gamefinder.get_normalized_dict()
    games = games_dict['LeagueGameFinderResults']
    games_to_process = games[:games_to_process]
    
    all_games_df = pd.DataFrame()

    for game in games_to_process:
        game_id = game['GAME_ID']
        print(f"Processing game {game_id}")
        
        # Fetch play-by-play data
        df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
        
        # Clean the data (handling scores and calculating momentum shifts)
        df_scores = df.dropna(subset=['SCORE']).copy()
        df_scores[['Visitor_Score', 'Home_Score']] = df_scores['SCORE'].str.split(' - ', expand=True).astype(int)
        
        df['Home_Lead'] = df_scores['Home_Score'] - df_scores['Visitor_Score']
        df['Lead_Change'] = df['Home_Lead'].diff().fillna(0)
        df['Score_Change'] = df_scores[['Home_Score', 'Visitor_Score']].diff().fillna(0).sum(axis=1)
        df['Time_Since_Last_Score'] = df['PCTIMESTRING'].apply(convert_time_to_seconds).diff(-1).fillna(0).abs()
        
        df['Momentum_Shift'] = ((df['Score_Change'].rolling(window=4).sum() >= 6) &
                                (df['Time_Since_Last_Score'].rolling(window=4).sum() <= 120)).astype(int)
        
        df['GAME_ID'] = game_id
        
        # Append this game's data to the all_games_df DataFrame
        all_games_df = pd.concat([all_games_df, df], ignore_index=True)

    return all_games_df
