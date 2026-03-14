import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplay, leaguegamefinder

NBA_API_TIMEOUT = 60  # seconds


def _with_retry(fn, retries=3, backoff=5):
    """Call fn(), retrying up to `retries` times on exception with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  Retrying after {wait}s (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait)

FEATURES = [
    'quarter',
    'time_remaining_seconds',
    'score_differential',
    'total_score',
    'home_points_last_2min',
    'away_points_last_2min',
    'home_points_last_4min',
    'away_points_last_4min',
    'time_since_home_scored',
    'time_since_away_scored',
    'timeout_last_2min',
    'lead_changes_last_5min',
]
HOME_LABEL = 'home_run_coming'
AWAY_LABEL = 'away_run_coming'


def convert_time_to_seconds(time_str):
    """Convert time from MM:SS format to seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds


def _to_game_time(period: int, pctimestring: str) -> float:
    """Convert period + clock to total seconds elapsed in the game."""
    remaining = convert_time_to_seconds(pctimestring)
    period_duration = 720 if period <= 4 else 300
    elapsed_before = sum(720 if p <= 4 else 300 for p in range(1, period))
    return float(elapsed_before + (period_duration - remaining))


def _compute_features_from_df(df: pd.DataFrame, include_labels: bool = False) -> pd.DataFrame:
    """Compute features (and optionally forward-looking labels) from raw play-by-play data."""
    df = df.copy().reset_index(drop=True)
    df['game_time'] = df.apply(
        lambda r: _to_game_time(int(r['PERIOD']), str(r['PCTIMESTRING'])), axis=1
    )

    # Parse cumulative scores ("away - home" format)
    scores = df['SCORE'].str.split(' - ', expand=True)
    df['away_score'] = pd.to_numeric(scores[0], errors='coerce').ffill().fillna(0)
    df['home_score'] = pd.to_numeric(scores[1], errors='coerce').ffill().fillna(0)

    # Points scored on each event
    df['home_pts'] = df['home_score'].diff().clip(lower=0).fillna(0)
    df['away_pts'] = df['away_score'].diff().clip(lower=0).fillna(0)
    df['is_timeout'] = (df['EVENTMSGTYPE'] == 9).astype(int)

    times = df['game_time'].values
    home_pts = df['home_pts'].values
    away_pts = df['away_pts'].values
    home_score_arr = df['home_score'].values
    away_score_arr = df['away_score'].values
    timeouts = df['is_timeout'].values

    # Prefix cumulative sums: cum[i+1] = sum of first i+1 elements
    cum_home = np.concatenate([[0], np.cumsum(home_pts)])
    cum_away = np.concatenate([[0], np.cumsum(away_pts)])
    cum_timeout = np.concatenate([[0], np.cumsum(timeouts)])

    def window_sum(cum, pos, window_back):
        lo = np.searchsorted(times, times[pos] - window_back, side='left')
        hi = pos + 1
        return float(cum[hi] - cum[lo])

    scoring_positions = np.where(df['SCORE'].notna())[0]

    rows = []
    for pos in scoring_positions:
        t = times[pos]
        period = int(df.iloc[pos]['PERIOD'])
        period_duration = 720 if period <= 4 else 300
        elapsed_before = sum(720 if p <= 4 else 300 for p in range(1, period))
        time_remaining = float(elapsed_before + period_duration - t)

        # Time since each team last scored
        home_scored_times = times[(home_pts > 0) & (times <= t)]
        away_scored_times = times[(away_pts > 0) & (times <= t)]
        time_since_home = float(t - home_scored_times[-1]) if len(home_scored_times) > 0 else float(t)
        time_since_away = float(t - away_scored_times[-1]) if len(away_scored_times) > 0 else float(t)

        # Lead changes in last 5 minutes
        lo5 = int(np.searchsorted(times, t - 300, side='left'))
        lead_window = home_score_arr[lo5:pos + 1] - away_score_arr[lo5:pos + 1]
        lead_changes = int(np.sum(np.diff(np.sign(lead_window)) != 0)) if len(lead_window) > 1 else 0

        row = {
            'quarter': period,
            'time_remaining_seconds': time_remaining,
            'score_differential': float(home_score_arr[pos] - away_score_arr[pos]),
            'total_score': float(home_score_arr[pos] + away_score_arr[pos]),
            'home_points_last_2min': window_sum(cum_home, pos, 120),
            'away_points_last_2min': window_sum(cum_away, pos, 120),
            'home_points_last_4min': window_sum(cum_home, pos, 240),
            'away_points_last_4min': window_sum(cum_away, pos, 240),
            'time_since_home_scored': time_since_home,
            'time_since_away_scored': time_since_away,
            'timeout_last_2min': window_sum(cum_timeout, pos, 120),
            'lead_changes_last_5min': lead_changes,
        }

        if include_labels:
            lo_fwd = int(pos + 1)
            hi_fwd = int(np.searchsorted(times, t + 180, side='right'))
            fh = float(cum_home[hi_fwd] - cum_home[lo_fwd])
            fa = float(cum_away[hi_fwd] - cum_away[lo_fwd])
            row[HOME_LABEL] = int(fh - fa >= 8)
            row[AWAY_LABEL] = int(fa - fh >= 8)

        rows.append(row)

    return pd.DataFrame(rows)


def process_game(game_id: str) -> pd.DataFrame:
    df = _with_retry(lambda: playbyplay.PlayByPlay(game_id, timeout=NBA_API_TIMEOUT).get_data_frames()[0])
    return _compute_features_from_df(df, include_labels=True)


def get_latest_features(game_id: str):
    """Return the feature vector for the most recent scoring play, or None if unavailable."""
    df = _with_retry(lambda: playbyplay.PlayByPlay(game_id, timeout=NBA_API_TIMEOUT).get_data_frames()[0])
    features_df = _compute_features_from_df(df, include_labels=False)
    if features_df.empty:
        return None
    return features_df.iloc[-1][FEATURES]


def fetch_and_process_data(season='2023-24', season_type='Regular Season', games_to_process=None):
    gamefinder = _with_retry(lambda: leaguegamefinder.LeagueGameFinder(
        team_id_nullable=None,
        season_nullable=season,
        season_type_nullable=season_type,
        timeout=NBA_API_TIMEOUT,
    ))
    games = gamefinder.get_normalized_dict()['LeagueGameFinderResults']
    games = games[:games_to_process] if games_to_process is not None else games

    # Deduplicate: LeagueGameFinder returns two entries per game (one per team)
    seen = set()
    unique_games = []
    for g in games:
        if g['GAME_ID'] not in seen:
            seen.add(g['GAME_ID'])
            unique_games.append(g)

    all_dfs = []
    for game in unique_games:
        game_id = game['GAME_ID']
        print(f"Processing game {game_id}")
        try:
            game_df = process_game(game_id)
            all_dfs.append(game_df)
        except Exception as e:
            print(f"  Skipping {game_id}: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=FEATURES + [HOME_LABEL, AWAY_LABEL])
    return pd.concat(all_dfs, ignore_index=True)


def get_game_id(team1, team2, date):
    gamefinder = _with_retry(lambda: leaguegamefinder.LeagueGameFinder(
        season_nullable="2023-24",
        season_type_nullable="Regular Season",
        timeout=NBA_API_TIMEOUT,
    ))
    games = gamefinder.get_normalized_dict()["LeagueGameFinderResults"]
    for game in games:
        if team1 in game["MATCHUP"] and team2 in game["MATCHUP"] and game["GAME_DATE"] == date:
            return game["GAME_ID"]
    return None
