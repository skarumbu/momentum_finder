import re
import time
import requests as _requests
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplay
import nba_api.library.http as nba_http

# stats.nba.com blocks non-browser requests (e.g. from CI).
# Override headers to mimic a real browser request.
nba_http.STATS_HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Connection': 'keep-alive',
    'Referer': 'https://stats.nba.com/',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

# CDN endpoints are not IP-blocked and are used for training data.
_CDN_SCHEDULE_URL = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json'
_CDN_PBP_URL = 'https://cdn.nba.com/v1/api/live/nba/today/game/{game_id}/play-by-play.json'
_CDN_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
}

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


def _parse_iso_clock(clock: str) -> str:
    """Convert ISO 8601 duration 'PT10M12.00S' to 'MM:SS' time-remaining string."""
    m = re.match(r'PT(\d+)M([\d.]+)S', clock)
    if not m:
        return '00:00'
    mins = int(m.group(1))
    secs = int(float(m.group(2)))
    return f'{mins:02d}:{secs:02d}'


def _live_pbp_to_df(actions: list) -> pd.DataFrame:
    """Convert NBA CDN live play-by-play actions to the stats-style DataFrame format."""
    rows = []
    score_home = 0
    score_away = 0

    for a in actions:
        sh = a.get('scoreHome', '')
        sa = a.get('scoreAway', '')

        is_scoring = False
        if sh != '' and sa != '':
            try:
                new_home = int(sh)
                new_away = int(sa)
                if new_home != score_home or new_away != score_away:
                    score_home = new_home
                    score_away = new_away
                    is_scoring = True
            except (ValueError, TypeError):
                pass

        action_type = a.get('actionType', '')
        event_type = 9 if action_type == 'timeout' else 0

        rows.append({
            'PERIOD': int(a.get('period', 1)),
            'PCTIMESTRING': _parse_iso_clock(a.get('clock', 'PT00M00.00S')),
            # Only set SCORE on scoring plays; _compute_features_from_df uses notna() to find them
            'SCORE': f'{score_away} - {score_home}' if is_scoring else None,
            'EVENTMSGTYPE': event_type,
        })

    return pd.DataFrame(rows)


def _fetch_cdn_pbp(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play from NBA CDN (cdn.nba.com). Works for all completed games this season."""
    url = _CDN_PBP_URL.format(game_id=game_id)
    resp = _with_retry(lambda: _requests.get(url, headers=_CDN_HEADERS, timeout=30))
    resp.raise_for_status()
    actions = resp.json().get('game', {}).get('actions', [])
    if not actions:
        return pd.DataFrame()
    return _live_pbp_to_df(actions)


def process_game(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play via CDN (training path). Falls back to stats.nba.com on CDN failure."""
    try:
        df = _fetch_cdn_pbp(game_id)
    except Exception as e:
        print(f"  CDN failed for {game_id} ({e}), trying stats.nba.com...")
        df = _with_retry(lambda: playbyplay.PlayByPlay(game_id, timeout=NBA_API_TIMEOUT).get_data_frames()[0])
    return _compute_features_from_df(df, include_labels=True)


def get_latest_features(game_id: str):
    """Return the feature vector for the most recent scoring play, or None if unavailable."""
    df = _with_retry(lambda: playbyplay.PlayByPlay(game_id, timeout=NBA_API_TIMEOUT).get_data_frames()[0])
    features_df = _compute_features_from_df(df, include_labels=False)
    if features_df.empty:
        return None
    return features_df.iloc[-1][FEATURES]


def fetch_game_ids_from_cdn(season: str) -> list:
    """Get IDs of all completed games for a season from the NBA CDN schedule."""
    resp = _with_retry(lambda: _requests.get(_CDN_SCHEDULE_URL, headers=_CDN_HEADERS, timeout=30))
    resp.raise_for_status()
    schedule = resp.json().get('leagueSchedule', {})

    cdn_season = schedule.get('seasonYear', '')  # e.g. "2025-26"
    if cdn_season != season:
        raise ValueError(f"CDN schedule is for season {cdn_season}, requested {season}")

    game_ids = []
    for game_date in schedule.get('gameDates', []):
        for game in game_date.get('games', []):
            if game.get('gameStatus') == 3:  # 3 = completed
                game_ids.append(game['gameId'])
    return game_ids


def fetch_and_process_data(season='2023-24', season_type='Regular Season', games_to_process=None):
    game_ids = fetch_game_ids_from_cdn(season)
    if games_to_process is not None:
        game_ids = game_ids[:games_to_process]

    all_dfs = []
    for game_id in game_ids:
        print(f"Processing game {game_id}")
        try:
            game_df = process_game(game_id)
            if not game_df.empty:
                all_dfs.append(game_df)
        except Exception as e:
            print(f"  Skipping {game_id}: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=FEATURES + [HOME_LABEL, AWAY_LABEL])
    return pd.concat(all_dfs, ignore_index=True)


def get_game_id(team1, team2, date):
    season = _date_to_season(date)
    try:
        game_ids_meta = _fetch_cdn_schedule_meta(season)
        for game in game_ids_meta:
            if (team1 in game['matchup'] and team2 in game['matchup']
                    and game['date'] == date):
                return game['game_id']
    except Exception:
        pass
    return None


def _fetch_cdn_schedule_meta(season: str) -> list:
    """Return list of {game_id, matchup, date} dicts from the CDN schedule."""
    resp = _with_retry(lambda: _requests.get(_CDN_SCHEDULE_URL, headers=_CDN_HEADERS, timeout=30))
    resp.raise_for_status()
    schedule = resp.json().get('leagueSchedule', {})

    results = []
    for game_date in schedule.get('gameDates', []):
        for game in game_date.get('games', []):
            home = game.get('homeTeam', {}).get('teamTricode', '')
            away = game.get('awayTeam', {}).get('teamTricode', '')
            raw_date = game.get('gameDateEst', '')[:10]  # "2025-10-22"
            results.append({
                'game_id': game['gameId'],
                'matchup': f'{away} vs. {home}',
                'date': raw_date,
            })
    return results


def _date_to_season(date: str) -> str:
    """Convert a YYYY-MM-DD date string to NBA season format (e.g. '2025-26')."""
    year = int(date[:4])
    month = int(date[5:7])
    if month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    else:
        return f"{year - 1}-{str(year)[2:]}"
