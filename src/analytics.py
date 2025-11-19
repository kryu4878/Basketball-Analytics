"""Utility functions for working with the NBA analytics data."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# -------------------------------------------------------------------
# Team name / code helpers
# -------------------------------------------------------------------

# Basketball-Reference style team codes → pretty names
TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Full name → code (e.g. "Atlanta Hawks" → "ATL")
TEAM_FULL_TO_CODE = {full: code for code, full in TEAM_NAME_MAP.items()}
# Also handle "Atlanta Hawks (ATL)" style strings
TEAM_FULL_TO_CODE.update({f"{full} ({code})": code for code, full in TEAM_NAME_MAP.items()})

# Canonicalize alternative codes (PHX → PHO, BKN → BRK, etc.)
TEAM_CODE_CANONICAL = {
    "PHX": "PHO",
    "PHO": "PHO",
    "BKN": "BRK",
    "BRK": "BRK",
    "CHA": "CHO",
    "CHO": "CHO",
    "NOH": "NOP",
    "NOP": "NOP",
    # everything else: just use itself
}


def canonical_team_code(code: str) -> str:
    code = str(code).strip().upper()
    return TEAM_CODE_CANONICAL.get(code, code)


def normalize_team_str(value: str) -> str:
    """
    Take anything that looks like a team (code, full name, 'Name (CODE)' etc.)
    and return a canonical 3-letter code like 'ATL' or 'PHO'.
    """
    s = str(value).strip()

    # Handle patterns like "Atlanta Hawks (ATL)"
    if "(" in s and ")" in s:
        before_paren = s.split("(", 1)[0].strip()
        code_in_paren = s.split("(", 1)[1].split(")", 1)[0].strip()
        if 2 <= len(code_in_paren) <= 4:
            s = code_in_paren
        else:
            s = before_paren

    # Full team name?
    if s in TEAM_FULL_TO_CODE:
        return TEAM_FULL_TO_CODE[s]

    # Otherwise assume it's already a code and canonicalize
    return canonical_team_code(s)


def get_team_display_name(code: str) -> str:
    code = canonical_team_code(code)
    full = TEAM_NAME_MAP.get(code, code)
    return f"{full} ({code})" if full != code else code


# -------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_player_data() -> pd.DataFrame:
    """Return the cached player-level data set with normalized team codes."""
    players = pd.read_csv(DATA_DIR / "players.csv")

    if "team" in players.columns:
        players["team"] = players["team"].apply(normalize_team_str)

    return players


@lru_cache(maxsize=1)
def load_game_data() -> pd.DataFrame:
    """Return the cached team game log data set with normalized team codes."""
    games = pd.read_csv(DATA_DIR / "team_games.csv")
    games["date"] = pd.to_datetime(games["date"])

    if "team" in games.columns:
        games["team"] = games["team"].apply(normalize_team_str)
    if "opponent" in games.columns:
        games["opponent"] = games["opponent"].apply(normalize_team_str)

    return games


@lru_cache(maxsize=1)
def load_upcoming_games() -> pd.DataFrame:
    """Return the cached list of upcoming matchups with normalized team codes."""
    upcoming = pd.read_csv(DATA_DIR / "upcoming_games.csv")
    upcoming["date"] = pd.to_datetime(upcoming["date"])

    if "team" in upcoming.columns:
        upcoming["team"] = upcoming["team"].apply(normalize_team_str)
    if "opponent" in upcoming.columns:
        upcoming["opponent"] = upcoming["opponent"].apply(normalize_team_str)

    return upcoming


# -------------------------------------------------------------------
# Team + player level utilities
# -------------------------------------------------------------------

def team_list() -> Iterable[str]:
    """Return sorted list of display names like 'Atlanta Hawks (ATL)'."""
    teams = load_player_data()["team"].dropna().astype(str)

    # Drop multi-team totals from Basketball-Reference
    teams = teams[~teams.isin(["2TM", "3TM", "4TM", "TOT"])]

    codes = sorted(canonical_team_code(t) for t in teams.unique())
    return [get_team_display_name(code) for code in codes]


def compute_team_summary(team: str) -> Dict[str, float]:
    """
    Aggregate a mix of traditional and advanced metrics for a team.

    `team` is the UI label (e.g. 'Atlanta Hawks (ATL)').
    """
    players = load_player_data()
    games = load_game_data()

    team_code = normalize_team_str(team)

    team_players = players[players["team"] == team_code]
    team_games = games[games["team"] == team_code]

    summary = {
        "PPG": team_players["points"].mean(),
        "Usage": team_players["usage_rate"].mean(),
        "Win Shares": team_players["win_shares"].sum(),
        "Avg Minutes": team_players["minutes"].mean(),
        "Off Rating": team_games["offensive_rating"].mean(),
        "Def Rating": team_games["defensive_rating"].mean(),
        "Pace": team_games["pace"].mean(),
        "Rebound %": team_games["rebound_pct"].mean(),
    }

    return {k: round(v, 2) for k, v in summary.items()}


def search_players(query: str) -> pd.DataFrame:
    """Return a filtered player table for the supplied search query."""
    players = load_player_data()
    if not query:
        return players

    q = str(query).strip()

    mask = players["player"].str.contains(q, case=False, na=False)

    # Allow searching by team code or full name
    code_guess = canonical_team_code(q)
    full_matches = [
        name for name in TEAM_NAME_MAP.values()
        if q.lower() in name.lower()
    ]
    codes_from_full = {TEAM_FULL_TO_CODE[name] for name in full_matches}
    team_codes = {code_guess, *codes_from_full}

    if team_codes:
        mask |= players["team"].isin(team_codes)

    return players[mask]


def calculate_true_shooting(fg_pct: float, three_pct: float, ft_pct: float) -> float:
    """Very rough proxy for true shooting percentage."""
    return (fg_pct + three_pct + ft_pct) / 3


def player_projection(player_name: str) -> Dict[str, float]:
    """Estimate per-game production by blending season data and usage."""
    players = load_player_data()
    player_row = players[players["player"] == player_name]
    if player_row.empty:
        raise ValueError(f"Unknown player: {player_name}")
    row = player_row.iloc[0]

    usage_delta = row["usage_rate"] - players["usage_rate"].mean()
    projection_multiplier = 1 + (usage_delta / 100)
    projected_points = row["points"] * projection_multiplier
    projected_rebounds = row["rebounds"] * (row["minutes"] / players["minutes"].mean())
    projected_assists = row["assists"] * projection_multiplier
    true_shooting = calculate_true_shooting(
        row["fg_pct"], row["three_pct"], row["ft_pct"]
    )

    return {
        "Projected Points": round(projected_points, 1),
        "Projected Rebounds": round(projected_rebounds, 1),
        "Projected Assists": round(projected_assists, 1),
        "True Shooting": round(true_shooting, 3),
    }


# -------------------------------------------------------------------
# Simple projection engine for upcoming games (no scikit-learn)
# -------------------------------------------------------------------

def project_upcoming_games() -> pd.DataFrame:
    """
    Return predictions for every entry in the upcoming games file.

    Uses simple aggregated team stats and a logistic transform to
    produce projected points and win probabilities.
    """
    games = load_game_data().copy()
    upcoming = load_upcoming_games().copy()

    if games.empty or upcoming.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "home",
                                     "projected_points", "win_probability"])

    # Mark wins and compute team-level averages
    games["win"] = (games["team_points"] > games["opponent_points"]).astype(int)

    team_stats = (
        games.groupby("team")
        .agg(
            avg_points=("team_points", "mean"),
            avg_opp_points=("opponent_points", "mean"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .set_index("team")
    )
    team_stats["avg_margin"] = (
        team_stats["avg_points"] - team_stats["avg_opp_points"]
    )

    rows = []
    for _, matchup in upcoming.iterrows():
        team_code = normalize_team_str(matchup["team"])
        opp_code = normalize_team_str(matchup["opponent"])

        if team_code not in team_stats.index or opp_code not in team_stats.index:
            # Skip matchups where we don't have stats yet
            continue

        team_profile = team_stats.loc[team_code]
        opp_profile = team_stats.loc[opp_code]

        projected_points = float(team_profile["avg_points"])

        margin_diff = float(
            team_profile["avg_margin"] - opp_profile["avg_margin"]
        )
        # Smooth logistic for win probability
        win_probability = float(1.0 / (1.0 + np.exp(-margin_diff / 5.0)))

        rows.append(
            {
                "date": matchup["date"],
                "team": team_code,
                "opponent": opp_code,
                "home": bool(matchup["home"]),
                "projected_points": round(projected_points, 1),
                "win_probability": round(win_probability, 3),
            }
        )

    return pd.DataFrame(rows)


def team_trend(team: str) -> pd.DataFrame:
    """
    Game-by-game offensive/defensive rating + pace for `team`.

    `team` is whatever label comes from the UI; we normalize it
    to a canonical code and match against the game log.
    """
    games = load_game_data()
    team_code = normalize_team_str(team)
    subset = games[games["team"] == team_code].sort_values("date")
    if subset.empty:
        # Return empty DataFrame with the right columns so Altair doesn't crash
        return pd.DataFrame(columns=["date", "offensive_rating", "defensive_rating", "pace"])
    return subset[["date", "offensive_rating", "defensive_rating", "pace"]]
