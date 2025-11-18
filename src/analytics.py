"""Utility functions for working with the sample NBA analytics data."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# -------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_player_data() -> pd.DataFrame:
    """Return the cached player level data set."""
    return pd.read_csv(DATA_DIR / "players.csv")


@lru_cache(maxsize=1)
def load_game_data() -> pd.DataFrame:
    """Return the cached team game log data set."""
    games = pd.read_csv(DATA_DIR / "team_games.csv")
    games["date"] = pd.to_datetime(games["date"])
    return games


@lru_cache(maxsize=1)
def load_upcoming_games() -> pd.DataFrame:
    """Return the cached list of upcoming matchups."""
    upcoming = pd.read_csv(DATA_DIR / "upcoming_games.csv")
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    return upcoming


# -------------------------------------------------------------------
# Team + player level utilities
# -------------------------------------------------------------------
def team_list() -> Iterable[str]:
    return sorted(load_player_data()["team"].unique())


def compute_team_summary(team: str) -> Dict[str, float]:
    """Aggregate a mix of traditional and advanced metrics for a team."""
    players = load_player_data()
    team_players = players[players["team"] == team]
    summary = {
        "PPG": team_players["points"].mean(),
        "Usage": team_players["usage_rate"].mean(),
        "Win Shares": team_players["win_shares"].sum(),
        "Avg Minutes": team_players["minutes"].mean(),
    }
    games = load_game_data()
    team_games = games[games["team"] == team]
    summary.update(
        {
            "Off Rating": team_games["offensive_rating"].mean(),
            "Def Rating": team_games["defensive_rating"].mean(),
            "Pace": team_games["pace"].mean(),
            "Rebound %": team_games["rebound_pct"].mean(),
        }
    )
    return {k: round(v, 2) for k, v in summary.items()}


def search_players(query: str) -> pd.DataFrame:
    """Return a filtered player table for the supplied search query."""
    players = load_player_data()
    if not query:
        return players
    mask = players["player"].str.contains(query, case=False, na=False)
    mask |= players["team"].str.contains(query, case=False, na=False)
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

    Instead of training scikit-learn models, this uses simple
    aggregated team stats and a logistic transform to produce
    projected points and win probabilities.
    """
    games = load_game_data().copy()
    upcoming = load_upcoming_games().copy()

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
        team = matchup["team"]
        opp = matchup["opponent"]

        team_profile = team_stats.loc[team]
        opp_profile = team_stats.loc[opp]

        projected_points = float(team_profile["avg_points"])

        margin_diff = float(
            team_profile["avg_margin"] - opp_profile["avg_margin"]
        )
        win_probability = float(1.0 / (1.0 + np.exp(-margin_diff / 5.0)))

        rows.append(
            {
                "date": matchup["date"],
                "team": team,
                "opponent": opp,
                "home": bool(matchup["home"]),
                "projected_points": round(projected_points, 1),
                "win_probability": round(win_probability, 3),
            }
        )

    return pd.DataFrame(rows)


def team_trend(team: str) -> pd.DataFrame:
    games = load_game_data()
    subset = games[games["team"] == team].sort_values("date")
    subset = subset[["date", "offensive_rating", "defensive_rating", "pace"]]
    return subset

