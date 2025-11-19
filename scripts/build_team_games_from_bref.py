"""
Build data/team_games.csv from Basketball-Reference team game logs.

This script:
- Detects all team codes from your players.csv
- Downloads each team's 2024 game log page
- Pulls per-game score + basic "Scoring" stats (ORtg, DRtg, Pace)
- Derives rebound_pct and assist_ratio from box-score columns
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEASON = 2024


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_team_codes() -> List[str]:
    """Read players.csv and return unique team codes (NBA teams only)."""
    players = pd.read_csv(DATA_DIR / "players.csv")
    teams = (
        players["team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # strip multi-team totals if any slipped in
    return [t for t in teams if t not in {"2TM", "3TM", "TOT"}]


def fetch_html(team: str) -> str:
    url = f"https://www.basketball-reference.com/teams/{team}/{SEASON}/gamelog/"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def find_gamelog_table(html: str) -> pd.DataFrame:
    """
    Return the big game-log table as a DataFrame.

    In the current layout (like your ATL_2024_gamelog.html), the correct table:
    - is a MultiIndex table
    - has a top-level column named 'Score'
    """
    tables = pd.read_html(html)
    for df in tables:
        if isinstance(df.columns, pd.MultiIndex):
            top = set(str(l0) for l0 in df.columns.get_level_values(0))
            if any("Score" in t for t in top):
                return df

    raise ValueError("Could not find game-log table with 'Score' columns")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns into single strings like 'Scoring_ORtg', 'Score_Tm', etc.
    """
    out = df.copy()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if c not in ("", "nan") and not pd.isna(c)]
            flat_cols.append("_".join(parts))
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    return out


def extract_team_games(team: str) -> pd.DataFrame:
    """
    Download and parse a single team's game log into the canonical columns:

    date, season, team, opponent, home, team_points, opponent_points,
    pace, offensive_rating, defensive_rating, rebound_pct, assist_ratio
    """
    html = fetch_html(team)
    raw = find_gamelog_table(html)
    df = flatten_columns(raw)

    # Try to locate the key columns generically so this works for all teams
    def col_endswith(suffix: str) -> str:
        matches = [c for c in df.columns if c.endswith(suffix)]
        if not matches:
            raise KeyError(f"Could not find column ending with '{suffix}'")
        return matches[0]

    date_col = col_endswith("_Date")           # e.g. 'Unnamed: 2_level_0_Date'
    opp_col = col_endswith("_Opp")            # e.g. 'Unnamed: 4_level_0_Opp'
    tm_pts_col = "Score_Tm"
    opp_pts_col = "Score_Opp"
    pace_col = "Scoring_Pace"
    ortg_col = "Scoring_ORtg"
    drtg_col = "Scoring_DRtg"

    # Box-score columns for derived stats
    trb_tm_col = "Team_TRB"
    trb_opp_col = "Opponent_TRB"
    ast_col = "Team_AST"
    fga_col = "Team_FGA"
    fta_col = "Team_FTA"
    tov_col = "Team_TOV"

    # Filter out header / total rows: keep rows with real dates
    games = df.copy()
    games = games[games[date_col].notna()].copy()
    games = games[games[date_col] != "Date"].copy()

    # Parse date
    games["date"] = pd.to_datetime(games[date_col])

    # Opponent + home/away flag (Basketball-Reference uses '@' for away)
    opp_raw = games[opp_col].astype(str)
    games["home"] = ~opp_raw.str.startswith("@")
    games["opponent"] = opp_raw.str.replace("@", "", regex=False)

    # Scores
    games["team_points"] = pd.to_numeric(games[tm_pts_col], errors="coerce")
    games["opponent_points"] = pd.to_numeric(games[opp_pts_col], errors="coerce")

    # Advanced team metrics
    games["pace"] = pd.to_numeric(games[pace_col], errors="coerce")
    games["offensive_rating"] = pd.to_numeric(games[ortg_col], errors="coerce")
    games["defensive_rating"] = pd.to_numeric(games[drtg_col], errors="coerce")

    # Rebound % from TRB
    tm_trb = pd.to_numeric(games[trb_tm_col], errors="coerce")
    opp_trb = pd.to_numeric(games[trb_opp_col], errors="coerce")
    games["rebound_pct"] = (tm_trb / (tm_trb + opp_trb)) * 100.0

    # Assist ratio: AST per "possession proxy"
    ast = pd.to_numeric(games[ast_col], errors="coerce")
    fga = pd.to_numeric(games[fga_col], errors="coerce")
    fta = pd.to_numeric(games[fta_col], errors="coerce")
    tov = pd.to_numeric(games[tov_col], errors="coerce")
    poss_proxy = fga + 0.44 * fta + tov
    games["assist_ratio"] = (ast / poss_proxy) * 100.0

    # Add season + team and select final columns
    games["season"] = SEASON
    games["team"] = team

    cols = [
        "date",
        "season",
        "team",
        "opponent",
        "home",
        "team_points",
        "opponent_points",
        "pace",
        "offensive_rating",
        "defensive_rating",
        "rebound_pct",
        "assist_ratio",
    ]
    games = games[cols].sort_values("date").reset_index(drop=True)
    return games


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    teams = get_team_codes()
    print(f"Building team_games for season {SEASON}")
    print(f"Found {len(teams)} team codes: {teams}")

    all_rows: List[pd.DataFrame] = []

    for i, team in enumerate(teams, start=1):
        print(f"[{i}/{len(teams)}] {team}")
        try:
            tg = extract_team_games(team)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! Failed for {team}: {exc!r}")
            continue

        print(f"  -> {len(tg)} games")
        all_rows.append(tg)

    if not all_rows:
        print("No games scraped; aborting.")
        sys.exit(1)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(DATA_DIR / "team_games.csv", index=False)
    print(f"Wrote {len(out)} rows to {DATA_DIR / 'team_games.csv'}")


if __name__ == "__main__":
    main()
