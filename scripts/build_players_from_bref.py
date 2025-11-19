# scripts/build_players_from_bref.py
"""
Build data/players.csv with real 2023-24 NBA stats for all teams
using Basketball-Reference per-game + advanced tables.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SEASON = 2024  # 2023-24 season on Basketball-Reference

PER_GAME_URL = f"https://www.basketball-reference.com/leagues/NBA_{SEASON}_per_game.html"
ADVANCED_URL = f"https://www.basketball-reference.com/leagues/NBA_{SEASON}_advanced.html"


def load_per_game() -> pd.DataFrame:
    print("Downloading per-game stats…")
    # Sometimes Basketball-Reference uses multi-level headers; flatten them.
    tables = pd.read_html(PER_GAME_URL, header=0)
    df = tables[0]

    # If we got a MultiIndex for columns, use the last level (actual label names)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Some versions label the team column "Tm", others "Team" – handle both.
    team_col = None
    for candidate in ["Tm", "Team", "team"]:
        if candidate in df.columns:
            team_col = candidate
            break

    if team_col is None:
        raise ValueError(f"Could not find team column in per-game table. Columns: {list(df.columns)}")

    # Drop repeated header rows and league aggregates
    df = df[df["Player"] != "Player"]
    df = df[df[team_col] != "TOT"].copy()

    rename_map = {
        "Player": "player",
        "Pos": "position",
        team_col: "team",
        "G": "games_played",
        "MP": "minutes",
        "PTS": "points",
        "TRB": "rebounds",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "FG%": "fg_pct",
        "3P%": "three_pct",
        "FT%": "ft_pct",
    }
    # Keep only the columns we care about that actually exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}

    df = df.rename(columns=rename_map)

    df["season"] = SEASON

    numeric_cols = [
        "games_played",
        "minutes",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "fg_pct",
        "three_pct",
        "ft_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df.get(numeric_cols, pd.DataFrame()).fillna(0.0)

    return df



def load_advanced() -> pd.DataFrame:
    print("Downloading advanced stats…")
    tables = pd.read_html(ADVANCED_URL, header=0)
    adv = tables[0]

    if isinstance(adv.columns, pd.MultiIndex):
        adv.columns = adv.columns.get_level_values(-1)

    team_col = None
    for candidate in ["Tm", "Team", "team"]:
        if candidate in adv.columns:
            team_col = candidate
            break

    if team_col is None:
        raise ValueError(f"Could not find team column in advanced table. Columns: {list(adv.columns)}")

    adv = adv[adv["Player"] != "Player"]
    adv = adv[adv[team_col] != "TOT"].copy()

    adv = adv.rename(
        columns={
            "Player": "player",
            team_col: "team",
            "USG%": "usage_rate",
            "WS": "win_shares",
        }
    )

    for col in ["usage_rate", "win_shares"]:
        if col in adv.columns:
            adv[col] = pd.to_numeric(adv[col], errors="coerce").fillna(0.0)

    return adv[["player", "team", "usage_rate", "win_shares"]]



def main() -> None:
    per_game = load_per_game()
    advanced = load_advanced()

    merged = per_game.merge(
        advanced,
        on=["player", "team"],
        how="left",
    ).fillna({"usage_rate": 0.0, "win_shares": 0.0})

    cols = [
        "player",
        "team",
        "position",
        "season",
        "games_played",
        "minutes",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "fg_pct",
        "three_pct",
        "ft_pct",
        "usage_rate",
        "win_shares",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "players.csv"
    merged[cols].to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
