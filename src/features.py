"""Pre-game features for every game in one season, built only from information known before tip-off.

    python -m src.features --season 2026

Writes data/processed/features_{season}.parquet: one row per game with ID_COLS, the
`spread` target, and home_/away_ feature blocks. Every other column is a model input.
"""

import argparse

import pandas as pd

from src import config
from src.build import BOX_STATS

ID_COLS = ["game_id", "season", "season_type", "game_date", "home_team_id", "away_team_id",
           "home_d1", "away_d1", "spread"]

GAME_CONTEXT = ["neutral_site", "conference_game"]

# Opponent-adjusted efficiency and tempo already come with the pre-game T-Rank snapshot.
TRANK = ["barthag", "adj_o", "adj_d", "adj_tempo", "wab", "rank"]

# off_* is the team's own offense; def_* is its opponents' offense against it.
FOUR_FACTORS = [f"{side}_{f}" for side in ("off", "def") for f in ("efg", "tov", "orb", "ftr")]


def team_games(games):
    """Reshape games into one row per team per game: the team's box score as team_*, its opponent's as opp_*."""
    stats = ["score", *BOX_STATS]
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        part = games[["game_id", "game_date", "game_date_time"]].copy()
        part["team_id"] = games[f"{side}_team_id"]
        part["opp_id"] = games[f"{opp}_team_id"]
        part["home_away"] = side
        for s in stats:
            part[f"team_{s}"] = games[f"{side}_{s}"]
            part[f"opp_{s}"] = games[f"{opp}_{s}"]
        sides.append(part)
    return pd.concat(sides, ignore_index=True).sort_values(["game_date_time", "game_id"], ignore_index=True)


def _four_factor_parts(tg):
    """Numerators and denominators of the four factors for each team-game, columns FOUR_FACTORS.

    eFG% = (FGM + 0.5 * 3PM) / FGA, turnover rate = TO / possessions, offensive rebound
    rate = ORB / (ORB + opponent DRB), free-throw rate = FTA / FGA. Possessions are
    FGA - ORB + TO + 0.475 * FTA, matching Barttorvik.
    """
    num, den = {}, {}
    for side, t, o in (("off", "team_", "opp_"), ("def", "opp_", "team_")):
        fga = tg[f"{t}field_goals_attempted"]
        orb = tg[f"{t}offensive_rebounds"]
        to = tg[f"{t}total_turnovers"]
        fta = tg[f"{t}free_throws_attempted"]
        num[f"{side}_efg"] = tg[f"{t}field_goals_made"] + 0.5 * tg[f"{t}three_point_field_goals_made"]
        den[f"{side}_efg"] = fga
        num[f"{side}_tov"], den[f"{side}_tov"] = to, fga - orb + to + 0.475 * fta
        num[f"{side}_orb"], den[f"{side}_orb"] = orb, orb + tg[f"{o}defensive_rebounds"]
        num[f"{side}_ftr"], den[f"{side}_ftr"] = fta, fga
    return pd.DataFrame(num)[FOUR_FACTORS], pd.DataFrame(den)[FOUR_FACTORS]


def four_factors_to_date(tg):
    """Each team's four factors over its games strictly before each game date.

    Rates are season totals (sum of numerators over sum of denominators), not averages of
    per-game rates. They use every completed game, including against non-D1 opponents, and
    are not adjusted for opponent strength. A team's first game day has no earlier games,
    so its values are NaN. Returns team_id, game_date, and FOUR_FACTORS.
    """
    keys = tg[["team_id", "game_date"]]

    def totals_before(part):
        daily = pd.concat([keys, part], axis=1).groupby(["team_id", "game_date"]).sum()
        return daily.groupby(level="team_id").cumsum().groupby(level="team_id").shift()

    num, den = _four_factor_parts(tg)
    return (totals_before(num) / totals_before(den)).reset_index()


def game_stats(tg):
    """Per-game possessions, offensive and defensive efficiency, tempo, and margin."""
    raise NotImplementedError


def season_to_date(tg):
    """Each team's raw efficiencies and margin over its games strictly before the game date, plus games played and rest days."""
    raise NotImplementedError


def build(games):
    """Each game's ID_COLS, game context, pre-game T-Rank, and both teams' season-to-date four factors."""
    trank = [f"{side}_trank_{c}" for side in ("home", "away") for c in TRANK]
    out = games[ID_COLS + GAME_CONTEXT + trank]

    ff = four_factors_to_date(team_games(games))
    for side in ("home", "away"):
        side_ff = ff.rename(columns={"team_id": f"{side}_team_id", **{c: f"{side}_{c}" for c in FOUR_FACTORS}})
        out = out.merge(side_ff, on=[f"{side}_team_id", "game_date"], how="left")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                        help="ending year of the season, e.g. 2026 for 2025-26")
    season = parser.parse_args().season

    features = build(pd.read_parquet(config.PROCESSED_DIR / f"games_{season}.parquet"))
    features.to_parquet(config.PROCESSED_DIR / f"features_{season}.parquet", index=False)
    d1 = features["home_d1"] & features["away_d1"]
    both = features.loc[d1, [f"{s}_off_efg" for s in ("home", "away")]].notna().all(axis=1)
    print(f"  features: {len(features)} games, {features.shape[1] - len(ID_COLS)} feature columns; "
          f"four factors for both teams in {both.sum()} of {d1.sum()} D1 vs D1 games")


if __name__ == "__main__":
    main()
