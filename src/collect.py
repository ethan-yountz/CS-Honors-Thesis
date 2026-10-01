"""Collect and build one season of data.

    python -m src.collect --season 2026            # 2025-26 season
    python -m src.collect --season 2026 --refresh  # re-download raw files

Writes to data/processed/:
    teams_{season}.parquet              D1 teams: ESPN id, Barttorvik name, conference
    games_{season}.parquet              one row per game, box scores, spread, pre-game T-Rank
    preseason_rosters_{season}.parquet  each D1 team's opening-game roster
    coaches_{season}.parquet            head coach(es) per D1 team, with offseason coach change
"""

import argparse

from src import build, config, espn, torvik


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                        help="ending year of the season, e.g. 2026 for 2025-26")
    parser.add_argument("--refresh", action="store_true", help="ignore cached raw files")
    args = parser.parse_args()
    season, refresh = args.season, args.refresh
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"Season {season - 1}-{str(season)[-2:]}")
    team_box = espn.team_box(season, refresh)
    teams = build.d1_teams(espn.team_crosswalk(season, refresh), torvik.season_teams(season, refresh), team_box)
    teams.to_parquet(out / f"teams_{season}.parquet", index=False)
    print(f"  teams: {len(teams)} Division I")

    games = build.games(team_box, espn.schedule(season, refresh), teams)
    try:
        archive = torvik.ratings_archive(season, refresh)
        games = torvik.pregame_ratings(games, archive, teams)
        rated = games["home_trank_date"].notna() & games["away_trank_date"].notna()
        print(f"  T-Rank: pre-game ratings for both teams in {rated.sum()} of {len(games)} games")
    except torvik.MissingCredentials as e:
        print(f"  T-Rank: skipped ({e})")
    games.to_parquet(out / f"games_{season}.parquet", index=False)
    d1 = games["home_d1"] & games["away_d1"]
    print(f"  games: {len(games)} completed ({d1.sum()} D1 vs D1), "
          f"{games['game_date'].min()} to {games['game_date'].max()}")

    rosters = build.preseason_rosters(espn.game_rosters(season, refresh), team_box,
                                      espn.season_rosters(season, refresh), teams)
    rosters.to_parquet(out / f"preseason_rosters_{season}.parquet", index=False)
    print(f"  preseason rosters: {len(rosters)} players on {rosters['team_id'].nunique()} teams")

    moves = torvik.coaching_moves(season, refresh)
    coaches = build.coaches(espn.coaches(season, teams["espn_team_id"].tolist(), refresh), moves, teams)
    coaches.to_parquet(out / f"coaches_{season}.parquet", index=False)
    by_team = coaches.drop_duplicates("team_id")
    primary = coaches[coaches["is_primary"] == True]
    print(f"  coaches: {len(primary)} of {len(teams)} teams have a coach "
          f"({(primary['coach_source'] == 'torvik_moves').sum()} filled from Torvik moves), "
          f"{int(by_team['coach_change'].sum())} offseason coach changes, "
          f"{int((primary['n_coaches'] > 1).sum())} teams with more than one coach this season")
    unmatched = sorted(set(moves["team"]) - set(teams["bart_team"]))
    if unmatched:
        print(f"    coaching moves not matched to a D1 team: {unmatched}")


if __name__ == "__main__":
    main()
