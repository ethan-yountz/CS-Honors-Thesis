"""Turn raw ESPN pulls into the processed per-season tables."""

import pandas as pd

BOX_STATS = [
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "total_rebounds",
    "assists", "steals", "blocks",
    "turnovers", "team_turnovers", "total_turnovers",
    "fouls", "technical_fouls", "flagrant_fouls",
    "points_in_paint", "fast_break_points", "turnover_points",
    "largest_lead", "lead_changes", "lead_percentage",
]


# ESPN team id -> Barttorvik name for D1 teams the sportsdataverse crosswalk leaves unmatched
# or omits (mostly schools reclassifying from Division II).
BART_NAME_FIXES = {
    2443: "New Orleans",
    2900: "St. Thomas",
    2815: "Lindenwood",
    2511: "Queens",
    2598: "Saint Francis",
    88: "Southern Indiana",
}


def d1_teams(crosswalk, torvik_teams, team_box):
    """Division I teams for the season (those T-Rank rates), keyed by ESPN team id.

    Returns espn_team_id, espn_display_name, bart_team, bart_conf. Raises if any
    T-Rank team cannot be matched to an ESPN id, so new mismatches are caught.
    """
    names = team_box.drop_duplicates("team_id").set_index("team_id")["team_display_name"]
    fixes = pd.DataFrame({"espn_team_id": list(BART_NAME_FIXES), "bart_team": list(BART_NAME_FIXES.values())})
    fixes["espn_display_name"] = fixes["espn_team_id"].map(names)

    teams = crosswalk[["espn_team_id", "espn_display_name", "bart_team"]]
    teams = pd.concat([teams[~teams["espn_team_id"].isin(fixes["espn_team_id"])], fixes])
    teams = teams.merge(torvik_teams.rename(columns={"team": "bart_team", "conf": "bart_conf"}),
                        on="bart_team")

    unmatched = sorted(set(torvik_teams["team"]) - set(teams["bart_team"]))
    if unmatched:
        raise ValueError(f"T-Rank teams with no ESPN id (add them to BART_NAME_FIXES): {unmatched}")
    return teams.sort_values("espn_display_name").reset_index(drop=True)


def games(team_box, schedule, teams):
    """One row per completed game: home/away teams, scores, spread, and both teams' box scores."""
    box = team_box.copy()
    box[BOX_STATS] = box[BOX_STATS].apply(pd.to_numeric, errors="coerce")
    box = box[box["team_score"].notna() & box["opponent_team_score"].notna()]

    sides = pd.crosstab(box["game_id"], box["team_home_away"])
    complete = sides[(sides["home"] == 1) & (sides["away"] == 1)].index
    box = box[box["game_id"].isin(complete)]

    keep = ["team_id", "team_display_name", "team_score", *BOX_STATS]
    def side(name):
        part = box.loc[box["team_home_away"] == name, ["game_id", *keep]]
        part = part.rename(columns={"team_display_name": "team", "team_score": "score"})
        return part.rename(columns=lambda c: c if c == "game_id" else f"{name}_{c}")

    meta = (box[["game_id", "season", "season_type", "game_date", "game_date_time"]]
            .drop_duplicates("game_id"))
    sched = schedule[["game_id", "neutral_site", "conference_competition",
                      "venue_full_name", "notes_headline"]].rename(
        columns={"conference_competition": "conference_game", "venue_full_name": "venue",
                 "notes_headline": "event_note"})

    out = (meta.merge(side("home"), on="game_id")
               .merge(side("away"), on="game_id")
               .merge(sched, on="game_id", how="left"))
    out["spread"] = out["home_score"] - out["away_score"]

    d1 = set(teams["espn_team_id"])
    out["home_d1"] = out["home_team_id"].isin(d1)
    out["away_d1"] = out["away_team_id"].isin(d1)

    lead = ["game_id", "season", "season_type", "game_date", "game_date_time",
            "neutral_site", "conference_game", "venue", "event_note",
            "home_team_id", "home_team", "home_d1", "away_team_id", "away_team", "away_d1",
            "home_score", "away_score", "spread"]
    return out[lead + [c for c in out.columns if c not in lead]] \
        .sort_values(["game_date_time", "game_id"]).reset_index(drop=True)


def preseason_rosters(game_rosters, team_box, season_rosters, teams):
    """Each Division I team's roster as listed for its first game of the season.

    The ESPN season roster endpoint is only snapshotted after the season, so the
    opening-game box score roster is the closest pre-season record of who was on the team.
    """
    gr = game_rosters.copy()
    gr["game_id"] = gr["game_id"].astype("int64")

    first = (team_box.sort_values("game_date_time")
             .drop_duplicates("team_id")[["team_id", "game_id", "game_date"]]
             .rename(columns={"game_id": "first_game_id", "game_date": "first_game_date"}))
    first = first[first["team_id"].isin(teams["espn_team_id"])]

    roster = gr.merge(first, left_on=["team_id", "game_id"], right_on=["team_id", "first_game_id"])

    bio = season_rosters.assign(athlete_id=lambda d: d["athlete_id"].astype("int64"))[
        ["athlete_id", "team_id", "position_abbreviation", "height", "weight",
         "experience_years", "experience_display_value"]]
    roster = roster.merge(bio, on=["athlete_id", "team_id"], how="left")

    cols = {
        "season": "season", "team_id": "team_id", "team_display_name": "team",
        "first_game_id": "first_game_id", "first_game_date": "first_game_date",
        "athlete_id": "athlete_id", "athlete_display_name": "player",
        "athlete_jersey": "jersey", "athlete_position": "position",
        "height": "height", "weight": "weight",
        "experience_years": "class_year", "experience_display_value": "class",
        "starter": "started_opener", "did_not_play": "dnp_opener",
    }
    out = roster[list(cols)].rename(columns=cols)
    out["class_year"] = pd.to_numeric(out["class_year"], errors="coerce").astype("Int64")
    return out.sort_values(["team", "player"]).reset_index(drop=True)


def coaches(espn_coaches, moves, teams):
    """One row per Division I team and coach who coached it during the season.

    Who coached comes from ESPN: only coaches with at least one regular-season game at
    the team are kept, and `is_primary` marks the one with the most games (a team can
    have two after a mid-season firing). Whether the head coach changed over the
    preceding offseason comes from Barttorvik's coaching-moves list, since ESPN's coach
    records for earlier seasons are overwritten by later hires. When ESPN has no coach
    for a team that changed coaches, the new hire from the moves list fills in
    (`coach_source` says which).
    """
    teams = teams[["espn_team_id", "espn_display_name", "bart_team"]].rename(
        columns={"espn_team_id": "team_id", "espn_display_name": "team"})

    df = espn_coaches.merge(teams, on="team_id")
    df["games"] = df["wins"].fillna(0) + df["losses"].fillna(0)
    df = df[df["games"] > 0].sort_values(["team_id", "games"], ascending=[True, False])
    df["is_primary"] = ~df.duplicated("team_id")
    df["n_coaches"] = df.groupby("team_id")["coach_id"].transform("nunique")
    df["coach_source"] = "espn"

    df = teams.merge(df.drop(columns=["team", "bart_team"]), on="team_id", how="left")
    moved = moves.rename(columns={"team": "bart_team", "old_coach": "offseason_old_coach",
                                  "new_coach": "offseason_new_coach"})
    df = df.merge(moved[["bart_team", "offseason_old_coach", "offseason_new_coach"]],
                  on="bart_team", how="left")
    df["coach_change"] = df["offseason_new_coach"].notna()

    fill = df["coach_name"].isna() & df["coach_change"]
    df.loc[fill, "coach_name"] = df.loc[fill, "offseason_new_coach"]
    df.loc[fill, ["is_primary", "n_coaches", "coach_source"]] = [True, 1, "torvik_moves"]
    df = df.astype({"is_primary": "boolean", "coach_id": "Int64", "n_coaches": "Int64",
                    "wins": "Int64", "losses": "Int64", "games": "Int64"})
    df["season"] = espn_coaches["season"].iloc[0]

    cols = ["season", "team_id", "team", "bart_team", "coach_id", "coach_name", "wins", "losses",
            "games", "is_primary", "n_coaches", "coach_source", "coach_change",
            "offseason_old_coach", "offseason_new_coach"]
    return df[cols].sort_values(["team", "games"], ascending=[True, False]).reset_index(drop=True)
