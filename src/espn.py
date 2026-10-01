"""Raw ESPN data for one season, pulled through sportsdataverse and cached under data/raw/espn."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from src import config  # noqa: F401  (sets SSL_CERT_FILE before sportsdataverse downloads)
import sportsdataverse.mbb as sdv

RAW_ESPN = config.RAW_DIR / "espn"
CORE_API = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"


def _cached(name, season, fetch, refresh=False):
    path = RAW_ESPN / f"{name}_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def team_box(season, refresh=False):
    """One row per team per game: score plus team box score totals."""
    return _cached("team_box", season,
                   lambda s: sdv.load_mbb_team_boxscore(s, return_as_pandas=True), refresh)


def schedule(season, refresh=False):
    """One row per game: venue, neutral site, conference game, status."""
    def fetch(s):
        df = sdv.load_mbb_schedule(s, return_as_pandas=True)
        # stringified nested lists; scores and records are available elsewhere
        return df.drop(columns=["home_linescores", "away_linescores",
                                "home_records", "away_records", "highlights"], errors="ignore")
    return _cached("schedule", season, fetch, refresh)


def game_rosters(season, refresh=False):
    """Every player listed for each team in each game's box score."""
    return _cached("game_rosters", season,
                   lambda s: sdv.load_mbb_game_rosters(s, return_as_pandas=True), refresh)


def season_rosters(season, refresh=False):
    """ESPN season roster with player bios (class, height, position). Snapshot taken after the season."""
    return _cached("season_rosters", season,
                   lambda s: sdv.load_mbb_rosters(s, return_as_pandas=True), refresh)


def team_crosswalk(season, refresh=False):
    """Division I teams with ESPN ids mapped to Barttorvik team names."""
    return _cached("team_crosswalk", season,
                   lambda s: sdv.load_mbb_team_crosswalk(s, return_as_pandas=True), refresh)


def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5,
                                                      status_forcelist=[429, 500, 502, 503, 504])))
    return s


def _get_json(session, url):
    r = session.get(url.replace("http://", "https://", 1), timeout=30)
    r.raise_for_status()
    return r.json()


def _id_from_ref(ref, key):
    # e.g. .../seasons/2026/teams/153?lang=en -> "153" for key="teams"
    return ref.split(f"/{key}/")[1].split("?")[0].split("/")[0]


def _get_json_or_none(session, url):
    try:
        return _get_json(session, url)
    except requests.HTTPError:
        return None


def _team_coach_ids(session, season, team_id):
    j = _get_json_or_none(session, f"{CORE_API}/seasons/{season}/teams/{team_id}/coaches")
    return [int(_id_from_ref(i["$ref"], "coaches")) for i in (j or {}).get("items", [])]


def _record(session, ref):
    stats = {s["name"]: s["value"] for s in _get_json(session, ref).get("stats", [])}
    return stats.get("wins"), stats.get("losses")


def coaches(season, team_ids, refresh=False):
    """Every coach with a regular-season record at one of `team_ids` in `season`.

    ESPN's coach endpoints (the roster `coach` field, the season coach list, and the
    per-team coach list) all report the team's *current* coach for the most recent
    season, so a coach replaced after that season disappears and the new hire shows up
    with a 0-0 record. Candidates are therefore pooled from the season list and the
    per-team lists for this and the previous season, and each candidate's record at the
    team decides who actually coached. Rows with 0 games are kept so they can be audited.
    """
    def fetch(s):
        session = _session()
        with ThreadPoolExecutor(max_workers=16) as pool:
            listed = sdv.espn_mbb_season_coaches(season=s, limit=1000, return_as_pandas=True)["$ref"]
            candidates = {int(_id_from_ref(r, "coaches")) for r in listed}
            for yr in (s, s - 1):
                for ids in pool.map(lambda t: _team_coach_ids(session, yr, t), team_ids):
                    candidates.update(ids)

            candidates = sorted(candidates)
            profiles = pool.map(
                lambda c: _get_json_or_none(session, f"{CORE_API}/seasons/{s}/coaches/{c}"), candidates)
            stints = [
                (c, f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
                 int(_id_from_ref(rec["team"]["$ref"], "teams")), rec["record"]["$ref"])
                for c, p in zip(candidates, profiles) if p
                for rec in p.get("records") or [] if "record" in rec
            ]
            records = list(pool.map(lambda st: _record(session, st[3]), stints))

        return pd.DataFrame(
            [{"season": s, "coach_id": c, "coach_name": name, "team_id": t, "wins": w, "losses": l}
             for (c, name, t, _), (w, l) in zip(stints, records)])
    return _cached("coaches", season, fetch, refresh)
