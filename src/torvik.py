"""Barttorvik T-Rank data (daily ratings, D1 team list, coaching moves), cached under data/raw/torvik."""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src import config

API = "https://www.cbbdata.com/api"
RAW_TORVIK = config.RAW_DIR / "torvik"


class MissingCredentials(RuntimeError):
    pass


def _api_key():
    user, pw = os.environ.get("CBD_USER"), os.environ.get("CBD_PW")
    if not user or not pw:
        raise MissingCredentials("Set CBD_USER and CBD_PW in .env to pull T-Rank ratings from cbbdata.")
    r = requests.post(f"{API}/auth/login", json={"username": user, "password": pw}, timeout=30)
    r.raise_for_status()
    return next(iter(r.json().values()))  # the R client also takes the first field of the response


def _barttorvik_session():
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    # barttorvik serves a "verifying browser" page until this cookie is set
    session.cookies.set("js_verified", "true", domain="barttorvik.com")
    return session


ARCHIVE_COLS = ["rank", "team", "conf", "record", "barthag", "adj_o", "adj_o_rk", "adj_d", "adj_d_rk",
                "adj_tempo", "adj_tempo_rk", "wab", "wab_rk", "date", "year"]

# Positions of the ARCHIVE_COLS fields in barttorvik's time-machine team_results rows,
# matched value-for-value against cbbdata's archive. adj_tempo_rk is not in the file.
TIME_MACHINE_FIELDS = {0: "rank", 1: "team", 2: "conf", 3: "record", 4: "adj_o", 5: "adj_o_rk",
                       6: "adj_d", 7: "adj_d_rk", 8: "barthag", 41: "wab", 42: "wab_rk", 44: "adj_tempo"}


def _cbbdata_archive(year):
    r = requests.get(f"{API}/torvik/ratings/archive", params={"year": year, "key": _api_key()}, timeout=180)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _time_machine_snapshot(session, day):
    r = session.get(f"https://barttorvik.com/timemachine/team_results/{day:%Y%m%d}_team_results.json.gz",
                    timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    rows = pd.DataFrame(r.json())
    df = rows[list(TIME_MACHINE_FIELDS)].rename(columns=TIME_MACHINE_FIELDS)
    df["date"] = day.date()
    return df


def _time_machine_archive(year):
    session = _barttorvik_session()
    days = pd.date_range(f"{year - 1}-10-01", f"{year}-04-30")
    with ThreadPoolExecutor(max_workers=8) as pool:
        snaps = [s for s in pool.map(lambda d: _time_machine_snapshot(session, d), days) if s is not None]
    df = pd.concat(snaps, ignore_index=True)
    df["adj_tempo_rk"] = df.groupby("date")["adj_tempo"].rank(ascending=False, method="min").astype(int)
    df["year"] = year
    return df


def ratings_archive(year, refresh=False):
    """Every team's T-Rank rating on every day of the season `year` (e.g. 2026 = 2025-26).

    Comes from cbbdata when it has the season. cbbdata stopped updating after June 2025,
    so later seasons are rebuilt from barttorvik's daily time-machine snapshots, which
    match cbbdata's rows exactly for the dates checked. `source` records which was used.
    """
    path = RAW_TORVIK / f"ratings_archive_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = _cbbdata_archive(year)
    source = "cbbdata"
    if df.empty:
        df, source = _time_machine_archive(year), "barttorvik_time_machine"
    df = df[ARCHIVE_COLS].assign(source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def season_teams(year, refresh=False):
    """Every Division I team T-Rank rated in season `year`, with its conference."""
    path = RAW_TORVIK / f"teams_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from sportsdataverse.mbb import torvik_ratings
    df = torvik_ratings(year=year, return_as_pandas=True)[["team", "conf"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def coaching_moves(year, refresh=False):
    """Head-coach changes made in the offseason before season `year`, scraped from barttorvik.com."""
    path = RAW_TORVIK / f"coaching_moves_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    r = _barttorvik_session().get("https://barttorvik.com/coaching_moves.php", params={"year": year}, timeout=60)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    df.columns = ["team", "conf", "old_coach", "new_coach"]
    df.insert(0, "year", year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def pregame_ratings(games, archive, teams):
    """Attach each team's most recent T-Rank snapshot dated strictly before the game day.

    Ratings stamped on the game date itself may already include that day's results,
    so they are excluded to avoid leakage. `trank_date` records which snapshot was used.
    """
    archive = archive.copy()
    archive["date"] = pd.to_datetime(archive["date"]).dt.as_unit("us")
    archive["team"] = archive["team"].astype("str")
    id_cols = {"team", "conf", "year", "date", "source"}
    rating_cols = [c for c in archive.columns if c not in id_cols]
    snap = (archive[["team", "date", *rating_cols]]
            .rename(columns={c: f"trank_{c}" for c in rating_cols})
            .assign(trank_date=lambda d: d["date"])
            .sort_values("date"))

    to_bart = teams.set_index("espn_team_id")["bart_team"]
    out = games.copy()
    out["_game_day"] = pd.to_datetime(out["game_date"]).dt.as_unit("us")
    for side in ("home", "away"):
        left = (out[["game_id", "_game_day", f"{side}_team_id"]]
                .assign(team=lambda d: d[f"{side}_team_id"].map(to_bart))
                .sort_values("_game_day"))
        left = left[left["team"].notna()].astype({"team": "str"})
        joined = pd.merge_asof(left, snap, left_on="_game_day", right_on="date",
                               by="team", direction="backward", allow_exact_matches=False)
        joined = joined.drop(columns=["_game_day", f"{side}_team_id", "team", "date"])
        joined = joined.rename(columns={c: f"{side}_{c}" for c in joined.columns if c != "game_id"})
        out = out.merge(joined, on="game_id", how="left")
    return out.drop(columns="_game_day")
