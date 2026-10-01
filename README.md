# Prediction Under Distribution Shift and Partial Observability: A Case Study in College Basketball

Senior honors thesis in Computer Science at UNC Chapel Hill (COMP 691H in fall, COMP 692H in spring).

- **Student:** Ethan Yountz
- **Advisor:** Tianlong Chen
- **Status:** COMP 691H (fall research semester). 2025–26 data collection is in place, and feature construction is next (see [Next steps](#next-steps)).

---

## Overview

This project asks how well machine learning models work when the data is noisy and the true values are never observed directly. College basketball is a natural test case. When a season begins, nobody knows how strong each team really is. Strength has to be guessed from last season's results, roster changes, and preseason polls. Every game played adds a little more evidence. On top of that, the sport changes from one season to the next: rosters turn over, coaches leave, and scoring trends move. This is a season-level **distribution shift**.

The thesis compares models of increasing complexity on the same prediction task:


| Model family             | Examples                                    | Role                                                                                |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| Dynamic rating baseline  | Elo/Glicko-style ratings, Barttorvik T-Rank | Reference point for "how good is a simple strength estimate"                        |
| Regularized linear       | Ridge (L2), Lasso (L1)                      | Simple and regularized                                                              |
| Gradient-boosted trees   | XGBoost / LightGBM                          | Flexible, tabular                                                                   |
| Recurrent neural network | LSTM/GRU                                    | Stretch goal. Flexible, sequential. May never get enough data for good performance. |


**Prediction target:** the point spread of each game (home score minus away score). This is the only target for now. The main metric is **mean absolute error (MAE)**. Prediction intervals on the spread may be added for the reliability analysis.

**Seasons:** 2014–15 onward (Torvik `year = 2015`), matching the start of the Barttorvik day-by-day ratings archive.

**Language:** Python

The bigger goal is to use sports forecasting as a concrete setting for studying robustness, generalization, and reliability under uncertainty. The lessons should carry over to other fields where predictions have to be made under shift and with incomplete information.

## Research questions

**RQ1. Robustness across seasons.** Under a season-level distribution shift, do simpler or more regularized models hold up better than flexible models?

- Train on one or more past seasons, then evaluate on a different season that the model never saw.
- Compare how prediction error changes across held-out seasons for the simple, regularized, and flexible models.

**RQ2. Within-season stabilization.** As more within-season information comes in, how quickly do predictive performance and reliability settle down?

- Measure each model's error and uncertainty calibration in different phases of the season.
- Possible extension: look at how **roster continuity** affects how fast predictions stabilize. Two candidate measures are:
  - *Returning minutes share:* the percentage of last season's minutes played by players who are back this season.
  - *Coach change:* whether the head coach changed.



## Planned repository layout

```
data/          raw/ and processed/ (gitignored; rebuilt by scripts)
src/           data ingestion, feature construction, models, evaluation
notebooks/     EDA and figures
docs/          experiment spec, feature-availability table, datasheet, reading notes
results/       model comparison tables and figures
```



## Data collection

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.collect --season 2026   # 2025-26; add --refresh to re-download
```

Put `CBD_USER` and `CBD_PW` in `.env` for the T-Rank step; without them it is skipped. cbbdata's ratings archive ends in June 2025 (through the 2024–25 season), so for later seasons the daily ratings are rebuilt from Barttorvik's own daily snapshots (`barttorvik.com/timemachine/team_results/YYYYMMDD_team_results.json.gz`), which match cbbdata's rows exactly on overlapping dates. Raw pulls are cached in `data/raw/`, and the outputs go to `data/processed/`:

| File | Contents |
| ---- | -------- |
| `teams_{season}` | Division I teams (the ones T-Rank rates): ESPN id, Barttorvik name, conference |
| `games_{season}` | One row per completed game: home/away teams, neutral site, scores, `spread`, both teams' box scores, and each team's latest T-Rank snapshot dated before game day |
| `preseason_rosters_{season}` | Each D1 team's roster as listed in its first game, with position, height, and class |
| `coaches_{season}` | Coach(es) with a regular-season record at each D1 team, plus whether the head coach changed in the preceding offseason (from Barttorvik's coaching-moves list) |

### Data structure

`games_{season}` is the central table, with one row per game. Its columns are game context, then `home_*` and `away_*` blocks for team, score, box score, and pre-game T-Rank, plus `spread`. The other tables are keyed by ESPN `team_id` and join onto games through `home_team_id` / `away_team_id`. `teams_{season}` maps ESPN ids to Barttorvik names, so Torvik data can be matched to ESPN games.

The scores and box scores in a game's row are that game's **outcome**. They can never be features for the same game, only inputs to features computed from a team's earlier games.

### Potential features

All of these are known before tip-off. Most would be used as home/away pairs or as home-minus-away differences.

- **Game context:** neutral site, conference game, postseason, date or days into the season, rest days since each team's last game.
- **Pre-game T-Rank (from Torvik):** barthag, adjusted offensive and defensive efficiency, adjusted tempo, wins above bubble, and rank, all from the snapshot dated the day before the game.
- **Previous season:** prior year-end T-Rank ratings, which serve as a preseason prior.
- **Current-season form (derived):** offensive and defensive efficiency (points per 100 possessions), tempo, four factors (eFG%, turnover rate, offensive rebound rate, free-throw rate), scoring margin, and games played. These use only the games completed before the game date, either raw or adjusted for opponent strength.
- **Roster:** roster size, class mix (share of freshmen and seniors), average height, and returning minutes share. Returning minutes share needs last season's player minutes, for example from Torvik's player-season data.
- **Coaching:** offseason coach change, and a coach change during the season.

**Efficiency features not taken from Torvik need to be derived** from the team box scores in `games_{season}`. Possessions are typically estimated as FGA − offensive rebounds + turnovers + 0.475 × FTA. Each derived feature must be computed only from games played before the game being predicted.

## Next steps

**Done:** 2025–26 (`--season 2026`) is collected end to end. That covers games with box scores and spread, pre-game T-Rank for both teams, preseason rosters, and coaches.

1. **Build current-season form features** (not started). Add a feature-building step that:
   - reshapes `games_{season}` into one row per team per game;
   - computes possessions, offensive and defensive efficiency, tempo, four factors, and margin for each game;
   - aggregates each team's games played *before* the game date (season-to-date, possibly with opponent adjustment), and also games played and rest days;
   - joins the results back onto each game as `home_*` / `away_*` features.
2. **Add previous-season and roster features.** These are the prior year-end T-Rank ratings, returning minutes share (last season's player minutes, likely from Torvik player-season data), and roster composition from `preseason_rosters_{season}`.
3. **Backfill earlier seasons (2015–2025).** Run `src.collect` for each season. Three things to resolve:
   - sportsdataverse's ESPN rosters, game rosters, and team crosswalk only go back to 2025, so earlier seasons need another roster source and team-name mapping;
   - re-check `BART_NAME_FIXES` for each season;
   - confirm that the cbbdata archive covers 2015–2025 as expected.
4. **Write the feature-availability table** in `docs/`. For each feature, list its source, when it becomes available, and its leakage risk.
5. **Fit baselines, then the model families**, using the chronological evaluation design below.

**Open data issues (2025–26):** no coach was found for Nicholls. For Cal State Bakersfield, Torvik's coaching-moves list and ESPN's coach records disagree.

## Data sources and APIs



### 1. sportsdataverse-py: game results and schedules

- **Docs:** [https://py.sportsdataverse.org/docs/intro](https://py.sportsdataverse.org/docs/intro)
- **Install:** `pip install sportsdataverse`
- **Language:** Python. Returns polars DataFrames by default. Pass `return_as_pandas=True` to get pandas instead.
- **Relevant module:** `sportsdataverse.mbb` (NCAA men's basketball). It covers ESPN endpoints (scoreboards, schedules, box scores, play-by-play), NCAA-only data (rankings, recruits), and stats.ncaa.org (play-by-play, lineups, stints).
- **Used for:** game outcomes (the spread target), schedules, game location (home, away, or neutral), poll rankings, and box scores for building current-season efficiency features from completed games only.
- **Access:** free, no API key.

```python
from sportsdataverse.mbb import load_mbb_schedule  # exact loader names to be confirmed during the dataset build
```



### 2. cbbdata: historical Barttorvik (T-Rank) ratings

- **Docs:** [https://cbbdata.aweatherman.com/](https://cbbdata.aweatherman.com/) (key endpoint: `[cbd_torvik_ratings_archive](https://cbbdata.aweatherman.com/reference/cbd_torvik_ratings_archive.html)`)
- **Access from Python:** the official client is an R package, but it is a thin wrapper over a REST API that serves Parquet files, so this project calls the API directly:

```python
import io, os, requests, pandas as pd

login = requests.post(
    "https://www.cbbdata.com/api/auth/login",
    json={"username": os.environ["CBD_USER"], "password": os.environ["CBD_PW"]},
)
login.raise_for_status()
key = next(iter(login.json().values()))  # the R client also takes the first field of the response

resp = requests.get(
    "https://www.cbbdata.com/api/torvik/ratings/archive",
    params={"year": 2023, "key": key},
)
ratings = pd.read_parquet(io.BytesIO(resp.content))
```

- **Most important endpoint:** `torvik/ratings/archive` (`cbd_torvik_ratings_archive` in the docs) returns **day-by-day T-Rank ratings and rankings from 2015 to the present**. It can be filtered by `team`, `conf`, `year`, or any other column. Because the ratings are stamped with a date, we can recover exactly what a public rating system believed about each team *on the day of each game*. That is what makes leakage-free features possible, and it doubles as a strong baseline.
- **Other useful endpoints** (API path, with the R docs name in parentheses):
  - `torvik/ratings` (`cbd_torvik_ratings`): year-end ratings, used for previous-season features.
  - `torvik/player/season` (`cbd_torvik_player_season`): player season averages, the likely source for returning minutes share.
  - `torvik/game/stats` and `torvik/schedules` (`cbd_torvik_game_stats`, `cbd_torvik_season_schedule`): per-game box scores and four factors.
  - `rankings/ap` (`cbd_ap_rankings`): AP poll history.
  - `torvik/game/prediction` (`cbd_torvik_game_prediction`): Torvik's own game predictions, a possible external benchmark.
  - `kenpom/ratings/archive` (`cbd_kenpom_ratings_archive`): day-by-day KenPom ratings. **Requires a paid KenPom subscription.**
- **Access:** free account required. Register once with a `POST` to `https://www.cbbdata.com/api/auth/register` (JSON body: `username`, `email`, `password`, `confirm_password`). Set `CBD_USER` and `CBD_PW` as environment variables, for example in a gitignored `.env` file. Never commit these credentials.
- **Coverage limit:** the ratings archive starts with the 2014–15 season, which is why the project is limited to 2015 onward. That caps how many seasons can be used for leave-one-season-out testing.



### Feature availability (to prevent leakage)

Every feature must be computable **using only information available before tip-off**. The feature categories stay the same across season phases. What changes is how much current-season evidence feeds into them.


| When predicting                 | Information available to every model                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Before either team's first game | Previous-season metrics, current roster information, preseason ratings and polls, game location           |
| After a few games               | The same preseason information, plus current-season efficiency metrics computed only from completed games |
| Later in the season             | The same feature categories, with current-season estimates based on more games                            |


A full feature-availability table will live in `docs/` as a deliverable. It will list each feature's source, the timestamp when it becomes available, and its leakage risk.

## Evaluation design

- **Chronological splits only.** No random cross-validation. Training, tuning, and testing follow season order (rolling-origin or leave-one-season-out), and hyperparameters are tuned only on *earlier* seasons.
- **Season phases.** Results are reported separately for early, middle, and late season (exact cutoffs, such as game counts or dates, still to be decided).
- **Metrics.** MAE on the spread is the primary metric. Interval score and coverage are planned for prediction intervals.
- **Shift diagnostics.** Compare seasons by feature distributions, statistical tests, missing data, and a "which season is this from?" classifier.
- **Uncertainty.** Conformal prediction intervals, with documented limits, since seasonal data is not exchangeable.

