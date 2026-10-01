# Prediction Under Distribution Shift and Partial Observability: A Case Study in College Basketball

Senior honors thesis in Computer Science at UNC Chapel Hill (COMP 691H in fall, COMP 692H in spring).

- **Student:** Ethan Yountz
- **Advisor:** Tianlong Chen
- **Status:** COMP 691H (fall research semester). The data pipeline is under construction.

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

