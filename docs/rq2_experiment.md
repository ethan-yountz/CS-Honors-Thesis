# RQ2 experiment: within-season stabilization

**Question.** Within a single season, how quickly does each model family's prediction error settle as more of that season's games become available to train on?

## Design

- **Season:** 2025–26 (`--season 2026`).
- **Training data:** only games from the season being evaluated. No earlier seasons are used, on purpose. The small early-season training set is part of what is measured.
- **Walk-forward:** for each game day, every trained model is refit on all D1-vs-D1 games completed before that day and predicts that day's games (`src/rq2.py`, `walk_forward`).
- **Evaluation set:** D1-vs-D1 games (5,752 in 2025–26). Games against non-D1 opponents have no T-Rank snapshot.
- **Target and metric:** spread (home minus away score), MAE.
- **Inputs:** `data/processed/features_{season}.parquet` from `src/features.py`. Pre-game T-Rank adjusted efficiencies and tempo are available for every game. Season-to-date form features (four factors, raw efficiencies, margin, games played, rest) are derived from earlier box scores.

## Models

| Model | Training | Notes |
| ----- | -------- | ----- |
| T-Rank baseline | None | Spread implied by the pre-game adjusted efficiencies and tempo, plus home-court advantage |
| Ridge | Walk-forward | |
| Lasso | Walk-forward | |
| Gradient-boosted trees | Walk-forward | XGBoost (pinned in `requirements.txt`) |
| LSTM/GRU | Walk-forward | Stretch goal |

## Open decisions

- **Phase cutoffs** (`assign_phase`): by date, by teams' games played, or both. Fix them before looking at results.
- **Minimum training games** (`MIN_TRAIN_GAMES`, currently 100). The T-Rank baseline predicts games before the trained models start, so early-phase comparisons should use the games every model predicted.
- **Hyperparameters:** fixed in advance, or tuned at each refit only on games before the prediction day.
- **Missing form features** for a team's first game (no earlier games): impute, fall back to T-Rank only, or add a games-played indicator.
- **Postseason:** report separately (`season_type`) or fold into the late phase. There are 13 games in April.

## Future work

- **Calibration.** Add prediction intervals on the spread (e.g. conformal) and report interval score and coverage by phase, so RQ2 also covers how reliability settles, not just error.
