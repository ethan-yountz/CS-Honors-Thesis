"""RQ2: within-season stabilization. Every model trains only on the season it is evaluated on.

    python -m src.rq2 --season 2026

The season is walked forward one game day at a time: each trained model is refit on all
D1-vs-D1 games completed before that day and predicts that day's games. Early-season
models therefore see very little data; that is part of what RQ2 measures, not something
to fix with earlier seasons. The T-Rank baseline needs no fitting and predicts every game.

Writes to results/rq2/:
    predictions_{season}.parquet   one row per game and model: phase, prediction, actual spread
    mae_by_phase_{season}.csv      MAE and game count per model and phase
"""

import argparse

import pandas as pd

from src import config
from src.features import ID_COLS

OUT_DIR = config.RESULTS_DIR / "rq2"

# Trained models make no prediction until this many games are available to fit on.
MIN_TRAIN_GAMES = 100


def assign_phase(features):
    """Label each game with its season phase (early, middle, late, postseason).

    Cutoffs (game counts or dates) are still to be decided; see docs/rq2_experiment.md.
    """
    raise NotImplementedError


def trank_spread(features):
    """Spread implied by both teams' pre-game T-Rank efficiencies and tempo, plus home-court advantage."""
    raise NotImplementedError


def models():
    """Model name -> factory returning an unfitted regressor with fit/predict.

    Ridge, Lasso, and gradient-boosted trees. LSTM/GRU is a stretch goal.
    """
    raise NotImplementedError


def walk_forward(features, make_model, feature_cols):
    """Refit before each game day on every earlier game, then predict that day's games."""
    preds = []
    for day in sorted(features["game_date"].unique()):
        train = features[features["game_date"] < day]
        if len(train) < MIN_TRAIN_GAMES:
            continue
        test = features[features["game_date"] == day]
        model = make_model().fit(train[feature_cols], train["spread"])
        preds.append(test[["game_id"]].assign(prediction=model.predict(test[feature_cols]),
                                              n_train=len(train)))
    return pd.concat(preds, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                        help="ending year of the season, e.g. 2026 for 2025-26")
    season = parser.parse_args().season

    features = pd.read_parquet(config.PROCESSED_DIR / f"features_{season}.parquet")
    features = features[features["home_d1"] & features["away_d1"]].reset_index(drop=True)
    feature_cols = [c for c in features.columns if c not in ID_COLS]
    features["phase"] = assign_phase(features)

    preds = [features[["game_id"]].assign(model="trank", prediction=trank_spread(features))]
    for name, make_model in models().items():
        preds.append(walk_forward(features, make_model, feature_cols).assign(model=name))
    preds = (pd.concat(preds, ignore_index=True)
             .merge(features[["game_id", "game_date", "phase", "spread"]], on="game_id"))
    preds["abs_error"] = (preds["prediction"] - preds["spread"]).abs()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_DIR / f"predictions_{season}.parquet", index=False)
    mae = preds.groupby(["model", "phase"])["abs_error"].agg(mae="mean", games="size").reset_index()
    mae.to_csv(OUT_DIR / f"mae_by_phase_{season}.csv", index=False)
    print(mae.pivot(index="phase", columns="model", values="mae").round(2))


if __name__ == "__main__":
    main()
