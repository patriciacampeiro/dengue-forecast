"""Pipeline completo: dados brutos -> features -> treino -> avaliação -> artefatos.

Uso:
    python -m dengue_forecast.train
    python -m dengue_forecast.train --data-dir data/raw --test-start-year 2022
"""

import argparse
import json
import logging
from pathlib import Path

import joblib

from dengue_forecast import config
from dengue_forecast.data import build_weekly_dataset, load_raw_data
from dengue_forecast.features import add_climate_lags, add_target_lags
from dengue_forecast.model import (
    evaluate,
    feature_importances,
    temporal_train_test_split,
    train_model,
)
from dengue_forecast import plots

logger = logging.getLogger("dengue_forecast")


def run(data_dir: Path, test_start_year: int, make_plots: bool = True) -> dict:
    for directory in (config.DATA_PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Dados
    dengue, climate = load_raw_data(data_dir)
    weekly = build_weekly_dataset(dengue, climate)
    weekly[config.WEEKLY_HISTORY_COLUMNS].to_csv(
        config.DATA_PROCESSED_DIR / config.WEEKLY_HISTORY_FILE, index=False
    )

    # 2. Features
    with_climate_lags = add_climate_lags(weekly)
    dataset = add_target_lags(with_climate_lags)
    dataset.to_csv(config.DATA_PROCESSED_DIR / config.PREPARED_DATA_FILE, index=False)

    # 3. Treino e avaliação
    train, test = temporal_train_test_split(dataset, test_start_year)
    model = train_model(train[config.FEATURES], train[config.TARGET])
    y_pred = model.predict(test[config.FEATURES])

    metrics = {
        "test_start_year": test_start_year,
        "train_weeks": len(train),
        "test_weeks": len(test),
        "random_forest": evaluate(test[config.TARGET], y_pred),
        # Referência: prever que a próxima semana repete a semana anterior.
        "baseline_last_week": evaluate(test[config.TARGET], test["Casos_lag_1"]),
    }
    logger.info(
        "Random Forest -> MAE: %.2f | RMSE: %.2f",
        metrics["random_forest"]["mae"],
        metrics["random_forest"]["rmse"],
    )
    logger.info(
        "Baseline (semana anterior) -> MAE: %.2f | RMSE: %.2f",
        metrics["baseline_last_week"]["mae"],
        metrics["baseline_last_week"]["rmse"],
    )

    # 4. Artefatos
    joblib.dump(model, config.MODELS_DIR / config.MODEL_FILE)
    (config.REPORTS_DIR / config.METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if make_plots:
        importances = feature_importances(model, config.FEATURES)
        plots.plot_yearly_cases(weekly, config.FIGURES_DIR / "casos_por_ano.png")
        plots.plot_correlation_heatmap(with_climate_lags, config.FIGURES_DIR / "correlacao_heatmap.png")
        plots.plot_target_correlation(with_climate_lags, config.FIGURES_DIR / "correlacao_com_casos.png")
        plots.plot_predictions(test, y_pred, config.FIGURES_DIR / "previsao_vs_real.png")
        plots.plot_feature_importances(importances, config.FIGURES_DIR / "importancia_features.png")
        logger.info("Gráficos salvos em %s", config.FIGURES_DIR)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de predição de casos de dengue.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_RAW_DIR,
                        help="Pasta com dengue.csv e clima_inmet.csv")
    parser.add_argument("--test-start-year", type=int, default=config.TEST_START_YEAR,
                        help="Primeiro ano usado como teste")
    parser.add_argument("--no-plots", action="store_true", help="Não gera os gráficos")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    args = parse_args()
    run(args.data_dir, args.test_start_year, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
