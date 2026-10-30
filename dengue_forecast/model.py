"""Divisão temporal, treino e avaliação do modelo de predição."""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from dengue_forecast import config

logger = logging.getLogger(__name__)


def temporal_train_test_split(
    df: pd.DataFrame, test_start_year: int = config.TEST_START_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste por ano, sem embaralhar, para evitar vazamento do futuro."""
    train = df[df[config.YEAR] < test_start_year]
    test = df[df[config.YEAR] >= test_start_year]
    if train.empty or test.empty:
        raise ValueError(f"Ano de corte {test_start_year} deixa treino ou teste vazio.")
    logger.info("Treino: %d semanas | Teste: %d semanas", len(train), len(test))
    return train, test


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Treina um Random Forest com clima defasado e componente autorregressivo."""
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X, y)
    return model


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE e RMSE (em número de casos)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def feature_importances(model: RandomForestRegressor, features: list[str]) -> pd.Series:
    """Importância de cada feature segundo o modelo, em ordem decrescente."""
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
