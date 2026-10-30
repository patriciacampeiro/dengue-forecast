"""Engenharia de features: sazonalidade cíclica e defasagens (lags) temporais."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dengue_forecast import config


def add_seasonal_features(df: pd.DataFrame, period: int = config.WEEKS_PER_YEAR) -> pd.DataFrame:
    """Codifica a semana do ano com seno e cosseno.

    Assim o modelo entende que a semana 52 é vizinha da semana 1.
    """
    df = df.copy()
    angle = 2 * np.pi * df[config.WEEK] / period
    df["semana_sin"] = np.sin(angle)
    df["semana_cos"] = np.cos(angle)
    return df


def add_lag_features(df: pd.DataFrame, columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """Cria colunas `<coluna>_lag_<n>` com o valor de n semanas atrás.

    O DataFrame deve estar ordenado cronologicamente.
    """
    df = df.copy()
    lags = list(lags)
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_climate_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona sazonalidade e lags climáticos de 1 a 8 semanas, removendo linhas incompletas."""
    df = add_seasonal_features(df)
    df = add_lag_features(df, config.LAGGED_CLIMATE_COLUMNS, config.CLIMATE_LAGS)
    return df.dropna()


def add_target_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona lags dos próprios casos (componente autorregressivo)."""
    df = add_lag_features(df, [config.TARGET], config.TARGET_LAGS)
    return df.dropna()
