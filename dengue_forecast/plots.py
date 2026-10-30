"""Gráficos da análise exploratória e da avaliação do modelo.

Cada função salva a figura em disco e a fecha, para rodar também sem interface gráfica.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dengue_forecast import config


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_yearly_cases(weekly: pd.DataFrame, path: Path) -> Path:
    """Total de casos de dengue por ano."""
    totals = weekly.groupby(config.YEAR)[config.TARGET].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(totals.index.astype(str), totals.values)
    ax.set_title("Casos de dengue por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Total de casos")
    return _save(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    """Mapa de calor da correlação de Pearson entre todas as variáveis."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlação entre dengue e variáveis climáticas")
    return _save(fig, path)


def plot_target_correlation(df: pd.DataFrame, path: Path) -> Path:
    """Correlação de cada variável com o número de casos, em ordem decrescente."""
    corr = df.corr()[config.TARGET].drop(config.TARGET).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette="viridis", legend=False, ax=ax)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Correlação das variáveis com o número de casos de dengue")
    ax.set_xlabel("Coeficiente de correlação (Pearson)")
    return _save(fig, path)


def plot_predictions(test: pd.DataFrame, y_pred: np.ndarray, path: Path) -> Path:
    """Casos reais vs previstos no período de teste."""
    labels = (
        test[config.YEAR].astype(str) + "-S" + test[config.WEEK].astype(str).str.zfill(2)
    ).to_list()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(labels, test[config.TARGET].to_numpy(), label="Casos reais", color="red", marker="o")
    ax.plot(labels, y_pred, label="Previsão (clima + autorregressão)", color="blue", linestyle="--")
    ax.set_xticks(labels[::10])
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Casos de dengue: reais vs previstos")
    ax.set_xlabel("Ano e semana epidemiológica")
    ax.set_ylabel("Número de casos")
    ax.legend()
    ax.grid(True)
    return _save(fig, path)


def plot_feature_importances(importances: pd.Series, path: Path) -> Path:
    """Importância das features no Random Forest."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue", ax=ax)
    ax.set_title("Importância das features (Random Forest)")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    return _save(fig, path)
