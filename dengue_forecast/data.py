"""Leitura, limpeza e unificação dos dados de dengue (DATASUS) e clima (INMET)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_forecast import config

logger = logging.getLogger(__name__)


def read_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV no formato exportado pelo TabNet/INMET (separador ';', latin1)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Veja a seção 'Dados' do README."
        )
    return pd.read_csv(path, sep=config.CSV_SEPARATOR, encoding=config.CSV_ENCODING)


def load_raw_data(raw_dir: Path = config.DATA_RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos brutos de dengue e clima."""
    dengue = read_csv(raw_dir / config.DENGUE_FILE)
    climate = read_csv(raw_dir / config.CLIMATE_FILE)
    logger.info("Dengue: %d linhas | Clima: %d linhas", len(dengue), len(climate))
    return dengue, climate


def clean_dengue(dengue: pd.DataFrame) -> pd.DataFrame:
    """Converte a tabela do TabNet (semana x ano) para o formato longo Ano/Semana/Casos.

    O TabNet usa "-" para semanas sem casos; esses valores viram 0.
    """
    year_columns = [col for col in dengue.columns if str(col).strip().isdigit()]

    long_df = dengue.melt(
        id_vars=[config.DENGUE_WEEK_COLUMN],
        value_vars=year_columns,
        var_name=config.YEAR,
        value_name=config.TARGET,
    ).rename(columns={config.DENGUE_WEEK_COLUMN: config.WEEK})

    long_df[config.TARGET] = long_df[config.TARGET].replace(config.DENGUE_MISSING_VALUE, 0)
    long_df[config.WEEK] = pd.to_numeric(long_df[config.WEEK], errors="coerce")
    long_df = long_df.dropna(subset=[config.WEEK])  # descarta linhas como "Total"

    long_df[config.YEAR] = long_df[config.YEAR].astype(int)
    long_df[config.WEEK] = long_df[config.WEEK].astype(int)
    long_df[config.TARGET] = pd.to_numeric(long_df[config.TARGET]).astype(int)
    return long_df


def clean_climate(climate: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas, converte tipos e trata valores ausentes dos dados diários do INMET.

    - Corta a série no último dia com medição válida de temperatura média.
    - Precipitação ausente é tratada como 0 (sem chuva registrada).
    - Temperatura e umidade ausentes são interpoladas linearmente.
    """
    climate = climate.rename(columns=config.CLIMATE_COLUMNS)
    climate["data"] = pd.to_datetime(climate["data"], format=config.CLIMATE_DATE_FORMAT)

    for col in config.CLIMATE_NUMERIC_COLUMNS:
        climate[col] = (
            climate[col].replace("null", np.nan).astype(str).str.replace(",", ".")
        )
        climate[col] = pd.to_numeric(climate[col], errors="coerce")

    last_valid_date = climate.loc[climate["temp_media_c"].last_valid_index(), "data"]
    rows_before = len(climate)
    climate = climate[climate["data"] <= last_valid_date].copy()
    logger.info(
        "Clima cortado em %s (%d -> %d linhas)",
        last_valid_date.date(),
        rows_before,
        len(climate),
    )

    climate["precipitacao_mm"] = climate["precipitacao_mm"].fillna(0)
    climate[config.CLIMATE_INTERPOLATED_COLUMNS] = climate[
        config.CLIMATE_INTERPOLATED_COLUMNS
    ].interpolate(method="linear")
    return climate


def aggregate_climate_weekly(climate: pd.DataFrame) -> pd.DataFrame:
    """Agrega os dados diários em semanas ISO (Ano, Semana)."""
    iso = climate["data"].dt.isocalendar()
    climate = climate.assign(
        **{config.YEAR: iso["year"].astype(int), config.WEEK: iso["week"].astype(int)}
    )
    return (
        climate.groupby([config.YEAR, config.WEEK])
        .agg(config.WEEKLY_AGGREGATION)
        .reset_index()
    )


def build_weekly_dataset(dengue: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """Une casos e clima semanais em um único dataset ordenado cronologicamente."""
    dengue_long = clean_dengue(dengue)
    climate_weekly = aggregate_climate_weekly(clean_climate(climate))

    weekly = pd.merge(dengue_long, climate_weekly, on=[config.YEAR, config.WEEK], how="inner")
    weekly = weekly.sort_values([config.YEAR, config.WEEK]).reset_index(drop=True)
    logger.info("Dataset semanal: %d semanas", len(weekly))
    return weekly
