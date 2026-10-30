"""Configurações centrais do projeto: caminhos, colunas, features e hiperparâmetros."""

from pathlib import Path

# --- Caminhos ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Arquivos de entrada (DATASUS e INMET) -----------------------------------
DENGUE_FILE = "dengue.csv"
CLIMATE_FILE = "clima_inmet.csv"
CSV_SEPARATOR = ";"
CSV_ENCODING = "latin1"

# --- Dados de dengue (DATASUS / TabNet) --------------------------------------
DENGUE_WEEK_COLUMN = "Semana epidem. 1.Sintomas(s)"
DENGUE_MISSING_VALUE = "-"  # o TabNet usa "-" para semanas sem casos notificados

# --- Dados climáticos (INMET) ------------------------------------------------
CLIMATE_COLUMNS = {
    "Data Medicao": "data",
    "PRECIPITACAO TOTAL, DIARIO (AUT)(mm)": "precipitacao_mm",
    "TEMPERATURA MAXIMA, DIARIA ": "temp_max_c",
    "TEMPERATURA MEDIA, DIARIA ": "temp_media_c",
    "TEMPERATURA MINIMA, DIARIA": "temp_min_c",
    "UMIDADE RELATIVA DO AR, MEDIA DIARIA (AUT)": "umidade_media_percent",
}
CLIMATE_DATE_FORMAT = "%d/%m/%Y"
CLIMATE_NUMERIC_COLUMNS = [
    "precipitacao_mm",
    "temp_max_c",
    "temp_media_c",
    "temp_min_c",
    "umidade_media_percent",
]
CLIMATE_INTERPOLATED_COLUMNS = [
    "temp_max_c",
    "temp_media_c",
    "temp_min_c",
    "umidade_media_percent",
]
# Chuva é acumulada na semana; temperatura e umidade são médias.
WEEKLY_AGGREGATION = {
    "precipitacao_mm": "sum",
    "temp_max_c": "mean",
    "temp_media_c": "mean",
    "temp_min_c": "mean",
    "umidade_media_percent": "mean",
}

# --- Colunas do dataset semanal ----------------------------------------------
YEAR = "Ano"
WEEK = "Semana"
TARGET = "Casos"
WEEKLY_HISTORY_COLUMNS = [YEAR, WEEK, TARGET, *CLIMATE_NUMERIC_COLUMNS]

# --- Engenharia de features --------------------------------------------------
WEEKS_PER_YEAR = 52
LAGGED_CLIMATE_COLUMNS = ["precipitacao_mm", "temp_media_c", "umidade_media_percent"]
CLIMATE_LAGS = range(1, 9)  # 1 a 8 semanas (~2 meses), ciclo do mosquito + incubação
TARGET_LAGS = [1, 2, 3]  # componente autorregressivo

FEATURES = [
    YEAR,
    "semana_sin",
    "semana_cos",
    "umidade_media_percent_lag_6",
    "umidade_media_percent_lag_7",
    "umidade_media_percent_lag_8",
    "temp_media_c_lag_6",
    "temp_media_c_lag_7",
    "temp_media_c_lag_8",
    "precipitacao_mm_lag_6",
    "precipitacao_mm_lag_7",
    "precipitacao_mm_lag_8",
    "Casos_lag_1",
    "Casos_lag_2",
    "Casos_lag_3",
]

# --- Modelo ------------------------------------------------------------------
TEST_START_YEAR = 2022  # treino: anos < 2022 | teste: anos >= 2022
RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 200, "random_state": RANDOM_STATE, "n_jobs": -1}

# --- Saídas ------------------------------------------------------------------
WEEKLY_HISTORY_FILE = "historico_semanal.csv"
PREPARED_DATA_FILE = "dados_preparados.csv"
MODEL_FILE = "modelo_dengue_rf.joblib"
METRICS_FILE = "metrics.json"
