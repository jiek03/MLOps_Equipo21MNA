# =========================================================
# LIMPIEZA Y NORMALIZACIÓN DE DATASET
# =========================================================
# Objetivo del script:
# - Cargar el CSV modificado.
# - Tratar CUALQUIER cadena de texto: volverla NaN.
# - Rellenar NaN con mediana por columna.
# - Eliminar duplicados.
# - Guardar dataset limpio y métricas básicas para el reporte.
# =========================================================

import os
import json
import numpy as np
import pandas as pd

# ------------------------
# 1) Cargar dataset origen
# ------------------------
SRC_PATH = "dataset/insurance_company_modified.csv"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)

df_raw = pd.read_csv(SRC_PATH, low_memory=False)
print("✅ Cargado:", SRC_PATH, "| Shape:", df_raw.shape)

# -----------------------------------------------------
# 2) Función para convertir string a número
# -----------------------------------------------------
def to_number_or_nan(x):
    # Si ya es numérico, lo devuelvo tal cual
    if isinstance(x, (int, float, np.number)):
        return x

    # Si es NaN de pandas/numpy, mantenerlo como NaN
    if pd.isna(x):
        return np.nan

    # Si es string, hago conversión a NaN
    if isinstance(x, str):
        s = x.strip().lower()

        # Casos que equivalen a missing
        if s in {"", "na", "n/a", "none", "null", "nan", "?", "-"}:
            return np.nan

        # Intento convertir a float
        try:
            val = float(s)
            return val
        except ValueError:
            # No es convertible -> NaN 
            return np.nan
    return np.nan

# -----------------------------------------------------------------
# 3) Aplicar conversión
# -----------------------------------------------------------------
df = df_raw.copy()

# Conteo previo de nulos y duplicados
nulls_before = int(df.isna().sum().sum())
dups_before = int(df.duplicated().sum())

# Limpieza
df = df.applymap(to_number_or_nan)

# Fuerzo que TODAS las columnas queden numéricas
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------------------------
# 4) Rellenar NaN con mediana por columna
# -------------------------------------
cols_all_nan = []
for c in df.columns:
    if df[c].isna().all():
        cols_all_nan.append(c)
    else:
        med = df[c].median()
        df[c].fillna(med, inplace=True)

# --------------------------------
# 5) Eliminar duplicados completos
# --------------------------------
df.drop_duplicates(inplace=True)

# --------------------------------
# 6) Métricas de limpieza (básicas)
# --------------------------------
metrics = {
    "filas_y_columnas_original": list(df_raw.shape),
    "filas_y_columnas_limpio": list(df.shape),
    "valores_nulos_antes_de_limpieza": nulls_before,
    "valores_nulos_despues_de_limpieza": int(df.isna().sum().sum()),
    "filas_duplicadas_eliminadas": int(dups_before - df.duplicated().sum()),
    "columnas_completamente_vacias": cols_all_nan,
}

print("📊 Métricas de limpieza:", metrics)

# --------------------------------
# 7) Guardar resultados
# --------------------------------
OUT_DATA_PATH = os.path.join(OUT_DIR_DATA, "insurance_clean.csv")
OUT_METRICS_PATH = os.path.join(OUT_DIR_RES, "eda_metrics.json")

df.to_csv(OUT_DATA_PATH, index=False)
with open(OUT_METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Guardado dataset limpio: {OUT_DATA_PATH}")
print(f"✅ Guardado métricas EDA:   {OUT_METRICS_PATH}")
