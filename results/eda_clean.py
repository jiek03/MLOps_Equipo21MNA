# =========================================================
# LIMPIEZA Y NORMALIZACIÓN DE DATASET
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo:
# - Cargar el CSV modificado.
# - Tratar CUALQUIER cadena de texto: volverla NaN.
# - Clasificar variables (binarias, sociodemográficas, P*, A*, adicional).
# - Variables binarias: sustituir con moda si hay valores != 0 y 1.
# - Variables no binarias: sustituir vacíos con mediana.
# - Detectar y corregir outliers con IQR.
# - Eliminar duplicados.
# - Guardar dataset limpio y métricas para el reporte.
# =========================================================

import os
import json
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime

# ------------------------
# 1) Cargar dataset origen
# ------------------------
SRC_PATH = "dataset/insurance_company_modified.csv"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)

df_raw = pd.read_csv(SRC_PATH, header=None, low_memory=False)
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
        if s in {"", "na", "n/a", "none", "null", "nan", "?", "-", "--"}:
            return np.nan
        # Intento convertir a float
        valor_numerico = pd.to_numeric(s, errors="coerce")
        return valor_numerico
    
    # Cualquier otro tipo se convierte en NaN
    return np.nan

# -----------------------------------------------------
# 3) Calcular moda de forma segura
# -----------------------------------------------------
def safe_mode(series):
    # Calcular moda excluyendo NaN
    moda = series.mode(dropna=True)
    # Si hay moda, devolver el primer valor
    if not moda.empty:
        return moda.iloc[0]
    # Si no hay moda, devolver NaN
    return np.nan

# -----------------------------------------------------
# 4) Detectar y remover outliers usando IQR
# -----------------------------------------------------
def remove_outliers_iqr(series):
    # Convertir a numérico
    s = pd.to_numeric(series, errors="coerce")
    # Si está vacía, devolver la serie original
    if s.dropna().empty:
        return s, 0
    
    # Calcular Q1, Q3 e IQR
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir límites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers_mask = (s < limite_inferior) | (s > limite_superior)
    cantidad_outliers = int(outliers_mask.sum())
    
    # Reemplazar outliers con la mediana
    if cantidad_outliers > 0:
        mediana = s.median()
        s_limpia = s.copy()
        s_limpia[outliers_mask] = mediana
        return s_limpia, cantidad_outliers
    
    return s, 0

# -----------------------------------------------------
# 5) Guardar CSV de forma segura con reintentos
# -----------------------------------------------------
def safe_to_csv(df, path, retries=5, wait_sec=1.0):
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Crear archivo temporal
    fd, tmp = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=os.path.dirname(path) or ".")
    os.close(fd)
    # Guardar en archivo temporal
    df.to_csv(tmp, index=False, encoding="utf-8")
    # Intentar mover el archivo temporal al destino
    os.replace(tmp, path)

# -----------------------------------------------------------------
# 6) Aplicar conversión a numérico
# -----------------------------------------------------------------
df = df_raw.copy()
nulls_before = int(df.isna().sum().sum())

# Limpieza: CUALQUIER cadena de texto se convierte en NaN
df = df.apply(lambda col: col.map(to_number_or_nan))

# Fuerzo que TODAS las columnas queden numéricas
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

nulls_after_conversion = int(df.isna().sum().sum())
print(f" Valores nulos antes: {nulls_before} | después: {nulls_after_conversion}")

# -----------------------------------------------------------------
# 7) Clasificación de variables
# -----------------------------------------------------------------
extra_idx = 86  # Columna adicional continua

# Detectar columnas binarias automáticamente
# Criterio: más del 90% de los valores no nulos son 0 o 1
binary_cols = []
for c in df.columns:
    if c == extra_idx:
        continue
    valores_no_nulos = df[c].dropna()
    if len(valores_no_nulos) == 0:
        continue
    count_0_o_1 = ((valores_no_nulos == 0) | (valores_no_nulos == 1) | (valores_no_nulos == 0.0) | (valores_no_nulos == 1.0)).sum()
    proporcion_binaria = count_0_o_1 / len(valores_no_nulos)
    if proporcion_binaria > 0.90:
        binary_cols.append(c)

# Identificar columnas P* (contribuciones: dominio 0-9)
P_cols = [c for c in range(43, 64) if c not in binary_cols and c != extra_idx]

# Identificar columnas A* (número de seguros: dominio 0-12)
A_cols = [c for c in range(64, 86) if c not in binary_cols and c != extra_idx]

# Variables sociodemográficas (columnas 0-42)
sociodem_cols = [c for c in range(0, 43) if c not in binary_cols and c != extra_idx]

print(f"Variables binarias: {len(binary_cols)} | Sociodemográficas: {len(sociodem_cols)} | P*: {len(P_cols)} | A*: {len(A_cols)}")

# -----------------------------------------------------------------
# 8) Limpieza de variables binarias
# -----------------------------------------------------------------
# Solo deben contener valores 0 y 1
# Si hay valores que NO son 0 o 1, reemplazarlos con la moda
# Si hay valores vacíos (NaN), reemplazarlos con la moda

binary_cleaning_stats = {}

for c in binary_cols:
    stats = {
        "valores_nulos": int(df[c].isna().sum()),
        "valores_invalidos": 0,
        "moda_usada": None
    }
    
    # Contar valores que no son 0, 1 o NaN
    valores_unicos = df[c].dropna().unique()
    valores_invalidos = [v for v in valores_unicos if v not in [0, 0.0, 1, 1.0]]
    stats["valores_invalidos"] = len(valores_invalidos)
    
    # Si hay valores inválidos o nulos, imputar con la moda
    if stats["valores_nulos"] > 0 or stats["valores_invalidos"] > 0:
        # Calcular moda solo de valores válidos (0 y 1)
        valores_validos = df[c][(df[c] == 0) | (df[c] == 1) | (df[c] == 0.0) | (df[c] == 1.0)]
        moda = safe_mode(valores_validos)
        # Si no hay moda válida, usar 0 por defecto
        if pd.isna(moda):
            moda = 0
        stats["moda_usada"] = float(moda)
        # Crear máscara para identificar valores a reemplazar
        mascara_reemplazar = df[c].isna() | ~df[c].isin([0, 1, 0.0, 1.0])
        # Reemplazar SOLO donde la máscara es True
        df.loc[mascara_reemplazar, c] = moda
    
    binary_cleaning_stats[int(c)] = stats

print(f"✅ Variables binarias procesadas: {len(binary_cols)}")

# -----------------------------------------------------------------
# 9) Limpieza de variables sociodemográficas
# -----------------------------------------------------------------
# Reemplazar valores vacíos (NaN) con la MEDIANA
# Detectar y eliminar outliers usando IQR

sociodem_cleaning_stats = {}
total_outliers_sociodem = 0

for c in sociodem_cols:
    stats = {
        "valores_nulos": int(df[c].isna().sum()),
        "mediana_usada": None,
        "outliers_removidos": 0
    }
    
    # Calcular mediana antes de imputar
    mediana = df[c].median(skipna=True)
    stats["mediana_usada"] = float(mediana) if not pd.isna(mediana) else None
    
    # Imputar valores nulos con la mediana
    if stats["valores_nulos"] > 0:
        df[c] = df[c].fillna(mediana)
    
    # Detectar y remover outliers usando IQR
    df[c], outliers_count = remove_outliers_iqr(df[c])
    stats["outliers_removidos"] = outliers_count
    total_outliers_sociodem += outliers_count
    
    sociodem_cleaning_stats[int(c)] = stats

print(f"✅ Variables sociodemográficas procesadas: {len(sociodem_cols)} | Outliers: {total_outliers_sociodem}")

# -----------------------------------------------------------------
# 10) Limpieza de variables P* (contribuciones)
# -----------------------------------------------------------------
# Dominio: 0-9 según escala L4
# Reemplazar valores vacíos con la MEDIANA
# Detectar y eliminar outliers usando IQR

P_cleaning_stats = {}
total_outliers_P = 0

for c in P_cols:
    stats = {
        "valores_nulos": int(df[c].isna().sum()),
        "mediana_usada": None,
        "outliers_removidos": 0
    }
    
    mediana = df[c].median(skipna=True)
    stats["mediana_usada"] = float(mediana) if not pd.isna(mediana) else None
    
    if stats["valores_nulos"] > 0:
        df[c] = df[c].fillna(mediana)
    
    df[c], outliers_count = remove_outliers_iqr(df[c])
    stats["outliers_removidos"] = outliers_count
    total_outliers_P += outliers_count
    
    P_cleaning_stats[int(c)] = stats

print(f"✅ Variables P* procesadas: {len(P_cols)} | Outliers: {total_outliers_P}")

# -----------------------------------------------------------------
# 11) Limpieza de variables A* (número de seguros)
# -----------------------------------------------------------------
# Dominio: 0-12 (número de pólizas)
# Reemplazar valores vacíos con la MEDIANA
# Detectar y eliminar outliers usando IQR

A_cleaning_stats = {}
total_outliers_A = 0

for c in A_cols:
    stats = {
        "valores_nulos": int(df[c].isna().sum()),
        "mediana_usada": None,
        "outliers_removidos": 0
    }
    
    mediana = df[c].median(skipna=True)
    stats["mediana_usada"] = float(mediana) if not pd.isna(mediana) else None
    
    if stats["valores_nulos"] > 0:
        df[c] = df[c].fillna(mediana)
    
    df[c], outliers_count = remove_outliers_iqr(df[c])
    stats["outliers_removidos"] = outliers_count
    total_outliers_A += outliers_count
    
    A_cleaning_stats[int(c)] = stats

print(f"✅ Variables A* procesadas: {len(A_cols)} | Outliers: {total_outliers_A}")

# -----------------------------------------------------------------
# 12) Limpieza de columna adicional (continua)
# -----------------------------------------------------------------
extra_cleaning_stats = {}

if extra_idx in df.columns:
    stats = {
        "valores_nulos": int(df[extra_idx].isna().sum()),
        "mediana_usada": None,
        "outliers_removidos": 0
    }
    
    mediana = df[extra_idx].median(skipna=True)
    stats["mediana_usada"] = float(mediana) if not pd.isna(mediana) else None
    
    if stats["valores_nulos"] > 0:
        df[extra_idx] = df[extra_idx].fillna(mediana)
    
    df[extra_idx], outliers_count = remove_outliers_iqr(df[extra_idx])
    stats["outliers_removidos"] = outliers_count
    
    extra_cleaning_stats[int(extra_idx)] = stats
    print(f"✅ Columna adicional procesada: {extra_idx} | Outliers: {stats['outliers_removidos']}")

# -----------------------------------------------------------------
# 13) Eliminar duplicados completos
# -----------------------------------------------------------------
duplicados_antes = int(df.duplicated().sum())
df.drop_duplicates(inplace=True)
duplicados_despues = int(df.duplicated().sum())

print(f"🧹 Filas duplicadas eliminadas: {duplicados_antes - duplicados_despues}")

# -----------------------------------------------------------------
# 14) Métricas de limpieza
# -----------------------------------------------------------------
caravan_idx = 85
tiene_caravan = caravan_idx in binary_cols

# Calcular totales
total_outliers_removed = total_outliers_sociodem + total_outliers_P + total_outliers_A
if extra_idx in extra_cleaning_stats:
    total_outliers_removed += extra_cleaning_stats[extra_idx]["outliers_removidos"]

total_nulls_imputed = (
    sum(s["valores_nulos"] for s in binary_cleaning_stats.values()) +
    sum(s["valores_nulos"] for s in sociodem_cleaning_stats.values()) +
    sum(s["valores_nulos"] for s in P_cleaning_stats.values()) +
    sum(s["valores_nulos"] for s in A_cleaning_stats.values())
)
if extra_idx in extra_cleaning_stats:
    total_nulls_imputed += extra_cleaning_stats[extra_idx]["valores_nulos"]

# Compilar métricas
eda_stats = {
    "resumen_general": {
        "filas_totales": int(df.shape[0]),
        "columnas_totales": int(df.shape[1]),
        "valores_nulos_finales": int(df.isna().sum().sum()),
        "porcentaje_completitud": round((1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    },
    "clasificacion_variables": {
        "binarias": len(binary_cols),
        "sociodemograficas": len(sociodem_cols),
        "contribuciones_P": len(P_cols),
        "numero_seguros_A": len(A_cols),
        "adicional_continua": 1 if extra_idx in df.columns else 0
    }
}

if tiene_caravan:
    eda_stats["distribucion_variable_objetivo"] = {
        "CARAVAN_0": int((df[caravan_idx] == 0).sum()),
        "CARAVAN_1": int((df[caravan_idx] == 1).sum()),
        "proporcion_positivos": round((df[caravan_idx] == 1).sum() / len(df) * 100, 2)
    }

metrics = {
    "dataset_original": {
        "filas": int(df_raw.shape[0]),
        "columnas": int(df_raw.shape[1]),
        "valores_nulos": nulls_before
    },
    "dataset_limpio": {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "valores_nulos": int(df.isna().sum().sum())
    },
    "cambios_realizados": {
        "filas_eliminadas": int(df_raw.shape[0] - df.shape[0]),
        "duplicados_eliminados": duplicados_antes - duplicados_despues,
        "valores_nulos_imputados": total_nulls_imputed,
        "outliers_corregidos": total_outliers_removed
    },
    "variables_binarias": binary_cleaning_stats,
    "variables_sociodemograficas": sociodem_cleaning_stats,
    "variables_contribuciones_P": P_cleaning_stats,
    "variables_numero_seguros_A": A_cleaning_stats,
    "variable_adicional": extra_cleaning_stats,
    "estadisticas_eda": eda_stats
}

print(f"📊 Métricas de limpieza:", metrics["cambios_realizados"])

# -----------------------------------------------------------------
# 15) Guardar resultados
# -----------------------------------------------------------------
OUT_DATA_PATH = os.path.join(OUT_DIR_DATA, "insurance_clean.csv")
OUT_METRICS_PATH = os.path.join(OUT_DIR_RES, "eda_cleaning_metrics.json")

safe_to_csv(df, OUT_DATA_PATH)
with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"✅ Guardado dataset limpio: {OUT_DATA_PATH}")
print(f"✅ Guardado métricas EDA:   {OUT_METRICS_PATH}")