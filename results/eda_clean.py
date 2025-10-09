# =========================================================
# LIMPIEZA Y NORMALIZACIÓN DE DATASET
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar el CSV modificado.
# - Tratar CUALQUIER cadena de texto: volverla NaN.
# - Clasificar variables (binarias, sociodemográficas, P*, A*, adicional).
# - Variables binarias: sustituir con moda si hay valores != 0 y 1.
# - Variables no binarias: sustituir vacíos con mediana.
# - Detectar y corregir outliers con IQR.
# - Eliminar duplicados.
# - Guardar dataset limpio y métricas básicas para el reporte.
# =========================================================

import os
import json
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
import pickle

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

# ------------------------
# 2) Carga del diccionario auxiliar
# ------------------------

with open( OUT_DIR_RES + "/nombres_columnas.pkl" , "rb") as f:
    dict_columnas = pickle.load(f)


print("✅ Cargado:", OUT_DIR_RES + "/nombres_columnas.pkl")


# ------------------------
# 3) Renombramos las columnas para facilitar el manejo de los datos
# ------------------------
df = df_raw.copy()
df = df.iloc[1:].reset_index(drop=True)
df.rename(columns=dict_columnas ,inplace=True)
df.columns = [str(c).strip().upper() for c in df.columns]

#print(df.head())

# -----------------------------------------------------
# 4) Función para convertir string a número
# -----------------------------------------------------
def to_number_or_nan(x):
    # Si ya es numérico, lo devuelvo tal cual
    if isinstance(x, (int, float, np.number)):
        return x
    
    # Si es NaN de pandas/numpy, mantenerlo como NaN
    if pd.isna(x):
        return np.nan
    
    # Si es string, intento convertir a número; si no se puede → NaN
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"", "na", "n/a", "none", "null", "nan", "?", "-", "--"}:
            return np.nan
        return pd.to_numeric(s, errors="coerce")
    
    # Cualquier otro tipo se convierte en NaN
    return np.nan

# -----------------------------------------------------
# 5) Calcular moda de forma segura
# -----------------------------------------------------
def safe_mode(series):
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

# -----------------------------------------------------
# 6) Winsorización por IQR
# -----------------------------------------------------
def winsorize_iqr(series, iqr_k=1.5, hard_clip=None, zero_threshold=0.6, min_nonzero=20):
    s = pd.to_numeric(series, errors="coerce")

    # 1) Serie vacía → nada que hacer
    if s.dropna().empty:
        return s, 0

    # 2) Base para IQR (maneja columnas con gran cantidad de ceros)
    s_notnull = s.dropna()
    share_zero = (s_notnull == 0).mean() if len(s_notnull) else 0.0
    s_nz = s_notnull[s_notnull != 0]
    base = s_nz if (share_zero >= zero_threshold and len(s_nz) >= min_nonzero) else s_notnull

    # Si base vacía, sólo aplicar hard_clip si existe
    if base.empty:
        s_w = s.copy()
        if hard_clip is not None:
            lo, hi = hard_clip
            s_w = s_w.clip(lower=lo, upper=hi)
        return s_w, 0

    # 3) Cuartiles e IQR
    q1, q3 = base.quantile(0.25), base.quantile(0.75)
    iqr = q3 - q1

    # 4) IQR no válido → sólo aplicar hard_clip si existe
    if not np.isfinite(iqr) or iqr <= 0:
        s_w = s.copy()
        if hard_clip is not None:
            lo, hi = hard_clip
            s_w = s_w.clip(lower=lo, upper=hi)
        n_recortes = int((~s_w.eq(s)).sum())
        return s_w, n_recortes

    # 5) Winsorizar por IQR
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    s_w = s.clip(lower=lo, upper=hi)

    # 6) hard_clip duro opcional
    if hard_clip is not None:
        hlo, hhi = hard_clip
        s_w = s_w.clip(lower=hlo, upper=hhi)

    n_recortes = int((~s_w.eq(s)).sum())
    return s_w, n_recortes

# -----------------------------------------------------
# 7) Guardar CSV de forma segura con reintentos
# -----------------------------------------------------
def safe_to_csv(df, path, retries=5, wait_sec=1.0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=os.path.dirname(path) or ".")
    os.close(fd)
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

# -----------------------------------------------------------------
# 8) Aplicar conversión a numérico a TODAS las columnas
# -----------------------------------------------------------------
nulls_before = int(df.isna().sum().sum())

df = df.apply(lambda col: col.map(to_number_or_nan))
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

nulls_after_conversion = int(df.isna().sum().sum())
print(f" Valores nulos antes: {nulls_before} | después: {nulls_after_conversion}")

# -----------------------------------------------------------------
# 9) Clasificación de variables (por NOMBRE, no por índice)
# -----------------------------------------------------------------
# - P*: columnas de contribuciones (prefijo 'P'), dominio 0–9 (escala L4).
# - A*: columnas de número de pólizas (prefijo 'A'), dominio 0–12.
# - Binaria real: sólo 'CARAVAN' (si existe).
# - Sociodemográficas: resto (no P*, no A*, no binaria real).
# - Ordinales pequeñas según diccionario: MOSTYPE(1–41), MOSHOOFD(1–10), MGEMLEEF(1–6), MGODRK(0–9), PWAPART(0–9).

cols = list(df.columns)

# Forzar nombres string si se usó dict_columnas; si no, intentar convertir índices a str para prefijos
columns_str = [str(c) for c in cols]

# Detectar familias por prefijo de nombre
P_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("P")]
A_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("A")]

# Ordinales pequeñas del diccionario
ordinal_small = [c for c in ["MOSTYPE", "MOSHOOFD", "MGEMLEEF", "MGODRK", "PWAPART"] if c in df.columns]

# Binaria real (objetivo)
target_col = "CARAVAN" if "CARAVAN" in df.columns else None
binary_cols = [target_col] if target_col else []

# Sociodemográficas = resto (excluyendo P*, A*, ordinales y la binaria)
ordinal_domains = {
    "MOSTYPE":  (1, 41),
    "MOSHOOFD": (1, 10),
    "MGEMLEEF": (1, 6),
    "MGODRK":   (0, 9),
    "PWAPART":  (0, 9)
}
excluir = set(P_cols) | set(A_cols) | set(ordinal_domains.keys()) | ({target_col} if target_col else set())
sociodem_cols = [c for c in df.columns if c not in excluir]

print(f"CARAVAN presente: {bool(target_col)} | P*: {len(P_cols)} | A*: {len(A_cols)} | "
      f"Ordinales: {len(ordinal_small)} | Sociodemográficas: {len(sociodem_cols)}")

# -----------------------------------------------------------------
# 10) CARAVAN (binaria real): imputar moda {0,1}
# -----------------------------------------------------------------
if target_col:
    moda = safe_mode(df[target_col].dropna().clip(0, 1))  # por seguridad al rango binario
    moda = int(moda) if pd.notna(moda) else 0
    df[target_col] = df[target_col].fillna(moda).clip(0, 1).round().astype("int8")

# -----------------------------------------------------------------
# 11) Sociodemográficas: imputar MEDIANA + winsorizar IQR (sin hard clip)
# -----------------------------------------------------------------
total_outliers_sociodem = 0
for c in sociodem_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median(skipna=True))
    df[c], rec = winsorize_iqr(df[c], iqr_k=1.5, hard_clip=None)
    total_outliers_sociodem += rec

# -----------------------------------------------------------------
# 12) P* (0–9): mediana + winsorizar(IQR) + clip duro [0,9] + entero
# -----------------------------------------------------------------
total_outliers_P = 0
for c in P_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median(skipna=True))
    df[c], rec = winsorize_iqr(df[c], iqr_k=1.5, hard_clip=(0, 9))
    total_outliers_P += rec
    df[c] = df[c].round().clip(0, 9).astype("int8")

# -----------------------------------------------------------------
# 13) A* (0–12): mediana + winsorizar(IQR) + clip duro [0,12] + entero
# -----------------------------------------------------------------
total_outliers_A = 0
for c in A_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median(skipna=True))
    df[c], rec = winsorize_iqr(df[c], iqr_k=1.5, hard_clip=(0, 12))
    total_outliers_A += rec
    df[c] = df[c].round().clip(0, 12).astype("int8")

# -----------------------------------------------------------------
# 14) Ordinales pequeñas: mediana + clip a dominio + entero
# -----------------------------------------------------------------

total_outliers_ordinal = 0
for c in ordinal_small:
    lo, hi = ordinal_domains[c]
    if df[c].isna().any():
        modo = safe_mode(df[c])
        df[c] = df[c].fillna(modo)
    before = df[c].copy()
    df[c] = df[c].clip(lo, hi)  # sin winsorize IQR
    rec = int((~df[c].eq(before)).sum())
    total_outliers_ordinal += rec
    df[c] = df[c].round().astype("int8")

# -----------------------------------------------------------------
# 15) Resto de M* (tasas/proporciones): mediana + winsorizar(IQR) + clip lógico
# -----------------------------------------------------------------
# Detectamos M* que no sean ordinales ni CARAVAN
M_rest = [c for c in df.columns
          if isinstance(c, str) and c.startswith("M")
          and c not in ordinal_small and c != target_col]

total_outliers_Mrest = 0
for c in M_rest:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median(skipna=True))
    # Elegir hard clip lógico: si el rango real ≤9, usar [0,9], si no asumir porcentaje [0,100]
    vmax = df[c].max(skipna=True)
    hard = (0, 9) if (pd.notna(vmax) and vmax <= 9) else (0, 100)
    df[c], rec = winsorize_iqr(df[c], iqr_k=1.5, hard_clip=hard)
    total_outliers_Mrest += rec
    # Mantener como float (proporción)
    df[c] = df[c].astype("float32")
# Floor o ceiling
df = df.apply(
    lambda col: np.where(
        np.issubdtype(col.dtype, np.number),
        np.where(col - np.floor(col) < 0.5, np.floor(col), np.ceil(col)),
        col
    )
) 
# -----------------------------------------------------------------
# 16) Métricas de limpieza
# -----------------------------------------------------------------
total_outliers_removed = (
    total_outliers_sociodem
    + total_outliers_P
    + total_outliers_A
    + total_outliers_ordinal
    + total_outliers_Mrest
)

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
        "outliers_corregidos": int(total_outliers_removed),
    },
    "familias": {
        "P_cols": len(P_cols),
        "A_cols": len(A_cols),
        "ordinales": len(ordinal_small),
        "sociodemograficas": len(sociodem_cols),
        "M_rest": len(M_rest),
        "tiene_caravan": bool(target_col)
    }
}

print("📊 Métricas de limpieza:", metrics["cambios_realizados"])

# -----------------------------------------------------------------
# 17) Guardar resultados
# -----------------------------------------------------------------
OUT_DATA_PATH = os.path.join(OUT_DIR_DATA, "insurance_clean.csv")
OUT_METRICS_PATH = os.path.join(OUT_DIR_RES, "eda_metrics.json")

safe_to_csv(df, OUT_DATA_PATH)
with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"✅ Guardado dataset limpio: {OUT_DATA_PATH}")
print(f"✅ Guardado métricas EDA:   {OUT_METRICS_PATH}")