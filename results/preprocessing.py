# =========================================================
# Preprocesamiento de datos 
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar el CSV limpio.
# - Unir las categorías poco frecuentes
# =========================================================


import os
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import tempfile
#from datetime import datetime


# ------------------------
# 1) Cargar dataset limpio
# ------------------------
SRC_PATH = "data/insurance_clean.csv"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
OUT_DIR_GRAPHS = "graphs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)
os.makedirs(OUT_DIR_GRAPHS, exist_ok=True)

df_limpio = pd.read_csv(SRC_PATH, low_memory=False)
print("✅ Cargado:", SRC_PATH, "| Shape:", df_limpio.shape)


# -----------------------------------------------------------------
# 2) Clasificación de variables (por NOMBRE, no por índice)
# -----------------------------------------------------------------
# - P*: columnas de contribuciones (prefijo 'P'), dominio 0–9 (escala L4).
# - A*: columnas de número de pólizas (prefijo 'A'), dominio 0–12.
# - Binaria real: sólo 'CARAVAN' (si existe).
# - Sociodemográficas: resto (no P*, no A*, no binaria real).
# - Ordinales pequeñas según diccionario: MOSTYPE(1–41), MOSHOOFD(1–10), MGEMLEEF(1–6), MGODRK(0–9), PWAPART(0–9).

cols = list(df_limpio.columns)

# Forzar nombres string si se usó dict_columnas; si no, intentar convertir índices a str para prefijos
columns_str = [str(c) for c in cols]

# Detectar familias por prefijo de nombre
P_cols = [c for c in df_limpio.columns if isinstance(c, str) and c.startswith("P")]
A_cols = [c for c in df_limpio.columns if isinstance(c, str) and c.startswith("A")]

# Ordinales pequeñas del diccionario
ordinal_small = [c for c in ["MOSTYPE", "MOSHOOFD", "MGEMLEEF", "MGODRK", "PWAPART"] if c in df_limpio.columns]

# Binaria real (objetivo)
target_col = "CARAVAN" if "CARAVAN" in df_limpio.columns else None
binary_cols = [target_col] if target_col else []

# Sociodemográficas = resto (excluyendo P*, A*, ordinales y la binaria)
excluir = set(P_cols) | set(A_cols) | set(ordinal_small) | set(binary_cols)
sociodem_cols = [c for c in df_limpio.columns if c not in excluir]


# ------------------------
# 3) reducción de categorías poco frecuentes
# ------------------------

def colapsar_categorias(series, prop_min=0.05, otra_cat=None):
    share = series.value_counts(normalize=True)
    keep = share[share >= prop_min].index
    if otra_cat is None:
        otra_cat = series.mode(dropna=True).iloc[0]
    return series.where(series.isin(keep), otra_cat)



for col in ordinal_small:
    print(f"Colapsando categorías en {col} ...")
    print("  - Categorías actuales:", df_limpio[col].value_counts().to_dict())
    df_limpio[col] = colapsar_categorias(df_limpio[col], prop_min=0.05)
    print("  - Nuevas categorías:", df_limpio[col].value_counts().to_dict())



# ------------------------
# 4) Gráficas - Proporción de variable objetivo
# ------------------------

fig, ax = plt.subplots()
df_limpio["CARAVAN"].value_counts().sort_index().plot(kind='bar', ax=ax, title=col)
#plt.tight_layout()

fig.suptitle("Distribución de la variable objetivo", y=1.05, fontsize=16)
fig.savefig(OUT_DIR_GRAPHS + '/hist_var_obj.png',bbox_inches='tight')


# ------------------------
# 5) Gráficas - frecuencia de las categorías
# ------------------------

fig, axes = plt.subplots(2,3, figsize=(15,8))
axes = axes.flatten()
for i, col in enumerate(ordinal_small):
    df_limpio[col].value_counts().sort_index().plot(kind='bar', ax=axes[i], title=col)
plt.tight_layout()

fig.suptitle("Variables categóricas con categorías colapsadas", y=1.05, fontsize=16)
fig.savefig(OUT_DIR_GRAPHS + '/categorias_colapsadas.png',bbox_inches='tight')



# ------------------------
# 6) Gráficas - P*: columnas de contribuciones (prefijo 'P')
# ------------------------

fig, axes = plt.subplots(7,3, figsize=(15,25))
axes = axes.flatten()
for i, col in enumerate(P_cols):
    df_limpio[col].plot(kind='hist', ax=axes[i], title=col)
plt.tight_layout()

fig.suptitle("Variables de contribuciones (prefijo 'P')", y=1.05, fontsize=16)
fig.savefig(OUT_DIR_GRAPHS + '/hist_contribuciones.png',bbox_inches='tight')


# ------------------------
# 7) Gráficas - A*: columnas de número de pólizas (prefijo 'A')
# ------------------------

fig, axes = plt.subplots(7,3, figsize=(15,25))
axes = axes.flatten()
for i, col in enumerate(A_cols):
    df_limpio[col].plot(kind='hist', ax=axes[i], title=col)
plt.tight_layout()

fig.suptitle("Variables de pólizas (prefijo 'A')", y=1.05, fontsize=16)
fig.savefig(OUT_DIR_GRAPHS + '/hist_polizas.png',bbox_inches='tight')


# ------------------------
# 8) Gráficas - Sociodemográficas
# ------------------------

fig, axes = plt.subplots(8,5, figsize=(20,30))
axes = axes.flatten()
for i, col in enumerate(sociodem_cols):
    df_limpio[col].plot(kind='hist', ax=axes[i], title=col)
plt.tight_layout()

fig.suptitle("Variables Sociodemográficas", y=1.05, fontsize=16)
fig.savefig(OUT_DIR_GRAPHS + '/hist_sociodem.png',bbox_inches='tight')



# -----------------------------------------------------------------
# 7) Clasificación de variables (por NOMBRE, no por índice)
# -----------------------------------------------------------------
# - Categóricas finales: las ordinales pequeñas colapsadas (MOSTYPE, MOSHOOFD, MGEMLEEF, MGODRK, PWAPART).
# - Numéricas: todas las demás columnas que no son categóricas.
# - Target (variable binaria): 'CARAVAN'.
# -----------------------------------------------------------------
TARGET = "CARAVAN" if "CARAVAN" in df_limpio.columns else None
cat_cols = [c for c in ["MOSTYPE", "MOSHOOFD", "MGEMLEEF", "MGODRK", "PWAPART"] if c in df_limpio.columns]
base_cols = [c for c in df_limpio.columns if c != TARGET]
num_cols = [c for c in base_cols if c not in cat_cols]

print(f"[F1] cat_cols={len(cat_cols)} -> {cat_cols}")
print(f"[F1] num_cols={len(num_cols)} (muestra) -> {num_cols[:10]}")

# -----------------------------------------------------------------
# 8) Train/Test split estratificado
# -----------------------------------------------------------------
# - Separamos los datos en 80% entrenamiento y 20% prueba.
# - Estratificamos usando 'CARAVAN' para que las proporciones de 0/1
#   se mantengan iguales en ambos conjuntos (muy importante en datasets desbalanceados).
# -----------------------------------------------------------------

if TARGET is not None:
    X = df_limpio.drop(columns=[TARGET])
    y = df_limpio[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
else:
    X_train, X_test = train_test_split(df_limpio, test_size=0.20, random_state=42)
    y_train = y_test = None

print(f"[F2] X_train={X_train.shape}, X_test={X_test.shape}")
if y_train is not None:
    print("[F2] y_train positivos:", int(y_train.sum()), "| y_test positivos:", int(y_test.sum()))


# -----------------------------------------------------------------
# 9) One-Hot Encoding
# -----------------------------------------------------------------
# - Convertimos las variables categóricas en columnas binarias (0/1).
# - Fit con TRAIN para definir las categorías.
# - En TEST creamos las mismas columnas aunque alguna categoría no aparezca.
# -----------------------------------------------------------------

def one_hot_fit_transform(X_df, cat_cols):
    # Usa solo columnas que realmente existan en X_df
    cols_in = [c for c in cat_cols if c in X_df.columns]
    if not cols_in:
        return pd.DataFrame(index=X_df.index), []
    # prefix=None => usa el nombre de la columna como prefijo automáticamente
    X_cat = pd.get_dummies(X_df[cols_in], prefix=None, drop_first=True, dtype="int8")
    return X_cat, X_cat.columns.tolist()

def one_hot_transform_with_cols(X_df, cat_cols, ohe_cols):
    cols_in = [c for c in cat_cols if c in X_df.columns]
    if not cols_in:
        return pd.DataFrame(index=X_df.index, columns=ohe_cols).fillna(0).astype("int8")
    X_cat = pd.get_dummies(X_df[cols_in], prefix=None, drop_first=True, dtype="int8")
    # Reindex para alinear con columnas vistas en TRAIN
    X_cat = X_cat.reindex(columns=ohe_cols, fill_value=0)
    return X_cat

X_train_cat, ohe_cols = one_hot_fit_transform(X_train, cat_cols)
X_test_cat  = one_hot_transform_with_cols(X_test, cat_cols, ohe_cols)

print(f"[F3] dummies en train: {len(ohe_cols)} columnas")

# -----------------------------------------------------------------
# 10) Escalado de variables numéricas
# -----------------------------------------------------------------
# - Estandarizamos las columnas numéricas (media=0, var=1).
# - Así todas las variables tienen "peso" comparable, evitando que
#   unas dominen solo porque tienen valores más grandes.
# - Fit con TRAIN y luego transform en TEST.
# -----------------------------------------------------------------

scaler = StandardScaler()
X_train_num = X_train[num_cols].copy()
X_test_num  = X_test[num_cols].copy()

if len(num_cols) > 0:
    X_train_num[num_cols] = scaler.fit_transform(X_train_num[num_cols])
    X_test_num[num_cols]  = scaler.transform(X_test_num[num_cols])

print(f"[F4] escaladas: {len(num_cols)} numéricas")

# -----------------------------------------------------------------
# 11) Ensamble final (numéricas + categóricas)
# -----------------------------------------------------------------
# - Pegamos las variables numéricas escaladas con las categóricas OHE.
# - Alineamos columnas de TEST para que coincidan exactamente con TRAIN.
# -----------------------------------------------------------------
X_train_final = pd.concat([X_train_num.reset_index(drop=True),
                           X_train_cat.reset_index(drop=True)], axis=1)
X_test_final  = pd.concat([X_test_num.reset_index(drop=True),
                           X_test_cat.reset_index(drop=True)], axis=1)

X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

print(f"[F5] X_train_final={X_train_final.shape}, X_test_final={X_test_final.shape}")

# -----------------------------------------------------------------
# 12) Guardar artefactos y datasets
# -----------------------------------------------------------------
# - Guardamos todo lo necesario para reproducir el preprocesamiento:
#   * scaler (pkl)
#   * columnas OHE y numéricas (json)
#   * orden final de features (json)
#   * datasets train/test (csv)
# -----------------------------------------------------------------

with open(os.path.join(OUT_DIR_RES, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(OUT_DIR_RES, "ohe_columns.json"), "w", encoding="utf-8") as f:
    json.dump(ohe_cols, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR_RES, "numeric_columns.json"), "w", encoding="utf-8") as f:
    json.dump(num_cols, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR_RES, "feature_order.json"), "w", encoding="utf-8") as f:
    json.dump(X_train_final.columns.tolist(), f, indent=2, ensure_ascii=False)

X_train_final.to_csv(os.path.join(OUT_DIR_DATA, "X_train.csv"), index=False)
X_test_final.to_csv(os.path.join(OUT_DIR_DATA, "X_test.csv"), index=False)
if TARGET is not None:
    y_train.to_csv(os.path.join(OUT_DIR_DATA, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR_DATA, "y_test.csv"), index=False)

print("\n✅ Pipeline de preprocesamiento listo.")


