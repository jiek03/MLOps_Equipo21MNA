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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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

def colapsar_categorias(series, prop_min=0.05, otra_cat=100):
    # prop_min = 0.05 → mantener solo las categorias con participación ≥5% 
    share = series.value_counts(normalize=True)
    keep = share[share >= prop_min].index
    #print(f"  - Categorías a mantener (≥{prop_min*100:.1f}%): {list(keep)}")
    #print(share)
    return series.where(series.isin(keep), otra_cat)


for col in ordinal_small:
    print(f"Colapsando categorías en {col} ...")
    print("  - Categorías actuales:", df_limpio[col].value_counts().to_dict())
    df_limpio[col] = colapsar_categorias(df_limpio[col], prop_min=0.05, otra_cat=100)
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




# ------------------------
# Notamos que la variable 'PWAPART' contiene un colo valor, por lo cual no aporta información.
# Se decide eliminarla en el siguiente paso de selección de variables.

#df_limpio.drop(columns=['PWAPART'], inplace=True)

# ------------------------