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
#import tempfile
#from datetime import datetime


# ------------------------
# 1) Cargar dataset limpio
# ------------------------
SRC_PATH = "data/insurance_clean.csv"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)

df_limpio = pd.read_csv(SRC_PATH, header=None, low_memory=False)
print("✅ Cargado:", SRC_PATH, "| Shape:", df_limpio.shape)


# ------------------------
# 2) Carga del diccionario auxiliar
# ------------------------

with open( OUT_DIR_RES + "/nombres_columnas.pkl" , "rb") as f:
    dict_columnas = pickle.load(f)


print("✅ Cargado:", OUT_DIR_RES + "/nombres_columnas.pkl")


# ------------------------
# 3) Renombramos las columnas para facilitar el manejo de los datos
# ------------------------

df_limpio = df_limpio.iloc[1:].reset_index(drop=True)
df_limpio.rename(columns=dict_columnas ,inplace=True)

#print(df_limpio.head())

# ------------------------
# 4) Reducimos las categorías poco frecuentes
# ------------------------

def colapsar_categorias(series, prop_min=0.05, otra_cat=None):
    # prop_min = 0.05 → mantener solo las categorias con participación ≥5% 
    share = series.value_counts(normalize=True)
    keep = share[share >= prop_min].index
    if otra_cat is None:
        otra_cat = series.mode(dropna=True).iloc[0]
    #print(f"  - Categorías a mantener (≥{prop_min*100:.1f}%): {list(keep)}")
    #print(share)
    return series.where(series.isin(keep), otra_cat)


var_categoricas = ['MGEMLEEF','MOSHOOFD','MGODRK','PWAPART'] # Variables con catalogo 

for col in var_categoricas:
    print(f"Colapsando categorías en {col} ...")
    print("  - Categorías actuales:", df_limpio[col].value_counts().to_dict())
    df_limpio[col] = colapsar_categorias(df_limpio[col], prop_min=0.05, otra_cat=None)
    print("  - Nuevas categorías:", df_limpio[col].value_counts().to_dict())



# -----------------------------------------------------------------
# 5) Clasificación de variables
# -----------------------------------------------------------------

variables_binarias_idx = [19, 40, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 
                          67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85] # Estos indices se obtubieron de la limpieza.

variables_binarias = [dict_columnas[i] for i in variables_binarias_idx]

# Identificar columnas P* (contribuciones: dominio 0-9)
P_cols = df_limpio.filter(regex=r'^P').columns.tolist()
P_cols = [c for c in P_cols if c not in variables_binarias]

# Identificar columnas A* (número de seguros: dominio 0-12)
A_cols = df_limpio.filter(regex=r'^A').columns.tolist()
A_cols = [c for c in A_cols if c not in variables_binarias]

# Variables sociodemográficas
sociodem_cols = [c for c in df_limpio.columns if c not in variables_binarias and c not in P_cols and c not in A_cols and c not in var_categoricas]

print(f"Variables binarias: {len(variables_binarias)} | Sociodemográficas: {len(sociodem_cols)} | categoricas:{len(var_categoricas)} | P*: {len(P_cols)} | A*: {len(A_cols)}")

# -----------------------------------------------------------------
# 5) descripción de variables
# -----------------------------------------------------------------

print("\nDescripción de variables binarias:")
print(df_limpio[variables_binarias].describe().T)

print("\nDescripción de variables sociodemográficas:")
print(df_limpio[sociodem_cols].describe().T)

print("\nDescripción de variables categoricas:")
print(df_limpio[var_categoricas].describe().T)

print("\nDescripción de variables P*:")
print(df_limpio[P_cols].describe().T)

# -----------------------------------------------------------------
# 5) Correccion de variables binarias 
# -----------------------------------------------------------------

# Al revisar los resumenes estadísticos, se observa que algunas variables binarias tienen valores distintos a 0 y 1.
# Se procede a corregir estas variables, asignando 1 a cualquier valor distinto de 0.

for col in variables_binarias:
    df_limpio[col] = df_limpio[col].apply(lambda x: 1 if x != 0 else 0)


