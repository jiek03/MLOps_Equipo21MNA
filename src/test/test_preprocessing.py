import sys,os
import pandas as pd 
import numpy as np
import pytest
from features.preprocessing import DataPreprocessor


# Creamos un fixture de pytest para un DataFrame pequeño para pruebas
@pytest.fixture
def tiny_df():
    return pd.DataFrame({
        "P1": [0.1, 0.2, 0.0, 0.3,0.5],
        "A1": [1, 0, 2, 1, 3],
        "MOSTYPE": ["A","A","B","C", "B"],
        "MOSHOOFD": ["X","X","X","Y","Y"],
        "OTRA_NUM": [10, 20, 10, 30,5],
        "CARAVAN": [0, 0, 1, 0, 1]
    })

# creamos un fixture de pytest para un DataPreprocessor con rutas temporales
@pytest.fixture
def preproc(tmp_path):
    return DataPreprocessor(
        src_path=str(tmp_path / "data.csv"),
        output_dir_processed=str(tmp_path / "processed"),
        output_dir_reports=str(tmp_path / "reports"),
        output_dir_figures=str(tmp_path / "reports" / "figures"),
        output_dir_models=str(tmp_path / "models"),
        output_dir_references=str(tmp_path / "references"),
    )

# Test 1: Carga y clasificación de variables
def test_carga_y_clasificacion(tmp_path, preproc, tiny_df):
    tiny_df.to_csv(preproc.src_path, index=False)

    preproc.cargar_datos_limpios()
    # Verificamos que los datos tienen la forma correcta
    assert preproc.df.shape == (5, 6)

    #Verificamos la clasificación de variables
    preproc.clasificar_variables()
    assert preproc.target_col == "CARAVAN"
    assert preproc.P_cols == ["P1"]
    assert preproc.A_cols == ["A1"]
    assert set(preproc.ordinal_small) >= {"MOSTYPE","MOSHOOFD"}
    assert set(preproc.sociodem_cols) <= {"OTRA_NUM"}


# Test 2: Colapsar categorías raras de variables ordinales pequeñas 
def test_colapsar_categorias_raras(preproc, tiny_df):
    preproc.df = tiny_df.copy()
    preproc.ordinal_small = ["MOSTYPE","MOSHOOFD"]

    #Verificamos que colapsa correctamente
    preproc.colapsar_categorias_raras(prop_min=0.34)
    assert set(preproc.df["MOSTYPE"].unique()) <= {"A","B"}


# Test 3: Preparar variables para modelado y dividir en train/test
def test_preparar_variables_y_split(preproc, tiny_df):
    preproc.df = tiny_df.copy()
    preproc.target_col = "CARAVAN"
    preproc.ordinal_small = ["MOSTYPE","MOSHOOFD"]

    preproc.preparar_variables_modelado()
    assert set(preproc.cat_cols) == {"MOSTYPE","MOSHOOFD"}
    assert all(col in preproc.num_cols for col in ["P1","A1","OTRA_NUM"])

    #Verificamos división train/test sea correcta
    preproc.dividir_train_test(test_size=0.25, random_state=0)
    assert preproc.X_train.shape[0] in (2,3)
    assert preproc.X_test.shape[0] in (1,2)


# Test 4: Aplicar One-Hot Encoding y escalado, y ensamblar datasets finales
def test_ohe_scaler_ensamble(preproc, tiny_df):
    preproc.df = tiny_df.copy()
    preproc.target_col = "CARAVAN"
    preproc.ordinal_small = ["MOSTYPE","MOSHOOFD"]
    preproc.preparar_variables_modelado()
    preproc.dividir_train_test(test_size=0.5, random_state=0)

    preproc.inicializar_transformer()
    preproc.entrenar_transformer()
    preproc.transformar_variables()

    # Verificamos que las columnas transformadas coinciden en train y test, en numero de columnas y nombres.
    assert set(preproc.X_train_transformed_df.columns) == set(preproc.X_test_transformed_df.columns)
    assert preproc.X_train_transformed_df.shape[1] == preproc.X_test_transformed_df.shape[1]
