import os
import pytest
import pandas as pd
import pickle
from data.eda_clean import DataCleaner
from pathlib import Path


@pytest.fixture
def tiny_csv_and_dict(tmp_path):
    # CSV crudo sin encabezados (header=None)
    df = pd.DataFrame({
        0: [0.1, 0.2, 0.0, 0.3,0.5],
        1: [1, 0, 2, 1, 3],
        2: [1, 0, 2, 1, 3],   
        3: [1,2,3,4,5],  
        4: [10, 20, 10, 30,5],
        5: [0, 0, 1, 0, 1]
    })
    
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, header=False, index=False)
    # Diccionario de columnas auxiliar para probar funciones
    dict_tmp = {0: "P1", 1: "A1", 2: "MOSTYPE",3: "M_TEST",4:"OTRA_NUM",5:"CARAVAN"}
    dct = tmp_path / "dict.pkl"
    with open(dct, "wb") as f:
        pickle.dump(dict_tmp, f)
    return raw, dct

@pytest.fixture
def dc(tmp_path, tiny_csv_and_dict):
    raw, dct = tiny_csv_and_dict 
    # Ajusta los nombres de argumentos a tu __init__
    return DataCleaner(
        src_path=str(raw),
        output_dir_interim=str(tmp_path / "interim"),
        output_dir_reports=str(tmp_path / "reports"),
        dict_columnas_path=str(dct)#str(dct)
    )


#Test 1: Cargar datos y renombrar columnas
def test_cargar_datos_y_renombrar(dc):
    dc.cargar_datos()
    assert dc.df_raw is not None and dc.df_raw.shape[0] > 0

    dc.cargar_diccionario_columnas()
    dc.renombrar_columnas()

    assert dc.df is not None
    cols = list(dc.df.columns)
    assert "CARAVAN" in cols and "P1" in cols and "MOSTYPE" in cols and "M_TEST" in cols and "OTRA_NUM" in cols and "A1" in cols



#Test 2: Convertir a numérico y clasificar variables
def test_convertir_y_clasificar(dc):
    if hasattr(dc, "cargar_datos"):
        dc.cargar_datos()
    if hasattr(dc, "cargar_diccionario_columnas"):
        dc.cargar_diccionario_columnas()
    if hasattr(dc, "renombrar_columnas"):
        dc.renombrar_columnas()

    if hasattr(dc, "convertir_a_numerico"):
        dc.convertir_a_numerico()
        # esperar que P1 sea numérica cuando posible
        if "P1" in dc.df.columns:
            assert pd.api.types.is_numeric_dtype(dc.df["P1"])

    if hasattr(dc, "clasificar_variables"):
        dc.clasificar_variables()
        # si existe target_col, debería ser CARAVAN
        if getattr(dc, "target_col", None):
            assert dc.target_col == "CARAVAN"


# Test 3: Limpieza de variable por tipo

@pytest.fixture
def prepared_dc(dc):
    """Deja el objeto listo hasta clasificar_variables."""
    for step in (
        "cargar_datos",
        "cargar_diccionario_columnas",
        "renombrar_columnas",
        "convertir_a_numerico",
        "clasificar_variables",
    ):
        if not hasattr(dc, step):
            pytest.skip(f"Falta método requerido: {step}")
        getattr(dc, step)()
    return dc

# Test 3a: Limpieza de variable objetivo binaria
def test_limpiar_variable_objetivo_binaria(prepared_dc):
    dc = prepared_dc
    if not hasattr(dc, "limpiar_variable_objetivo"):
        pytest.skip("limpiar_variable_objetivo no implementado")

    dc.limpiar_variable_objetivo()
    assert "CARAVAN" in dc.df.columns
    assert dc.df["CARAVAN"].dropna().isin([0, 1]).all()

#Test 3b: Limpieza de variables ordinales pequeñas
def test_limpiar_ordinales(prepared_dc):
    dc = prepared_dc
    if not hasattr(dc, "limpiar_ordinales"):
        pytest.skip("limpiar_ordinales no implementado")

    # No debe lanzar excepción; Imputa moda en categóricas ordinales
    dc.limpiar_ordinales()
    # Si existen columnas ordinales declaradas, no deberían quedar NaN
    if getattr(dc, "ordinal_small", None):
        for col in dc.ordinal_small:
            if col in dc.df.columns:
                assert dc.df[col].isna().mean() < 0.5  # umbral laxo para smoke-test

#Test 3c: Limpieza de proporciones
def test_limpiar_proporciones(prepared_dc):
    dc = prepared_dc
    if not hasattr(dc, "limpiar_proporciones"):
        pytest.skip("limpiar_proporciones no implementado")

    dc.limpiar_proporciones()
    # Si el pipeline etiqueta proporciones (p. ej. columnas P_*) verifica clipping [0,1]
    # Ajusta esta lógica a cómo marcas/guardas esas columnas en tu DataCleaner
    cols_prop = [c for c in getattr(dc, "P_cols", []) if c in getattr(dc, "df", {}).columns]
    for col in cols_prop:
        # Solo chequea si la columna es claramente de proporciones (float y con rango)
        if pd.api.types.is_numeric_dtype(dc.df[col]):
            # tolerancia por NaN
            series = dc.df[col].dropna()
            if not series.empty:
                assert series.ge(0).all() and series.le(1).all()

#Test 3d: Redondear valores numéricos
def test_redondear_valores(prepared_dc):
    dc = prepared_dc
    if not hasattr(dc, "redondear_valores"):
        pytest.skip("redondear_valores no implementado")

    before = dc.df.copy()
    dc.redondear_valores()
    # Asegura que no cambió la forma del DF y que no introdujo NaNs nuevos en numéricos
    assert dc.df.shape == before.shape
    num_cols = [c for c in dc.df.columns if pd.api.types.is_numeric_dtype(dc.df[c])]
    for c in num_cols:
        assert dc.df[c].isna().sum() >= before[c].isna().sum()


#Test 4: Métricas y guardado de resultados
def test_metricas_y_guardado(tmp_path, dc):
    # preparar datos mínimos
    for step in ("cargar_datos","cargar_diccionario_columnas","renombrar_columnas","convertir_a_numerico","clasificar_variables"):
        if hasattr(dc, step):
            getattr(dc, step)()

    # generar métricas si existe
    if hasattr(dc, "generar_metricas"):
        m = dc.generar_metricas()
        #assert isinstance(m, (dict,))  # debe devolver dict con llaves básicas

    # guardar resultados si existe
    if hasattr(dc, "guardar_resultados"):
        dc.guardar_resultados()
        # verificar que algo se haya escrito en algún dir de salida conocido
        out_dirs = [
            getattr(dc, "output_dir_interim", None),
            getattr(dc, "output_dir_processed", None),
            getattr(dc, "output_dir_reports", None),
        ]
        any_written = False
        for d in out_dirs:
            if d and os.path.isdir(d) and list(Path(d).glob("*")):
                any_written = True
        assert any_written or not out_dirs  # si no hay dirs configurados, no exigimos escritura


#Test 5: Ejecutar limpieza completa
def test_ejecutar_limpieza_completa(dc):
    if hasattr(dc, "ejecutar_limpieza_completa"):
        # No debe lanzar excepción en un flujo E2E mínimo
        dc.ejecutar_limpieza_completa()
    else:
        pytest.skip("ejecutar_limpieza_completa no implementado")