import pandas as pd
import pytest
from pathlib import Path
import os
import pickle
from unittest.mock import patch
from data.eda_clean import DataCleaner
from features.preprocessing import DataPreprocessor
from models.train_model import ModelTrainer
import models.train_model as tm



#Fixtures para pruebas de integración
@pytest.fixture
def dirs(tmp_path):
    base = tmp_path
    d = {
        "raw": base / "data" / "raw",
        "interim": base / "data" / "interim",
        "processed": base / "data" / "processed",
        "reports": base / "reports",
        "figures": base / "reports" / "figures",
        "models": base / "models",
        "references": base / "references",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

# fixture que crea un CSV crudo y un diccionario de columnas
@pytest.fixture
def raw_csv_and_dict(dirs):
    """
    Crea un CSV crudo SIN encabezados y un diccionario de columnas
    compatible con DataCleaner. Mantén las llaves como strings (0,1,2,...)
    si tu DataCleaner lo espera así.
    """
    # dataset simple, con target, numéricas y categórica/ordinal
    df = pd.DataFrame({
        0: [0.1, 0.2, 0.0, 0.3,0.5],
        1: [1, 0, 2, 1, 3],
        2: [1, 0, 2, 1, 3],   
        3: [1,2,3,4,5],  
        4: [10, 20, 10, 30,5],
        5: [0, 0, 1, 0, 1]
    })
    raw = dirs["raw"] / "raw.csv"
    df.to_csv(raw, header=False, index=False)

    # diccionario -> mapea índices a nombres
    col_dict =  {0: "P1", 1: "A1", 2: "MOSTYPE",3: "M_TEST",4:"OTRA_NUM",5:"CARAVAN"}
    dct = dirs["raw"] / "dict.json"
    with open(dct, "wb") as f:
        pickle.dump(col_dict, f)
        #dct.write_text(json.dumps(col_dict), encoding="utf-8") # cambiar a dict
    return raw, dct


#Test 1 : Prueba de  cleaner integración completa
def test_cleaner_end_to_end(dirs, raw_csv_and_dict):
    raw, dct = raw_csv_and_dict

    cleaner = DataCleaner(
        src_path=str(raw),
        dict_columnas_path =str(dct),
        output_dir_interim=str(dirs["interim"]),
        output_dir_reports=str(dirs["reports"])
    )

    # Ejecuta el flujo de limpieza completo (si existe),
    # de lo contrario invoca pasos esenciales.
    if hasattr(cleaner, "ejecutar_limpieza_completa"):
        cleaner.ejecutar_limpieza_completa()
    else:
        cleaner.cargar_datos()
        cleaner.cargar_diccionario_columnas()
        cleaner.renombrar_columnas()
        cleaner.convertir_a_numerico()
        cleaner.clasificar_variables()
        if hasattr(cleaner, "limpiar_variable_objetivo"): cleaner.limpiar_variable_objetivo()
        if hasattr(cleaner, "limpiar_sociodemograficas"): cleaner.limpiar_sociodemograficas()
        if hasattr(cleaner, "limpiar_ordinales"): cleaner.limpiar_ordinales()
        if hasattr(cleaner, "limpiar_proporciones"): cleaner.limpiar_proporciones()
        if hasattr(cleaner, "redondear_valores"): cleaner.redondear_valores()
        if hasattr(cleaner, "guardar_resultados"): cleaner.guardar_resultados()

    # Debe existir un CSV/artefacto en interim o processed y un json de métricas en reports
    wrote_any_csv = any(Path(dirs["interim"]).glob("*.csv")) or any(Path(dirs["processed"]).glob("*.csv"))
    wrote_any_json = any(Path(dirs["reports"]).glob("*.json"))
    assert wrote_any_csv, "Esperaba al menos un CSV limpio en interim/processed"
    assert wrote_any_json, "Esperaba al menos un JSON de métricas en reports"

    # Si dejó un DataFrame en memoria, valida esquema
    df = getattr(cleaner, "df", None)
    if df is not None:
        assert {"CARAVAN", "P1", "MOSTYPE", "A1"} <= set(df.columns)
        # target binario limpio
        if "CARAVAN" in df:
            assert df["CARAVAN"].dropna().isin([0,1]).all()


# Test 2: Prueba de preprocesador integración completa
def test_preprocessor_from_clean_to_features(dirs, raw_csv_and_dict):
    # Prepara un "clean.csv" mínimo como si viniera del cleaner.
    clean = pd.DataFrame({
        "CARAVAN": [0,1,0,1,0,0,1],
        "P1": [10,20,15,30,40,25,35],
        "A1": [5,6,7,8,30,9,10],            # incluye valor atípico previo limpiado
        "MOSTYPE": ["A","A","B","C","A","B","C"],
    })
    clean_path = (dirs["interim"] / "clean.csv")
    clean.to_csv(clean_path, index=False)

    pre = DataPreprocessor(
        src_path=str(clean_path),
        output_dir_processed=str(dirs["processed"]),
        output_dir_reports=str(dirs["reports"]),
        output_dir_figures=str(dirs["figures"]),
        output_dir_models=str(dirs["models"]),
        output_dir_references=str(dirs["references"]),
    )

    # Clasificación de variables (ajústalo a cómo lo hace tu clase)
    pre.cargar_datos_limpios()
    pre.clasificar_variables()
    pre.preparar_variables_modelado()
    pre.dividir_train_test(test_size=0.3, random_state=0)
    pre.inicializar_transformer()
    pre.entrenar_transformer()
    pre.transformar_variables()

    # Mock de SMOTETomek para que el test sea determinista/rápido
    class FakeSmoteTomek:
        def __init__(self, *a, **k): pass
        def fit_resample(self, X, y):  # no cambia nada, solo devuelve
            return X, y

    with patch("features.preprocessing.SMOTETomek", FakeSmoteTomek):
        pre.ejecutar_oversampling(random_state=42)

    pre.guardar_artefactos()
    pre.guardar_datasets()

    # Contratos
    assert set(pre.X_train_final.columns) == set(pre.X_test_final.columns), "Train/Test deben tener mismas columnas finales"
    assert not pre.X_train_final.isna().any().any(), "No debe haber NaNs en X_train_final"
    assert not pre.X_test_final.isna().any().any(),  "No debe haber NaNs en X_test_final"
    # Al guardar, deben existir CSVs
    pre.guardar_datasets()
    assert (Path(dirs["processed"]) / "X_train.csv").exists()
    assert (Path(dirs["processed"]) / "y_train.csv").exists()



# Test 3: Prueba de entrenador integración completa con mocks
class _DummyRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

class _DummyMLflow:
    def set_tracking_uri(self, *a, **k): pass
    def set_experiment(self, *a, **k): pass
    def start_run(self, *a, **k): return _DummyRun()
    def end_run(self): pass
    def log_param(self, *a, **k): pass
    def log_params(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    class sklearn:
        @staticmethod
        def log_model(*a, **k): pass

class _FakeXGB:
    def __init__(self, **_k): pass
    def fit(self, X, y): return self
    def predict_proba(self, X):
        X = np.asarray(X)
        p = (X[:, 0] > X[:, 1]).astype(float)*0.8 + 0.1
        return np.vstack([1-p, p]).T

def _write_minimal_processed(dirs):
    X_train = pd.DataFrame({"f1":[0,1,1,0,1], "f2":[1,1,0,0,1]})
    y_train = pd.Series([0,1,1,0,1], name="y")
    X_test  = pd.DataFrame({"f1":[0,1], "f2":[1,0]})
    y_test  = pd.Series([0,1], name="y")
    p = Path(dirs["processed"])
    X_train.to_csv(p/"X_train.csv", index=False)
    X_test.to_csv(p/"X_test.csv", index=False)
    y_train.to_csv(p/"y_train.csv", index=False)
    y_test.to_csv(p/"y_test.csv", index=False)

def test_trainer_end_to_end_with_mocks(dirs, monkeypatch):
    _write_minimal_processed(dirs)

    trainer = ModelTrainer(
        data_dir=str(dirs["processed"]),
        output_dir_reports=str(dirs["reports"]),
        output_dir_figures=str(dirs["figures"]),
    )

    # parchea mlflow y XGB para evitar calculos reales.
    monkeypatch.setattr(tm, "mlflow", _DummyMLflow())
    monkeypatch.setattr(tm, "XGBClassifier", _FakeXGB)

    # Entrena/evalúa baseline.
    trainer.cargar_datos()
    trainer.definir_modelos_baseline()
    trainer.entrenar_y_evaluar_baseline()

    # Guardado de resultados
    trainer.guardar_resultados()

    # Contratos: archivos escritos
    assert (Path(dirs["reports"]) / "resultados_train.json").exists()
    assert (Path(dirs["reports"]) / "resultados_test.json").exists()
    assert (Path(dirs["reports"]) / "metrics.json").exists()

    # Figuras: intercepta savefig para no escribir a disco real
    import matplotlib.pyplot as plt
    saved = []
    def fake_savefig(self, path, *a, **k):
        saved.append(path)
    monkeypatch.setattr(plt.Figure, "savefig", fake_savefig, raising=False)

    trainer.generar_matrices_confusion()
    assert saved, "Se esperaba al menos una figura de matriz de confusión"