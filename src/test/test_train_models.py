import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pytest
from unittest.mock import MagicMock

from models.train_model import ModelTrainer

# --- Fixtures ---

# Crear un conjunto de datos mínimo para pruebas
@pytest.fixture
def tiny_data(tmp_path):
    """Crea CSVs mínimos de X_train/X_test/y_train/y_test en data_dir."""
    data_dir = tmp_path / "data" / "processed"
    figs_dir = tmp_path / "reports" / "figures"
    reps_dir = tmp_path / "reports"
    data_dir.mkdir(parents=True,exist_ok=True)
    figs_dir.mkdir(parents=True,exist_ok=True)
    reps_dir.mkdir(parents=True,exist_ok=True)

    # dataset binario sencillo y estable para pruebas
    X_train = pd.DataFrame({"f1": [0,1,1,0,1,0], "f2": [1,1,0,0,1,0]})
    y_train = pd.Series([0,1,1,0,1,0], name="y")
    X_test  = pd.DataFrame({"f1": [0,1], "f2": [1,0]})
    y_test  = pd.Series([0,1], name="y")

    X_train.to_csv(data_dir/"X_train.csv", index=False)
    X_test.to_csv(data_dir/"X_test.csv", index=False)
    y_train.to_csv(data_dir/"y_train.csv", index=False)
    y_test.to_csv(data_dir/"y_test.csv", index=False)

    return {
        "data_dir": str(data_dir),
        "reports": str(reps_dir),
        "figures": str(figs_dir)
    }

# Fixture para crear un ModelTrainer con los datos mínimos
@pytest.fixture
def trainer(tiny_data):
    return ModelTrainer(
        data_dir=tiny_data["data_dir"],
        output_dir_reports=tiny_data["reports"],
        output_dir_figures=tiny_data["figures"],
    )


# --- Mocks ligeros para mlflow/optuna/xgboost ---

# Mocks para mlflow - clases y métodos mínimos necesarios
class _DummyRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

class _DummyMLflow:
    def set_tracking_uri(self, *_a, **_k): pass
    def set_experiment(self, *_a, **_k): pass
    def start_run(self, *_a, **_k): return _DummyRun()
    def end_run(self): pass
    def log_param(self, *_a, **_k): pass
    def log_params(self, *_a, **_k): pass
    def log_metric(self, *_a, **_k): pass
    class sklearn:
        @staticmethod
        def log_model(*_a, **_k): pass
    class xgboost:
        @staticmethod
        def log_model(*_a, **_k): pass

@pytest.fixture
def mock_mlflow(monkeypatch):
    dummy = _DummyMLflow()
    import models.train_model as train_model
    monkeypatch.setattr(train_model, "mlflow", dummy)
    return dummy

# Mocks para optuna - clases y métodos mínimos necesarios
# Se usa en la optimización de hiperparámetros de XGBoost
@pytest.fixture
def mock_optuna(monkeypatch):
    class _DummyStudy:
        best_value = 0.9
        best_params = {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
            "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 15
        }
        def optimize(self, *a, **k): pass
        def trials_dataframe(self): 
            return pd.DataFrame([{"trial":0, "value":0.9}])
    class _Optuna:
        @staticmethod
        def create_study(direction="maximize"): return _DummyStudy()
    import models.train_model as train_model
    monkeypatch.setattr(train_model, "optuna", _Optuna())

# Mocks para XGBoost - clase mínima necesaria
# Se usa en el modelo XGBoost balanceado
@pytest.fixture
def mock_xgb(monkeypatch):
    class _FakeXGB:
        def __init__(self, **_k): pass
        def fit(self, X, y): return self
        def predict_proba(self, X):
            import numpy as np
            # probabilidad simple basada en la 1a columna para ser determinista
            p = (np.array(X)[:, 0] > 0).astype(float) * 0.8 + 0.1
            return np.vstack([1-p, p]).T
    import models.train_model as train_model
    monkeypatch.setattr(train_model, "XGBClassifier", _FakeXGB)

#--- Tests ---

# test 1: Verificar que los datos se cargan correctamente
def test_cargar_datos_y_distribucion(trainer):
    trainer.cargar_datos()
    assert trainer.X_train.shape[0] > 0
    assert trainer.X_test.shape[0] > 0
    assert trainer.y_train.shape[0] > 0
    assert trainer.y_test.shape[0] > 0

#test 1.2: verificar los modelos baseline y balanceados y verificar que se agregan correctamente
def test_definir_modelos_baseline_y_balanceados(trainer):
    trainer.definir_modelos_baseline()
    assert {"Logistic Regression","KNN","Decision Tree","SVM"} <= set(trainer.modelos.keys())

    trainer.definir_modelos_balanceados()
    assert {"LogReg (balanced)","SVM (balanced)","Random Forest (balanced)","XGBoost (balanced)"} <= set(trainer.modelos.keys())


#test 2: Verificar que la evaluación de modelos funciona correctamente con un modelo simple
def test_evalua_modelos_privado_con_logreg(trainer):
    trainer.cargar_datos()
    modelos = {"LR": LogisticRegression(max_iter=200)}
    tr, te, entrenados = trainer._evalua_modelos(
        modelos,
        trainer.X_train.values, trainer.y_train,
        trainer.X_test.values, trainer.y_test
    )
    assert set(tr.columns) >= {"model","accuracy","precision","recall","f1","roc_auc"}
    assert set(te.columns) >= {"model","accuracy","precision","recall","f1","roc_auc"}
    assert set(entrenados.keys()) == {"LR"}

#Test 3: Verificar que el entrenamiento y evaluación de modelos baseline funciona correctamente
def test_entrenar_y_evaluar_baseline(trainer, mock_mlflow):
    trainer.cargar_datos()
    trainer.definir_modelos_baseline()
    trainer.entrenar_y_evaluar_baseline() 
    assert trainer.resultados_train is not None
    assert trainer.resultados_test is not None
    assert len(trainer.modelos_entrenados) > 0

#Test 4: generar matrices de confusión y verificar que se guardan como PNGs
def test_generar_matrices_confusion_guarda_pngs(trainer, mock_mlflow, monkeypatch):
    trainer.cargar_datos()
    trainer.definir_modelos_baseline()
    trainer.entrenar_y_evaluar_baseline()

    saved = []
    def fake_savefig(self, path, *a, **k):
        saved.append(path)
    # parchea savefig en la clase Figure para no escribir a disco realmente
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt.Figure, "savefig", fake_savefig, raising=False)

    trainer.generar_matrices_confusion()
    assert len(saved) >= 1  # al menos una figura por modelo

#Test 5: Verificar que la optimización de hiperparámetros con Optuna funciona correctamente con mocks
def test_optimizar_modelo_optuna_mock(trainer, mock_mlflow, mock_optuna, mock_xgb):
    trainer.cargar_datos()
    model = trainer.optimizar_modelo_optuna(n_trials=3)  # usa XGB y Optuna mockeados
    assert trainer.mejor_modelo is not None
    # archivo de resultados del estudio
    import os
    assert os.path.exists(trainer.output_dir_reports + "/optuna_trials.csv")


# Test 6: Verificar que el ajuste de umbral F1 funciona correctamente con un modelo simple
class _ToyModel:
    def fit(self, X, y): return self
    def predict_proba(self, X):
        import numpy as np
        p = (np.array(X)[:, 0] > 0).astype(float)*0.9 + 0.05
        return np.vstack([1-p, p]).T

def test_ajustar_umbral_f1_con_modelo_fake(trainer, monkeypatch):
    trainer.cargar_datos()

    # Evita Optuna real, devuelve un modelo determinista
    monkeypatch.setattr(trainer, "optimizar_modelo_optuna", lambda n_trials=30: _ToyModel())

    trainer.ajustar_umbral_f1()
    assert trainer.mejor_modelo is not None
    assert 0.0 <= trainer.mejor_umbral <= 1.0

# Test 7: Verificar que los resultados se guardan correctamente en archivos JSON
def test_guardar_resultados(trainer):
    # Prepara resultados mínimos para poder persistir
    trainer.resultados_train = pd.DataFrame([{"model":"LR","accuracy":100,"precision":100,"recall":100,"f1":100,"roc_auc":100}])
    trainer.resultados_test  = pd.DataFrame([{"model":"LR","accuracy":100,"precision":100,"recall":100,"f1":100,"roc_auc":100}])
    trainer.mejor_umbral = 0.42

    trainer.guardar_resultados()
    assert os.path.exists(trainer.output_dir_reports + "/resultados_train.json")
    assert os.path.exists(trainer.output_dir_reports + "/resultados_test.json")
    assert os.path.exists(trainer.output_dir_reports + "/metrics.json")