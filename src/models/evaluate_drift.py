import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from src.features import preprocessing  # mismo módulo que usa best_model


# === CONFIGURACIÓN DE RUTAS ===
SRC_PATH = os.path.join("data", "interim", "insurance_clean.csv")
OUTPUT_DIR_PROCESSED = os.path.join("data", "processed")
OUTPUT_DIR_REPORTS = "reports"
OUTPUT_DIR_FIGURES = os.path.join("reports", "figures")
OUTPUT_DIR_MODELS = "models"
OUTPUT_DIR_REFERENCES = "references"

# Usa el MISMO pipeline que se carga en FastAPI
# Si en CaravanInsuranceModelService.py cargan otro path, copia ese aquí.
MODEL_PATH = os.path.join("src", "serving", "pipeline.joblib")
# Alternativa si usan el de raíz:
# MODEL_PATH = "pipeline.joblib"

os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)

# Umbral de alerta para caída de performance (en puntos de F1)
ALERT_THRESHOLD = 0.03


def load_data():
    """
    Usa el mismo DataPreprocessor que best_model.py para construir X_test e y_test
    con las columnas que el pipeline espera.
    """
    preprocessor = preprocessing.DataPreprocessor(
        src_path=SRC_PATH,
        output_dir_processed=OUTPUT_DIR_PROCESSED,
        output_dir_reports=OUTPUT_DIR_REPORTS,
        output_dir_figures=OUTPUT_DIR_FIGURES,
        output_dir_models=OUTPUT_DIR_MODELS,
        output_dir_references=OUTPUT_DIR_REFERENCES
    )

    print("Cargando datos de test...")
    preprocessor.cargar_datos_limpios()
    preprocessor.clasificar_variables()
    preprocessor.colapsar_categorias_raras(prop_min=0.05)
    preprocessor.preparar_variables_modelado()
    preprocessor.dividir_train_test(test_size=0.20, random_state=42)

    X_test = preprocessor.X_test
    y_test = preprocessor.y_test

    return X_test, y_test


def load_model():
    """Carga el pipeline entrenado (el mismo que usa FastAPI)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    print("Cargando modelo/pipeline...")
    model = joblib.load(MODEL_PATH)
    return model


def evaluate_baseline(model, X_test, y_test):
    """Evalúa el modelo en los datos originales (sin drift)."""
    print("Evaluando data drift...")
    print("=== Evaluación baseline ===")
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"Baseline F1: {score:.4f}")
    return score


def generate_drifted_data(X_test):
    """
    Genera una copia de X_test con drift sintético en algunas columnas.

    - MOSTYPE: se fuerza a que ~70% de las observaciones tomen la categoría 8.
    - MINK4575, MINK7512: se incrementan en +2 para simular un cambio en nivel socioeconómico.
    """
    print("\n=== Generando datos con drift sintético ===")
    X_drift = X_test.copy()

    # Drift en tipo de vecindario
    if "MOSTYPE" in X_drift.columns:
        X_drift["MOSTYPE"] = np.where(
            np.random.rand(len(X_drift)) < 0.7,
            8,
            X_drift["MOSTYPE"]
        )

    # Drift en variables socioeconómicas / ingreso
    for col in ["MINK4575", "MINK7512"]:
        if col in X_drift.columns:
            X_drift[col] = X_drift[col] + 2

    return X_drift


def plot_feature_drift(X_original, X_drift, column, filename):
    """Guarda un histograma comparando la distribución de una columna antes y después del drift."""
    if column not in X_original.columns or column not in X_drift.columns:
        print(f"La columna {column} no existe en X_test, se omite gráfica.")
        return

    plt.figure()
    X_original[column].hist(alpha=0.5, label="Original")
    X_drift[column].hist(alpha=0.5, label="Drift")
    plt.legend()
    plt.title(f"Distribución de {column} antes y después del drift")
    out_path = os.path.join(OUTPUT_DIR_FIGURES, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfica de drift guardada en: {out_path}")


def evaluate_drift(model, X_test, y_test):
    """Compara desempeño baseline vs datos con drift, aplica umbral de alerta y guarda gráficas."""
    baseline = evaluate_baseline(model, X_test, y_test)

    X_drift = generate_drifted_data(X_test)

    print("\n=== Evaluando modelo con drift ===")
    y_pred_drift = model.predict(X_drift)
    drift_score = f1_score(y_test, y_pred_drift)
    print(f"F1 con drift: {drift_score:.4f}")

    perf_drop = baseline - drift_score
    print(f"Pérdida de performance (baseline - drift): {perf_drop:.4f}")

    # Umbral y criterio de alerta
    if perf_drop >= ALERT_THRESHOLD:
        print(
            f"⚠ ALERTA: La caída de F1 ({perf_drop:.4f}) supera el umbral de {ALERT_THRESHOLD:.4f}."
        )
        print("   Acción sugerida: revisar drift en features clave y considerar reentrenar el modelo.")
    else:
        print("✅ Sin alerta: la caída de performance está dentro del umbral aceptable.")

    # Guardar 1–2 gráficas de ejemplo
    for col in ["MOSTYPE", "MINK4575", "MINK7512"]:
        plot_feature_drift(X_test, X_drift, col, f"drift_{col}.png")


if __name__ == "__main__":
    X_test, y_test = load_data()
    model = load_model()
    evaluate_drift(model, X_test, y_test)
