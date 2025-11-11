# =========================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELO
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar datasets procesados desde data/processed/
# - Entrenar mejor modelo de clasificación
# - Evaluar modelo en train y test
# - Generar matriz de confusión
# - Ajustar umbral de decisión para maximizar F1
# - Guardar resultados y métricas en reports/
# - Guardar gráficas en reports/figures/
# =========================================================

import sys, os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.xgboost
import joblib
from src.features import preprocessing
from imblearn.combine import SMOTETomek
# =========================================================
# CLASE PRINCIPAL: ModelTrainer
# =========================================================
class ModelTrainer:

    def __init__(self, data_dir, output_dir_reports, output_dir_figures):
        # Atributos de rutas
        self.data_dir = data_dir
        self.output_dir_reports = output_dir_reports
        self.output_dir_figures = output_dir_figures

        # Crear directorios si no existen
        os.makedirs(self.output_dir_reports, exist_ok=True)
        os.makedirs(self.output_dir_figures, exist_ok=True)

        # Atributos para almacenar datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Modelos y resultados
        self.modelo = {}
        self.modelo_entrenado = {}
        self.resultados_train = None
        self.resultados_test = None

        self.src_path = os.path.join("data", "interim", "insurance_clean.csv")

        self.preprocess=None

    def cargar_datos(self):
        # Cargar datasets
        self.X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
        self.X_test = pd.read_csv(os.path.join(self.data_dir, 'X_test.csv'))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv')).values.ravel()
        self.y_test = pd.read_csv(os.path.join(self.data_dir, 'y_test.csv')).values.ravel()

        print(f"✅ Datos cargados:")
        print(f"   - X_train: {self.X_train.shape}")
        print(f"   - X_test: {self.X_test.shape}")
        print(f"   - y_train: {self.y_train.shape}")
        print(f"   - y_test: {self.y_test.shape}")

        # Mostrar distribución de clases
        pos_train = (self.y_train == 1).sum()
        pos_test = (self.y_test == 1).sum()
        total_train = len(self.y_train)
        total_test = len(self.y_test)

        print(f"\n📊 Distribución de clases:")
        print(
            f"   Train - Positivos: {pos_train} ({100 * pos_train / total_train:.1f}%) | Negativos: {total_train - pos_train} ({100 * (1 - pos_train / total_train):.1f}%)")
        print(
            f"   Test  - Positivos: {pos_test} ({100 * pos_test / total_test:.1f}%) | Negativos: {total_test - pos_test} ({100 * (1 - pos_test / total_test):.1f}%)")

    def definir_modelo(self):

        mejor_modelo = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=15,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )

        self.mejor_modelo = mejor_modelo
        self.mejor_umbral = None

        SRC_PATH = os.path.join("data", "interim", "insurance_clean.csv")
        OUTPUT_DIR_PROCESSED = os.path.join("data", "processed")
        OUTPUT_DIR_REPORTS = "reports"
        OUTPUT_DIR_FIGURES = os.path.join("reports", "figures")
        OUTPUT_DIR_MODELS = "models"
        OUTPUT_DIR_REFERENCES = "references"

        # Crear instancia del preprocesador
        preprocessor = preprocessing.DataPreprocessor(
            src_path=SRC_PATH,
            output_dir_processed=OUTPUT_DIR_PROCESSED,
            output_dir_reports=OUTPUT_DIR_REPORTS,
            output_dir_figures=OUTPUT_DIR_FIGURES,
            output_dir_models=OUTPUT_DIR_MODELS,
            output_dir_references=OUTPUT_DIR_REFERENCES
        )
        # 1. Carga y clasificación
        preprocessor.cargar_datos_limpios()
        preprocessor.clasificar_variables()

        # 2. Colapsar categorías raras
        preprocessor.colapsar_categorias_raras(prop_min=0.05)



        # 4. Preparación para modelado
        preprocessor.preparar_variables_modelado()
        preprocessor.dividir_train_test(test_size=0.20, random_state=42)

        # 5. Transformaciones
        preprocessor.inicializar_transformer()

        smote_tomek = SMOTETomek(sampling_strategy=0.1, random_state=42)
        # pipeline
        pipe = Pipeline([
            ('preprocessor', preprocessor.transformer),
            ('smote_tomek', smote_tomek),
            ('classifier', self.mejor_modelo)
        ])

        # Fit your pipeline as usual
        pipe.fit(preprocessor.X_train, preprocessor.y_train)

        # Save the entire pipeline
        joblib.dump(pipe, 'pipeline.joblib')
        # Later: Load your pipeline back
        loaded_pipe = joblib.load('pipeline.joblib')

        # Make predictions as usual
        y_pred = loaded_pipe.predict(preprocessor.X_test)

        # Get predicted probabilities for positive class (needed for auc/roc)
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            y_proba = pipe.predict_proba(preprocessor.X_test)[:, 1]  # Assumes binary classifier
        else:
            # For models with no predict_proba, use decision_function if available
            y_proba = pipe.decision_function(preprocessor.X_test)

        # Metrics
        # Calculate metrics and multiply by 100 to get percentages
        accuracy = accuracy_score(preprocessor.y_test, y_pred) * 100
        precision = precision_score(preprocessor.y_test, y_pred) * 100
        recall = recall_score(preprocessor.y_test, y_pred) * 100
        f1 = f1_score(preprocessor.y_test, y_pred) * 100
        auc = roc_auc_score(preprocessor.y_test, y_proba) * 100

        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")
        print(f"F1 Score:  {f1:.2f}%")
        print(f"AUC:       {auc:.2f}%")

        mejor_modelo.fit(self.X_train, self.y_train)
        resultados_train = []
        resultados_test = []

        # Predecir en train
        y_pred_train = self.mejor_modelo.predict(self.X_train)

        # Predecir en test
        y_pred = self.mejor_modelo.predict(self.X_test)

        # Calcular AUC si el modelo tiene predict_proba
        auc_train = np.nan
        auc_test = np.nan


        average = 'binary'
        if hasattr(self.mejor_modelo, 'predict_proba'):
            y_prob_train = self.mejor_modelo.predict_proba(self.X_train)[:, 1]
            y_prob = self.mejor_modelo.predict_proba(self.X_test)[:, 1]
            auc_train = roc_auc_score(self.y_train, y_prob_train)
            auc_test = roc_auc_score(self.y_test, y_prob)

        # Métricas de entrenamiento
        metrics_train = {
                'model': 'XGBClassifier',
                'accuracy': round(accuracy_score(self.y_train, y_pred_train) * 100, 2),
                'precision': round(precision_score(self.y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'recall': round(recall_score(self.y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'f1': round(f1_score(self.y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'roc_auc': round(auc_train * 100, 2) if not np.isnan(auc_train) else np.nan
            }

        # Métricas de prueba
        metrics_test = {
                'model': 'XGBClassifier',
                'accuracy': round(accuracy_score(self.y_test, y_pred) * 100, 2),
                'precision': round(precision_score(self.y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'recall': round(recall_score(self.y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'f1': round(f1_score(self.y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'roc_auc': round(auc_test * 100, 2) if not np.isnan(auc_test) else np.nan
            }

        resultados_train.append(metrics_train)
        resultados_test.append(metrics_test)

        # Convertir a DataFrames
        self.resultados_train = pd.DataFrame(resultados_train).sort_values(by='model', ascending=False).reset_index(
            drop=True)
        self.resultados_test = pd.DataFrame(resultados_test).sort_values(by='model', ascending=False).reset_index(
            drop=True)

        print(resultados_test)
        print(resultados_train)



    def guardar_resultados(self):
        print("\n💾 Guardando resultados...")

        # Guardar resultados de entrenamiento
        train_path = os.path.join(self.output_dir_reports, 'resultados_train.json')
        self.resultados_train.to_json(train_path, orient='columns', indent=4)
        print(f"  ✅ Resultados train: {train_path}")

        # Guardar resultados de prueba
        test_path = os.path.join(self.output_dir_reports, 'resultados_test.json')
        self.resultados_test.to_json(test_path, orient='columns', indent=4)
        print(f"  ✅ Resultados test: {test_path}")

        # Guardar métricas generales
        metrics = {
            "baseline": {
                "train": self.resultados_train.to_dict('records'),
                "test": self.resultados_test.to_dict('records')
            },
            "mejor_umbral": float(self.mejor_umbral) if self.mejor_umbral else None
        }

        metrics_path = os.path.join(self.output_dir_reports, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Métricas generales: {metrics_path}")

    def ejecutar_entrenamiento_completo(self):
        print("=" * 60)
        print("INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("=" * 60)

        # 1. Cargar datos
        self.cargar_datos()

        # 3. Modelos balanceados
        self.definir_modelo()
        #self.entrenar_y_evaluar_balanceados()

        # 4. Generar matrices de confusión
        #self.generar_matrices_confusion()

        # 5. Ajustar umbral F1
        #self.ajustar_umbral_f1()

        # 6. Guardar resultados
        self.guardar_resultados()

        print("=" * 60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================
def main():
    # Definir rutas según estructura de Cookiecutter
    DATA_DIR = os.path.join("data", "processed")


    # Definir rutas según estructura de Cookiecutter
    SRC_PATH = os.path.join("data", "interim", "insurance_clean.csv")
    OUTPUT_DIR_PROCESSED = os.path.join("data", "processed")
    OUTPUT_DIR_REPORTS = "reports"
    OUTPUT_DIR_FIGURES = "reports/figures"
    OUTPUT_DIR_MODELS = "models"
    OUTPUT_DIR_REFERENCES = "references"



    # Crear instancia del entrenador
    trainer = ModelTrainer(
        data_dir=DATA_DIR,
        output_dir_reports=OUTPUT_DIR_REPORTS,
        output_dir_figures=OUTPUT_DIR_FIGURES
    )

    # Ejecutar entrenamiento completo
    trainer.ejecutar_entrenamiento_completo()


if __name__ == "__main__":
    main()