# =========================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar datasets procesados desde data/processed/
# - Entrenar múltiples modelos de clasificación
# - Evaluar modelos en train y test
# - Generar matrices de confusión
# - Ajustar umbral de decisión para maximizar F1
# - Guardar resultados y métricas en reports/
# - Guardar gráficas en reports/figures/
# =========================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.xgboost

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
        
        # Configuración de MLflow local
        mlflow.set_tracking_uri("file:./mlruns")  
        mlflow.set_experiment("XGBoost_Optuna_COIL2000") 
        
        # Atributos para almacenar datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Modelos y resultados
        self.modelos = {}
        self.modelos_entrenados = {}
        self.resultados_train = None
        self.resultados_test = None
        
        # Mejor modelo y umbral
        self.mejor_modelo = None
        self.mejor_umbral = None
    
    
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
        print(f"   Train - Positivos: {pos_train} ({100*pos_train/total_train:.1f}%) | Negativos: {total_train-pos_train} ({100*(1-pos_train/total_train):.1f}%)")
        print(f"   Test  - Positivos: {pos_test} ({100*pos_test/total_test:.1f}%) | Negativos: {total_test-pos_test} ({100*(1-pos_test/total_test):.1f}%)")
    
    
    def definir_modelos_baseline(self):
        # Definir modelos baseline
        self.modelos = {
            'Logistic Regression': LogisticRegression(
                penalty=None, 
                solver='lbfgs', 
                max_iter=2000, 
                random_state=1
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        print(f"\n📋 Modelos baseline definidos: {len(self.modelos)}")
    
    
    def entrenar_y_evaluar_baseline(self):
        print("\n🔄 Entrenando modelos baseline...")
        
        # Convertir a arrays numpy
        x_train = self.X_train.values
        x_test = self.X_test.values
        
        # Entrenar y evaluar
        self.resultados_train, self.resultados_test, self.modelos_entrenados = self._evalua_modelos(
            self.modelos, x_train, self.y_train, x_test, self.y_test
        )
        
        print("\n📊 Resultados en entrenamiento:")
        print(self.resultados_train.to_string(index=False))
        
        print("\n📊 Resultados en prueba:")
        print(self.resultados_test.to_string(index=False))
    
    
    def definir_modelos_balanceados(self):
        # Definir modelos con balanceo
        self.modelos = {
            'LogReg (balanced)': LogisticRegression(
                penalty='l2',
                C=0.1,
                solver='lbfgs',
                max_iter=4000,
                class_weight='balanced',
                random_state=42
            ),
            'SVM (balanced)': SVC(
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest (balanced)': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight='balanced_subsample',
                random_state=42
            ),
            'XGBoost (balanced)': XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=15,  
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        }

        
        print(f"\n📋 Modelos balanceados definidos: {len(self.modelos)}")
    
    
    def entrenar_y_evaluar_balanceados(self):
        print("\n🔄 Entrenando modelos con class_weight='balanced'...")
        
        # Convertir a arrays numpy
        x_train = self.X_train.values
        x_test = self.X_test.values
        
        # Entrenar y evaluar
        self.resultados_train, self.resultados_test, self.modelos_entrenados = self._evalua_modelos(
            self.modelos, x_train, self.y_train, x_test, self.y_test
        )
        
        print("\n📊 Resultados en entrenamiento (balanced):")
        print(self.resultados_train.to_string(index=False))
        
        print("\n📊 Resultados en prueba (balanced):")
        print(self.resultados_test.to_string(index=False))
    
    
    def generar_matrices_confusion(self):
        print("\n📊 Generando matrices de confusión...")
        
        # Convertir a arrays numpy
        x_test = self.X_test.values
        
        # Generar matriz para cada modelo
        for name, model in self.modelos_entrenados.items():
            # Predecir en test
            y_pred = model.predict(x_test)
            
            # Crear matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['No póliza', 'Sí póliza']
            )
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            
            # Configurar título y guardar
            ax.set_title(f'Matriz de Confusión - {name}')
            fig.savefig(
                os.path.join(self.output_dir_figures, f'cm_counts_{name.replace(" ", "_")}.png'),
                bbox_inches='tight'
            )
            plt.close(fig)
        
        print(f"  ✅ Matrices guardadas en: {self.output_dir_figures}")
    
    def optimizar_modelo_optuna(self, n_trials=30):
        print("\n🔎 Iniciando optimización de hiperparámetros con Optuna...")

        def objective(trial):
            # Definir espacio de búsqueda a partir del mejor modelo actual
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_int('scale_pos_weight', 10, 25),
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42
            }

            # Modelo con los parámetros actuales
            model = XGBClassifier(**params)

            # Validación cruzada (3-fold CV) usando F1 como métrica
            f1 = cross_val_score(model, self.X_train, self.y_train, scoring='f1', cv=3, n_jobs=-1).mean()
            return f1

        # Crear el estudio de optimización
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Mostrar mejores resultados
        print("\n✅ Optimización completada.")
        print(f"🏆 Mejor F1 promedio en validación: {study.best_value:.4f}")
        print("📊 Mejor combinación encontrada:")
        for key, val in study.best_params.items():
            print(f"   - {key}: {val}")

        # Entrenar el mejor modelo final con los parámetros óptimos
        best_params = study.best_params
        best_params.update({
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        })

        mejor_modelo = XGBClassifier(**best_params)
        mejor_modelo.fit(self.X_train, self.y_train)
        self.mejor_modelo = mejor_modelo
        self.mejor_umbral = None  
        
        # =========================================================
        # REGISTRO EN MLFLOW
        # =========================================================
        with mlflow.start_run(run_name="XGBoost_Optuna_Best"):
            mlflow.log_params(study.best_params)

            y_prob = mejor_modelo.predict_proba(self.X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = f1_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_prob)
    
            mlflow.log_metric("f1_test", f1)
            mlflow.log_metric("recall_test", recall)
            mlflow.log_metric("precision_test", precision)
            mlflow.log_metric("roc_auc_test", auc)

            mlflow.xgboost.log_model(mejor_modelo, artifact_path="xgboost_model")

            if self.mejor_umbral is not None:
                mlflow.log_param("optimal_threshold", self.mejor_umbral)
        
        print("✅ Modelo y resultados registrados en MLflow.")
        mlflow.end_run()
        # =========================================================
        
        # Guardar resultados del estudio
        study.trials_dataframe().to_csv(os.path.join(self.output_dir_reports, 'optuna_trials.csv'), index=False)
        print(f"💾 Resultados de búsqueda guardados en: {self.output_dir_reports}/optuna_trials.csv")

        return self.mejor_modelo
    
    def ajustar_umbral_f1(self):
        print("\n🎯 Ajustando umbral para maximizar F1...")
        
        # Convertir a arrays numpy
        x_train = self.X_train.values
        x_test = self.X_test.values
        
        # Crear split de validación desde train
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train, self.y_train,
            test_size=0.25,
            stratify=self.y_train,
            random_state=42
        )
        
        # Re-entrenar con el mejor modelo detectado
        mejor_modelo = self.optimizar_modelo_optuna(n_trials=30)

        mejor_modelo.fit(X_tr, y_tr)
        
        # Encontrar umbral óptimo en validación
        probs_val = mejor_modelo.predict_proba(X_val)[:, 1]
        p, r, thr = precision_recall_curve(y_val, probs_val)
        f1 = 2 * (p * r) / (p + r + 1e-12)
        idx = f1.argmax()
        self.mejor_umbral = thr[max(idx - 1, 0)] if idx > 0 else 0.5
        
        print(f"\n  📍 Umbral óptimo en VALID: {self.mejor_umbral:.3f}")
        print(f"     - Precision: {p[idx]:.2f}")
        print(f"     - Recall: {r[idx]:.2f}")
        print(f"     - F1: {f1[idx]:.2f}")
        
        # Evaluar en test con umbral ajustado
        probs_test = mejor_modelo.predict_proba(x_test)[:, 1]
        ap_test = average_precision_score(self.y_test, probs_test)
        y_pred = (probs_test >= self.mejor_umbral).astype(int)
        
        print(f"\n  📊 Evaluación en TEST:")
        print(f"     - Average Precision (PR-AUC): {ap_test:.3f}")
        print(f"\n  📋 Classification Report (LogReg balanced con umbral F1):")
        print(classification_report(
            self.y_test, y_pred,
            target_names=['No póliza', 'Sí póliza'],
            zero_division=0
        ))
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        print(f"\n  📊 Matriz de Confusión [[TN FP] [FN TP]]:")
        print(cm)
        
        # Guardar modelo ajustado
        self.mejor_modelo = mejor_modelo
    
    
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
        
        # 2. Modelos baseline (sin balanceo)
        self.definir_modelos_baseline()
        self.entrenar_y_evaluar_baseline()
        
        # 3. Modelos balanceados
        self.definir_modelos_balanceados()
        self.entrenar_y_evaluar_balanceados()
        
        # 4. Generar matrices de confusión
        self.generar_matrices_confusion()
        
        # 5. Ajustar umbral F1
        self.ajustar_umbral_f1()
        
        # 6. Guardar resultados
        self.guardar_resultados()
        
        print("=" * 60)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
    
    
    # =========================================================
    # MÉTODOS AUXILIARES (PRIVADOS)
    # =========================================================
    
    def _evalua_modelos(self, modelos, X_train, y_train, X_test, y_test, average='binary'):
        resultados_train = []
        resultados_test = []
        modelos_entrenados = {}
        
        # Para cada modelo
        for name, model in modelos.items():
            # Entrenar
            model.fit(X_train, y_train)
            modelos_entrenados[name] = model
            
            # Predecir en train
            y_pred_train = model.predict(X_train)
            
            # Predecir en test
            y_pred = model.predict(X_test)
            
            # Calcular AUC si el modelo tiene predict_proba
            auc_train = np.nan
            auc_test = np.nan
            
            if hasattr(model, 'predict_proba'):
                y_prob_train = model.predict_proba(X_train)[:, 1]
                y_prob = model.predict_proba(X_test)[:, 1]
                auc_train = roc_auc_score(y_train, y_prob_train)
                auc_test = roc_auc_score(y_test, y_prob)
            
            # Métricas de entrenamiento
            metrics_train = {
                'model': name,
                'accuracy': round(accuracy_score(y_train, y_pred_train) * 100, 2),
                'precision': round(precision_score(y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'recall': round(recall_score(y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'f1': round(f1_score(y_train, y_pred_train, average=average, zero_division=0) * 100, 2),
                'roc_auc': round(auc_train * 100, 2) if not np.isnan(auc_train) else np.nan
            }
            
            # Métricas de prueba
            metrics_test = {
                'model': name,
                'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'precision': round(precision_score(y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'recall': round(recall_score(y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'f1': round(f1_score(y_test, y_pred, average=average, zero_division=0) * 100, 2),
                'roc_auc': round(auc_test * 100, 2) if not np.isnan(auc_test) else np.nan
            }
            
            resultados_train.append(metrics_train)
            resultados_test.append(metrics_test)
        
        # Convertir a DataFrames
        resultados_df_train = pd.DataFrame(resultados_train).sort_values(by='model', ascending=False).reset_index(drop=True)
        resultados_df_test = pd.DataFrame(resultados_test).sort_values(by='model', ascending=False).reset_index(drop=True)
        
        return resultados_df_train, resultados_df_test, modelos_entrenados


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================
def main():
    # Definir rutas según estructura de Cookiecutter
    DATA_DIR = "data/processed"
    OUTPUT_DIR_REPORTS = "reports"
    OUTPUT_DIR_FIGURES = "reports/figures"
    
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