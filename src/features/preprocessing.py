# =========================================================
# PREPROCESAMIENTO DE DATOS
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar el CSV limpio desde data/interim/
# - Unir categorías poco frecuentes
# - Generar visualizaciones exploratorias en reports/figures/
# - Aplicar One-Hot Encoding a variables categóricas
# - Escalar variables numéricas
# - Dividir en train/test
# - Aplicar oversampling (SMOTETomek)
# - Guardar datasets procesados en data/processed/
# - Guardar artefactos (scaler, columnas) en models/ y references/
# =========================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline as ImbPipeline


# =========================================================
# CLASE PRINCIPAL: DataPreprocessor
# =========================================================
class DataPreprocessor:

    def __init__(self, src_path, output_dir_processed, output_dir_reports, 
                 output_dir_figures, output_dir_models, output_dir_references):
        # Atributos de rutas
        self.src_path = src_path
        self.output_dir_processed = output_dir_processed
        self.output_dir_reports = output_dir_reports
        self.output_dir_figures = output_dir_figures
        self.output_dir_models = output_dir_models
        self.output_dir_references = output_dir_references
        
        # Crear directorios si no existen
        os.makedirs(self.output_dir_processed, exist_ok=True)
        os.makedirs(self.output_dir_reports, exist_ok=True)
        os.makedirs(self.output_dir_figures, exist_ok=True)
        os.makedirs(self.output_dir_models, exist_ok=True)
        os.makedirs(self.output_dir_references, exist_ok=True)
        
        # Atributos para almacenar datos
        self.df = None
        
        # Clasificación de variables
        self.P_cols = []
        self.A_cols = []
        self.ordinal_small = []
        self.target_col = None
        self.cat_cols = []
        self.num_cols = []
        self.sociodem_cols = []
        
        # Artefactos de preprocesamiento
        self.scaler = None
        self.ohe = None
        self.ohe_cols = []

        #transformer preprocesamiento
        self.numericas_pipe=None
        self.numericas_pipe_nombres = []
        self.categoricas_pipe = None
        self.categoricas_pipe_nombres = []
        self.columnas_transformer = None
        self.X_train_transformed=[]
        self.X_test_transformed=[]
        self.pipeline_preprocesamiento=None

        #oversampling
        self.metodo_uo=None
        
        # Datasets resultantes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_final = None
        self.X_test_final = None
        self.X_train_resampled = None
        self.y_train_resampled = None
    
    
    def cargar_datos_limpios(self):
        # Cargar CSV limpio
        self.df = pd.read_csv(self.src_path, low_memory=False)
        print(f"✅ Cargado: {self.src_path} | Shape: {self.df.shape}")
    
    
    def clasificar_variables(self):
        # Detectar columnas P* (contribuciones)
        self.P_cols = [c for c in self.df.columns if isinstance(c, str) and c.startswith("P")]
        
        # Detectar columnas A* (número de pólizas)
        self.A_cols = [c for c in self.df.columns if isinstance(c, str) and c.startswith("A")]
        
        # Variables ordinales pequeñas
        self.ordinal_small = [c for c in ["MOSTYPE", "MOSHOOFD", "MGEMLEEF", "MGODRK", "PWAPART"] 
                             if c in self.df.columns]
        
        # Variable objetivo (binaria)
        self.target_col = "CARAVAN" if "CARAVAN" in self.df.columns else None
        
        # Sociodemográficas = resto (excluyendo P*, A*, ordinales y target)
        excluir = set(self.P_cols) | set(self.A_cols) | set(self.ordinal_small) | {self.target_col}
        self.sociodem_cols = [c for c in self.df.columns if c not in excluir]
        
        print(f"📋 Clasificación de variables:")
        print(f"   - P* (contribuciones): {len(self.P_cols)}")
        print(f"   - A* (pólizas): {len(self.A_cols)}")
        print(f"   - Ordinales: {len(self.ordinal_small)}")
        print(f"   - Sociodemográficas: {len(self.sociodem_cols)}")
    
    
    def colapsar_categorias_raras(self, prop_min=0.05):
        print(f"\n🔄 Colapsando categorías con frecuencia < {prop_min*100}%...")
        
        # Para cada variable ordinal
        for col in self.ordinal_small:
            print(f"\n  Procesando: {col}")
            print(f"  - Categorías antes: {self.df[col].nunique()}")
            
            # Colapsar categorías raras
            self.df[col] = self._colapsar_categorias(self.df[col], prop_min=prop_min)
            
            print(f"  - Categorías después: {self.df[col].nunique()}")
    
    
    def generar_visualizaciones(self):
        print("\n📊 Generando visualizaciones...")
        
        # 1. Distribución de la variable objetivo
        self._graficar_variable_objetivo()
        
        # 2. Categorías colapsadas
        self._graficar_categorias_colapsadas()
        
        # 3. Variables de pólizas (A*)
        self._graficar_polizas()
        
        # 4. Variables sociodemográficas
        self._graficar_sociodemograficas()
        
        print("✅ Visualizaciones guardadas en reports/figures/")
    
    
    def preparar_variables_modelado(self):
        # Variables categóricas: las ordinales colapsadas
        self.cat_cols = self.ordinal_small
        
        # Variables numéricas: todas excepto categóricas y target
        base_cols = [c for c in self.df.columns if c != self.target_col]
        self.num_cols = [c for c in base_cols if c not in self.cat_cols]
        
        print(f"\n[Preparación] Categóricas: {len(self.cat_cols)}")
        print(f"[Preparación] Numéricas: {len(self.num_cols)}")
    
    
    def dividir_train_test(self, test_size=0.20, random_state=42):
        # Separar features y target
        if self.target_col is not None:
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col].astype(int)
            
            # Split estratificado para mantener proporción de clases
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Si no hay target, split simple
            self.X_train, self.X_test = train_test_split(
                self.df, test_size=test_size, random_state=random_state
            )
            self.y_train = None
            self.y_test = None
        
        print(f"\n[Split] X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
        if self.y_train is not None:
            print(f"[Split] y_train positivos: {int(self.y_train.sum())} | y_test positivos: {int(self.y_test.sum())}")


    def inicializar_transformer(self):
        # ************* Inlcuye aquí tu código:*****************************

        # Variables numéricas:
        self.numericas_pipe = Pipeline(steps=[('impMediana', SimpleImputer(strategy='median')),
                                         ('escalaNum', StandardScaler())])
        self.numericas_pipe_nombres = self.num_cols

        # Variables categóricas:
        self.categoricas_pipe = Pipeline(steps=[('impModa', SimpleImputer(strategy='most_frequent')),
                                         ('ohe', OneHotEncoder( handle_unknown='ignore',
                                                                 drop='first',
                                                                sparse_output=False,
                                                                dtype=np.int8))])
        self.categoricas_pipe_nombres = self.cat_cols



        # Conjuntas las transformaciones de todo tipo de variable y
        # deja sin procesar aquellas que hayas decidido no transformar:

        self.columnas_transformer = ColumnTransformer(transformers=[('numpipe', self.numericas_pipe, self.numericas_pipe_nombres),
                                                              ('catpipe', self.categoricas_pipe, self.categoricas_pipe_nombres)],
                                                remainder='passthrough')

    #Se entrena el pipelice de preprocesar solo con el set de entrenamiento para evitar data leakage
    def entrenar_transformer(self):
        print("\n[Transformer X Train] ...")
        self.columnas_transformer.fit(self.X_train)
        feature_names = self.columnas_transformer.get_feature_names_out()
        print(feature_names)
        #print(self.columnas_transformer.fit(self.X_train))

    #Se transforman las variables usando el pipeline de preprocesamiento
    def transformar_variables(self):
        self.X_train_transformed = self.columnas_transformer.transform(self.X_train)
        self.X_test_transformed = self.columnas_transformer.transform(self.X_test)

        #print(X_train_transformed)
        #print(X_test_transformed)

    def aplicar_one_hot_encoding(self):
        print("\n[One-Hot Encoding] Codificando variables categóricas...")
        
        # Fit y transform en train
        X_train_cat, self.ohe = self._one_hot_fit_transform(self.X_train, self.cat_cols)
        
        # Transform en test
        X_test_cat = self._one_hot_transform_with_cols(self.X_test, self.cat_cols, self.ohe)
        
        # Guardar nombres de columnas OHE
        self.ohe_cols = self.ohe.get_feature_names_out(self.cat_cols).tolist()
        
        print(f"[One-Hot Encoding] Dummies generadas: {len(self.ohe_cols)} columnas")
        
        # Guardar datasets categóricos transformados
        self.X_train_cat = X_train_cat
        self.X_test_cat = X_test_cat
    
    
    def escalar_variables_numericas(self):
        print("\n[Escalado] Estandarizando variables numéricas...")
        
        # Crear scaler
        self.scaler = StandardScaler()
        
        # Copiar datos numéricos
        X_train_num = self.X_train[self.num_cols].copy()
        X_test_num = self.X_test[self.num_cols].copy()
        
        # Fit y transform en train, transform en test
        if len(self.num_cols) > 0:
            X_train_num[self.num_cols] = self.scaler.fit_transform(X_train_num[self.num_cols])
            X_test_num[self.num_cols] = self.scaler.transform(X_test_num[self.num_cols])
        
        print(f"[Escalado] Variables escaladas: {len(self.num_cols)}")
        
        # Guardar datasets numéricos escalados
        self.X_train_num = X_train_num
        self.X_test_num = X_test_num
    
    
    def ensamblar_datasets_finales(self):
        print("\n[Ensamble] Combinando variables numéricas y categóricas...")
        
        # Concatenar numéricas + categóricas
        self.X_train_final = pd.concat([
            self.X_train_num.reset_index(drop=True),
            self.X_train_cat.reset_index(drop=True)
        ], axis=1)
        
        self.X_test_final = pd.concat([
            self.X_test_num.reset_index(drop=True),
            self.X_test_cat.reset_index(drop=True)
        ], axis=1)
        
        # Alinear columnas de test con train (por si falta alguna categoría)
        self.X_test_final = self.X_test_final.reindex(columns=self.X_train_final.columns, fill_value=0)
        
        print(f"[Ensamble] X_train_final: {self.X_train_final.shape}, X_test_final: {self.X_test_final.shape}")
    
    
    def definir_oversampling(self, random_state=42):
        print("\n[SMOTE] Aplicando oversampling + limpieza de frontera (SMOTETomek)...")
    
        # Aplicar SMOTETomek en train
        self.metodo_uo = SMOTETomek(random_state=random_state)
        self.X_train_resampled, self.y_train_resampled = self.metodo_uo.fit_resample(
        self.X_train_final, self.y_train
        )
        print(self.X_train_final)
        print(f"[SMOTETomek] X_train_resampled: {self.X_train_resampled.shape}")
        print(f"[SMOTETomek] Clase 1: {sum(self.y_train_resampled == 1)} | Clase 0: {sum(self.y_train_resampled == 0)}")

    def definir_pipeline_preprocesamiento(self):
        self.pipeline_preprocesamiento = ImbPipeline(steps=[
            ('preprocesamiento', self.columnas_transformer),
            ('sub_sobre_muestreo', self.metodo_uo),
            #('model', modelo)
        ])

    def ejecutar_pipeline_preprocesamiento(self):
        print("Ejecutando pipeline preprocesamiento...Alex")
        print( type(self.columnas_transformer.fit_transform(self.X_train)))

    def guardar_artefactos(self):
        print("\n💾 Guardando artefactos...")
        
        # Guardar scaler en models/
        scaler_path = os.path.join(self.output_dir_models, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"  ✅ Scaler: {scaler_path}")
        
        # Guardar columnas OHE en references/
        ohe_cols_path = os.path.join(self.output_dir_references, "ohe_columns.json")
        with open(ohe_cols_path, "w", encoding="utf-8") as f:
            json.dump(self.ohe_cols, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Columnas OHE: {ohe_cols_path}")
        
        # Guardar columnas numéricas en references/
        num_cols_path = os.path.join(self.output_dir_references, "numeric_columns.json")
        with open(num_cols_path, "w", encoding="utf-8") as f:
            json.dump(self.num_cols, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Columnas numéricas: {num_cols_path}")
        
        # Guardar orden de features en references/
        feature_order_path = os.path.join(self.output_dir_references, "feature_order.json")
        with open(feature_order_path, "w", encoding="utf-8") as f:
            json.dump(self.X_train_final.columns.tolist(), f, indent=2, ensure_ascii=False)
        print(f"  ✅ Orden de features: {feature_order_path}")
    
    
    def guardar_datasets(self):
        print("\n💾 Guardando datasets procesados...")
        
        # Guardar datasets finales
        self.X_train_final.to_csv(
            os.path.join(self.output_dir_processed, "X_train.csv"), index=False
        )
        self.X_test_final.to_csv(
            os.path.join(self.output_dir_processed, "X_test.csv"), index=False
        )
        self.X_train_resampled.to_csv(
            os.path.join(self.output_dir_processed, "X_train_resampled.csv"), index=False
        )
        
        # Guardar targets si existen
        if self.target_col is not None:
            self.y_train.to_csv(
                os.path.join(self.output_dir_processed, "y_train.csv"), index=False
            )
            self.y_test.to_csv(
                os.path.join(self.output_dir_processed, "y_test.csv"), index=False
            )
            self.y_train_resampled.to_csv(
                os.path.join(self.output_dir_processed, "y_train_resampled.csv"), index=False
            )
        
        print(f"  ✅ Datasets guardados en: {self.output_dir_processed}")
    
    
    def ejecutar_preprocesamiento_completo(self):
        print("=" * 60)
        print("INICIANDO PREPROCESAMIENTO DE DATOS")
        print("=" * 60)
        
        # 1. Carga y clasificación
        self.cargar_datos_limpios()
        self.clasificar_variables()
        
        # 2. Colapsar categorías raras
        self.colapsar_categorias_raras(prop_min=0.05)
        
        # 3. Generar visualizaciones
        self.generar_visualizaciones()
        
        # 4. Preparación para modelado
        self.preparar_variables_modelado()
        self.dividir_train_test(test_size=0.20, random_state=42)
        
        # 5. Transformaciones
        self.aplicar_one_hot_encoding()
        self.escalar_variables_numericas()
        self.ensamblar_datasets_finales()
        self.inicializar_transformer()
        self.entrenar_transformer()
        #self.definir_oversampling()
        self.definir_pipeline_preprocesamiento()
        self.ejecutar_pipeline_preprocesamiento()

        # 6. Oversampling
        self.definir_oversampling(random_state=42)
        
        # 7. Guardar resultados
        self.guardar_artefactos()
        self.guardar_datasets()
        
        print("=" * 60)
        print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
    
    
    # =========================================================
    # MÉTODOS AUXILIARES (PRIVADOS)
    # =========================================================
    
    def _colapsar_categorias(self, series, prop_min=0.05, otra_cat=None):
        # Calcular frecuencias relativas
        share = series.value_counts(normalize=True)
        
        # Identificar categorías a mantener
        keep = share[share >= prop_min].index
        
        # Si no se especifica categoría de reemplazo, usar moda
        if otra_cat is None:
            otra_cat = series.mode(dropna=True).iloc[0]
        
        # Reemplazar categorías raras
        return series.where(series.isin(keep), otra_cat)
    
    
    def _graficar_variable_objetivo(self):
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Gráfica de barras
        self.df[self.target_col].value_counts().sort_index().plot(
            kind='bar', ax=ax, title=self.target_col
        )
        
        # Configurar título y guardar
        fig.suptitle("Distribución de la variable objetivo", y=1.02, fontsize=16)
        fig.savefig(
            os.path.join(self.output_dir_figures, 'hist_var_obj.png'),
            bbox_inches='tight'
        )
        plt.close(fig)
    
    
    def _graficar_categorias_colapsadas(self):
        # Crear subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        # Graficar cada variable ordinal
        for i, col in enumerate(self.ordinal_small):
            self.df[col].value_counts().sort_index().plot(
                kind='bar', ax=axes[i], title=col
            )
        
        # Ocultar ejes vacíos
        for i in range(len(self.ordinal_small), len(axes)):
            axes[i].axis('off')
        
        # Configurar título y guardar
        plt.tight_layout()
        fig.suptitle("Variables categóricas con categorías colapsadas", y=1.02, fontsize=16)
        fig.savefig(
            os.path.join(self.output_dir_figures, 'categorias_colapsadas.png'),
            bbox_inches='tight'
        )
        plt.close(fig)
    
    
    def _graficar_polizas(self):
        # Calcular dimensiones del grid
        n_cols = len(self.A_cols)
        n_rows = (n_cols + 2) // 3
        
        # Crear subplots
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
        axes = axes.flatten()
        
        # Graficar cada variable A*
        for i, col in enumerate(self.A_cols):
            self.df[col].plot(kind='hist', ax=axes[i], title=col)
        
        # Ocultar ejes vacíos
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        # Configurar título y guardar
        plt.tight_layout()
        fig.suptitle("Variables de pólizas (prefijo 'A')", y=1.01, fontsize=16)
        fig.savefig(
            os.path.join(self.output_dir_figures, 'hist_polizas.png'),
            bbox_inches='tight'
        )
        plt.close(fig)
    
    
    def _graficar_sociodemograficas(self):
        # Calcular dimensiones del grid
        n_cols = len(self.sociodem_cols)
        n_rows = (n_cols + 4) // 5
        
        # Crear subplots
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, n_rows * 4))
        axes = axes.flatten()
        
        # Graficar cada variable sociodemográfica
        for i, col in enumerate(self.sociodem_cols):
            self.df[col].plot(kind='hist', ax=axes[i], title=col)
        
        # Ocultar ejes vacíos
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        # Configurar título y guardar
        plt.tight_layout()
        fig.suptitle("Variables Sociodemográficas", y=1.01, fontsize=16)
        fig.savefig(
            os.path.join(self.output_dir_figures, 'hist_sociodem.png'),
            bbox_inches='tight'
        )
        plt.close(fig)
    
    
    def _one_hot_fit_transform(self, X_df, cat_cols):
        # Filtrar solo columnas que existen en X_df
        cols_in = [c for c in cat_cols if c in X_df.columns]
        
        # Si no hay columnas, devolver DataFrame vacío
        if not cols_in:
            return pd.DataFrame(index=X_df.index), []
        
        # Crear y ajustar OneHotEncoder
        ohe = OneHotEncoder(
            handle_unknown='ignore',
            drop='first',
            sparse_output=False,
            dtype=np.int8
        )
        
        # Transformar datos
        X_cat = pd.DataFrame(
            ohe.fit_transform(X_df[cols_in]),
            columns=ohe.get_feature_names_out(cols_in),
            index=X_df.index
        )
        
        return X_cat, ohe
    
    
    def _one_hot_transform_with_cols(self, X_df, cat_cols, ohe):
        # Filtrar solo columnas que existen en X_df
        cols_in = [c for c in cat_cols if c in X_df.columns]
        
        # Si no hay columnas, devolver DataFrame con columnas esperadas lleno de ceros
        if not cols_in:
            return pd.DataFrame(
                index=X_df.index,
                columns=ohe.get_feature_names_out(cols_in)
            ).fillna(0).astype("int8")
        
        # Transformar datos
        X_cat = pd.DataFrame(
            ohe.transform(X_df[cols_in]),
            columns=ohe.get_feature_names_out(cols_in),
            index=X_df.index
        )
        
        return X_cat


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================
def main():
    # Definir rutas según estructura de Cookiecutter
    SRC_PATH = "data/interim/insurance_clean.csv"
    OUTPUT_DIR_PROCESSED = "data/processed"
    OUTPUT_DIR_REPORTS = "reports"
    OUTPUT_DIR_FIGURES = "reports/figures"
    OUTPUT_DIR_MODELS = "models"
    OUTPUT_DIR_REFERENCES = "references"
    
    # Crear instancia del preprocesador
    preprocessor = DataPreprocessor(
        src_path=SRC_PATH,
        output_dir_processed=OUTPUT_DIR_PROCESSED,
        output_dir_reports=OUTPUT_DIR_REPORTS,
        output_dir_figures=OUTPUT_DIR_FIGURES,
        output_dir_models=OUTPUT_DIR_MODELS,
        output_dir_references=OUTPUT_DIR_REFERENCES
    )
    
    # Ejecutar preprocesamiento completo
    preprocessor.ejecutar_preprocesamiento_completo()
    
if __name__ == "__main__":
    main()
