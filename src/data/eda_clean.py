# =========================================================
# LIMPIEZA Y NORMALIZACIÓN DE DATASET
# Dataset: Insurance Company Benchmark (COIL 2000)
# =========================================================
# Objetivo del script:
# - Cargar el CSV modificado.
# - Tratar CUALQUIER cadena de texto: volverla NaN.
# - Clasificar variables (binarias, sociodemográficas, P*, A*, adicional).
# - Variables binarias: sustituir con moda si hay valores != 0 y 1.
# - Variables no binarias: sustituir vacíos con mediana.
# - Detectar y corregir outliers con IQR.
# - Eliminar duplicados.
# - Guardar dataset limpio y métricas básicas para el reporte.
# =========================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import numpy as np
import pandas as pd
import tempfile
import pickle


# =========================================================
# CLASE PRINCIPAL: DataCleaner
# =========================================================
class DataCleaner:
    
    def __init__(self, src_path, output_dir_interim, output_dir_reports, dict_columnas_path):
        # Atributos de rutas
        self.src_path = src_path
        self.output_dir_interim = output_dir_interim
        self.output_dir_reports = output_dir_reports
        self.dict_columnas_path = dict_columnas_path
        
        # Crear directorios si no existen
        os.makedirs(self.output_dir_interim, exist_ok=True)
        os.makedirs(self.output_dir_reports, exist_ok=True)
        
        # Atributos para almacenar datos
        self.df_raw = None
        self.df = None
        self.dict_columnas = None
        
        # Atributos para clasificación de variables
        self.P_cols = []
        self.A_cols = []
        self.ordinal_small = []
        self.target_col = None
        self.binary_cols = []
        self.sociodem_cols = []
        self.M_rest = []
        
        # Diccionario de dominios para variables ordinales
        self.ordinal_domains = {
            "MOSTYPE":  (1, 41),
            "MOSHOOFD": (1, 10),
            "MGEMLEEF": (1, 6),
            "MGODRK":   (0, 9),
            "PWAPART":  (0, 9)
        }
        
        # Métricas de limpieza
        self.metrics = {}
        self.total_outliers_removed = 0
    
    
    def cargar_datos(self):
        # Cargar CSV sin encabezados
        self.df_raw = pd.read_csv(self.src_path, header=None, low_memory=False)
        print(f"✅ Cargado: {self.src_path} | Shape: {self.df_raw.shape}")
    
    
    def cargar_diccionario_columnas(self):
        # Leer archivo pickle con nombres de columnas
        with open(self.dict_columnas_path, "rb") as f:
            self.dict_columnas = pickle.load(f)
        
        print(f"✅ Cargado: {self.dict_columnas_path}")
    
    
    def renombrar_columnas(self):
        # Copiar DataFrame y eliminar primera fila (encabezados originales)
        self.df = self.df_raw.copy()
        self.df = self.df.iloc[1:].reset_index(drop=True)
        
        # Renombrar columnas usando diccionario
        self.df.rename(columns=self.dict_columnas, inplace=True)
        
        # Convertir nombres a mayúsculas y eliminar espacios
        self.df.columns = [str(c).strip().upper() for c in self.df.columns]
    
    
    def convertir_a_numerico(self):
        # Contar valores nulos antes de la conversión
        nulls_before = int(self.df.isna().sum().sum())
        
        # Aplicar función de conversión a todas las columnas
        self.df = self.df.apply(lambda col: col.map(self._to_number_or_nan))
        
        # Asegurar que todas las columnas sean numéricas
        for c in self.df.columns:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        
        # Contar valores nulos después de la conversión
        nulls_after = int(self.df.isna().sum().sum())
        
        print(f"📊 Valores nulos antes: {nulls_before} | después: {nulls_after}")
        
        # Guardar métricas iniciales
        self.metrics["dataset_original"] = {
            "filas": int(self.df_raw.shape[0]),
            "columnas": int(self.df_raw.shape[1]),
            "valores_nulos": nulls_before
        }
    
    
    def clasificar_variables(self):
        # Detectar columnas P* (contribuciones, dominio 0-9)
        self.P_cols = [c for c in self.df.columns if isinstance(c, str) and c.startswith("P")]
        
        # Detectar columnas A* (número de pólizas, dominio 0-12)
        self.A_cols = [c for c in self.df.columns if isinstance(c, str) and c.startswith("A")]
        
        # Detectar ordinales pequeñas del diccionario
        self.ordinal_small = [c for c in self.ordinal_domains.keys() if c in self.df.columns]
        
        # Detectar variable objetivo (binaria real)
        self.target_col = "CARAVAN" if "CARAVAN" in self.df.columns else None
        self.binary_cols = [self.target_col] if self.target_col else []
        
        # Detectar resto de variables M* (tasas/proporciones)
        self.M_rest = [c for c in self.df.columns
                       if isinstance(c, str) and c.startswith("M")
                       and c not in self.ordinal_small and c != self.target_col]
        
        # Sociodemográficas = resto (excluyendo todas las anteriores)
        excluir = set(self.P_cols) | set(self.A_cols) | set(self.ordinal_small) | set(self.binary_cols)
        self.sociodem_cols = [c for c in self.df.columns if c not in excluir]
        
        print(f"📋 Clasificación de variables:")
        print(f"   - CARAVAN presente: {bool(self.target_col)}")
        print(f"   - P* (contribuciones): {len(self.P_cols)}")
        print(f"   - A* (pólizas): {len(self.A_cols)}")
        print(f"   - Ordinales: {len(self.ordinal_small)}")
        print(f"   - M* resto: {len(self.M_rest)}")
        print(f"   - Sociodemográficas: {len(self.sociodem_cols)}")
    
    
    def limpiar_variable_objetivo(self):
        # Si existe la variable objetivo, imputar con moda y asegurar valores binarios
        if self.target_col:
            # Calcular moda de valores válidos (0 o 1)
            moda = self._safe_mode(self.df[self.target_col].dropna().clip(0, 1))
            moda = int(moda) if pd.notna(moda) else 0
            
            # Imputar valores faltantes y asegurar rango binario
            self.df[self.target_col] = self.df[self.target_col].fillna(moda).clip(0, 1).round().astype("int8")
    
    
    def limpiar_sociodemograficas(self):
        total_outliers = 0
        
        # Para cada variable sociodemográfica
        for c in self.sociodem_cols:
            # Imputar valores faltantes con mediana
            if self.df[c].isna().any():
                self.df[c] = self.df[c].fillna(self.df[c].median(skipna=True))
            
            # Winsorizar outliers usando IQR (sin límites duros)
            self.df[c], rec = self._winsorize_iqr(self.df[c], iqr_k=1.5, hard_clip=None)
            total_outliers += rec
        
        print(f"📊 Outliers corregidos en sociodemográficas: {total_outliers}")
        return total_outliers
    
    
    def limpiar_contribuciones(self):
        total_outliers = 0
        
        # Para cada variable de contribución
        for c in self.P_cols:
            # Imputar valores faltantes con mediana
            if self.df[c].isna().any():
                self.df[c] = self.df[c].fillna(self.df[c].median(skipna=True))
            
            # Winsorizar con límite duro [0, 9]
            self.df[c], rec = self._winsorize_iqr(self.df[c], iqr_k=1.5, hard_clip=(0, 9))
            total_outliers += rec
            
            # Convertir a entero en rango válido
            self.df[c] = self.df[c].round().clip(0, 9).astype("int8")
        
        print(f"📊 Outliers corregidos en P* (contribuciones): {total_outliers}")
        return total_outliers
    
    
    def limpiar_polizas(self):
        total_outliers = 0
        
        # Para cada variable de pólizas
        for c in self.A_cols:
            # Imputar valores faltantes con mediana
            if self.df[c].isna().any():
                self.df[c] = self.df[c].fillna(self.df[c].median(skipna=True))
            
            # Winsorizar con límite duro [0, 12]
            self.df[c], rec = self._winsorize_iqr(self.df[c], iqr_k=1.5, hard_clip=(0, 12))
            total_outliers += rec
            
            # Convertir a entero en rango válido
            self.df[c] = self.df[c].round().clip(0, 12).astype("int8")
        
        print(f"📊 Outliers corregidos en A* (pólizas): {total_outliers}")
        return total_outliers
    
    
    def limpiar_ordinales(self):
        total_outliers = 0
        
        # Para cada variable ordinal
        for c in self.ordinal_small:
            # Obtener límites del dominio
            lo, hi = self.ordinal_domains[c]
            
            # Imputar valores faltantes con moda
            if self.df[c].isna().any():
                modo = self._safe_mode(self.df[c])
                self.df[c] = self.df[c].fillna(modo)
            
            # Aplicar clip a dominio (sin winsorización IQR)
            before = self.df[c].copy()
            self.df[c] = self.df[c].clip(lo, hi)
            
            # Contar valores corregidos
            rec = int((~self.df[c].eq(before)).sum())
            total_outliers += rec
            
            # Convertir a entero
            self.df[c] = self.df[c].round().astype("int8")
        
        print(f"📊 Outliers corregidos en ordinales: {total_outliers}")
        return total_outliers
    
    
    def limpiar_proporciones(self):
        total_outliers = 0
        
        # Para cada variable M* restante
        for c in self.M_rest:
            # Imputar valores faltantes con mediana
            if self.df[c].isna().any():
                self.df[c] = self.df[c].fillna(self.df[c].median(skipna=True))
            
            # Determinar límite duro según el rango de valores
            vmax = self.df[c].max(skipna=True)
            hard = (0, 9) if (pd.notna(vmax) and vmax <= 9) else (0, 100)
            
            # Winsorizar con límite duro
            self.df[c], rec = self._winsorize_iqr(self.df[c], iqr_k=1.5, hard_clip=hard)
            total_outliers += rec
            
            # Mantener como float
            self.df[c] = self.df[c].astype("float32")
        
        print(f"📊 Outliers corregidos en M* resto: {total_outliers}")
        return total_outliers
    
    
    def redondear_valores(self):
        # Aplicar floor si la diferencia con el piso es < 0.5, si no ceiling
        self.df = self.df.apply(
            lambda col: np.where(
                np.issubdtype(col.dtype, np.number),
                np.where(col - np.floor(col) < 0.5, np.floor(col), np.ceil(col)),
                col
            )
        )
    
    
    def generar_metricas(self):
        # Calcular métricas del dataset limpio
        self.metrics["dataset_limpio"] = {
            "filas": int(self.df.shape[0]),
            "columnas": int(self.df.shape[1]),
            "valores_nulos": int(self.df.isna().sum().sum())
        }
        
        # Registrar cambios realizados
        self.metrics["cambios_realizados"] = {
            "outliers_corregidos": int(self.total_outliers_removed)
        }
        
        # Registrar clasificación de familias
        self.metrics["familias"] = {
            "P_cols": len(self.P_cols),
            "A_cols": len(self.A_cols),
            "ordinales": len(self.ordinal_small),
            "sociodemograficas": len(self.sociodem_cols),
            "M_rest": len(self.M_rest),
            "tiene_caravan": bool(self.target_col)
        }
        
        print("📊 Métricas de limpieza:", self.metrics["cambios_realizados"])
    
    
    def guardar_resultados(self):
        # Definir rutas de salida
        out_data_path = os.path.join(self.output_dir_interim, "insurance_clean.csv")
        out_metrics_path = os.path.join(self.output_dir_reports, "eda_metrics.json")
        
        # Guardar CSV limpio de forma segura
        self._safe_to_csv(self.df, out_data_path)
        
        # Guardar métricas en JSON
        with open(out_metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Guardado dataset limpio: {out_data_path}")
        print(f"✅ Guardado métricas EDA:   {out_metrics_path}")
    
    
    def ejecutar_limpieza_completa(self):
        print("=" * 60)
        print("INICIANDO LIMPIEZA Y NORMALIZACIÓN DE DATOS")
        print("=" * 60)
        
        # 1. Carga de datos
        self.cargar_datos()
        self.cargar_diccionario_columnas()
        self.renombrar_columnas()
        
        # 2. Conversión a numérico
        self.convertir_a_numerico()
        
        # 3. Clasificación de variables
        self.clasificar_variables()
        
        # 4. Limpieza por tipo de variable
        self.limpiar_variable_objetivo()
        outliers_sociodem = self.limpiar_sociodemograficas()
        outliers_P = self.limpiar_contribuciones()
        outliers_A = self.limpiar_polizas()
        outliers_ord = self.limpiar_ordinales()
        outliers_M = self.limpiar_proporciones()
        
        # Sumar total de outliers removidos
        self.total_outliers_removed = (
            outliers_sociodem + outliers_P + outliers_A + outliers_ord + outliers_M
        )
        
        # 5. Redondeo final
        self.redondear_valores()
        
        # 6. Generar métricas y guardar
        self.generar_metricas()
        self.guardar_resultados()
        
        print("=" * 60)
        print("LIMPIEZA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
    
    
    # =========================================================
    # MÉTODOS AUXILIARES (PRIVADOS)
    # =========================================================
    
    def _to_number_or_nan(self, x):
        # Si ya es numérico, devolverlo tal cual
        if isinstance(x, (int, float, np.number)):
            return x
        
        # Si es NaN, mantenerlo como NaN
        if pd.isna(x):
            return np.nan
        
        # Si es string, intentar convertir a número
        if isinstance(x, str):
            s = x.strip().lower()
            # Valores que representan ausencia de datos
            if s in {"", "na", "n/a", "none", "null", "nan", "?", "-", "--"}:
                return np.nan
            # Intentar conversión numérica
            return pd.to_numeric(s, errors="coerce")
        
        # Cualquier otro tipo se convierte en NaN
        return np.nan
    
    
    def _safe_mode(self, series):
        # Calcular moda eliminando valores nulos
        m = series.mode(dropna=True)
        # Devolver primer valor de la moda o NaN si está vacía
        return m.iloc[0] if not m.empty else np.nan
    
    
    def _winsorize_iqr(self, series, iqr_k=1.5, hard_clip=None, zero_threshold=0.6, min_nonzero=20):
        # Convertir a numérico
        s = pd.to_numeric(series, errors="coerce")
        
        # Si la serie está vacía, no hacer nada
        if s.dropna().empty:
            return s, 0
        
        # Determinar base para cálculo de IQR (maneja columnas con muchos ceros)
        s_notnull = s.dropna()
        share_zero = (s_notnull == 0).mean() if len(s_notnull) else 0.0
        s_nz = s_notnull[s_notnull != 0]
        
        # Usar valores no-cero si hay muchos ceros y suficientes valores no-cero
        base = s_nz if (share_zero >= zero_threshold and len(s_nz) >= min_nonzero) else s_notnull
        
        # Si base está vacía, solo aplicar hard_clip si existe
        if base.empty:
            s_w = s.copy()
            if hard_clip is not None:
                lo, hi = hard_clip
                s_w = s_w.clip(lower=lo, upper=hi)
            return s_w, 0
        
        # Calcular cuartiles e IQR
        q1, q3 = base.quantile(0.25), base.quantile(0.75)
        iqr = q3 - q1
        
        # Si IQR no es válido, solo aplicar hard_clip si existe
        if not np.isfinite(iqr) or iqr <= 0:
            s_w = s.copy()
            if hard_clip is not None:
                lo, hi = hard_clip
                s_w = s_w.clip(lower=lo, upper=hi)
            n_recortes = int((~s_w.eq(s)).sum())
            return s_w, n_recortes
        
        # Calcular límites con IQR
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        s_w = s.clip(lower=lo, upper=hi)
        
        # Aplicar hard_clip adicional si se especifica
        if hard_clip is not None:
            hlo, hhi = hard_clip
            s_w = s_w.clip(lower=hlo, upper=hhi)
        
        # Contar número de valores recortados
        n_recortes = int((~s_w.eq(s)).sum())
        return s_w, n_recortes
    
    
    def _safe_to_csv(self, df, path, retries=5, wait_sec=1.0):
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Crear archivo temporal en el mismo directorio
        fd, tmp = tempfile.mkstemp(prefix="tmp_", suffix=".csv", dir=os.path.dirname(path) or ".")
        os.close(fd)
        
        # Guardar a archivo temporal
        df.to_csv(tmp, index=False, encoding="utf-8")
        
        # Reemplazar archivo destino con temporal (operación atómica)
        os.replace(tmp, path)


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================
def main():
    # Definir rutas según estructura de Cookiecutter
    SRC_PATH = "data/raw/insurance_company_modified.csv"
    DICT_COLUMNAS_PATH = "references/nombres_columnas.pkl"
    OUTPUT_DIR_INTERIM = "data/interim"
    OUTPUT_DIR_REPORTS = "reports"
    
    # Crear instancia del limpiador
    cleaner = DataCleaner(
        src_path=SRC_PATH,
        output_dir_interim=OUTPUT_DIR_INTERIM,
        output_dir_reports=OUTPUT_DIR_REPORTS,
        dict_columnas_path=DICT_COLUMNAS_PATH
    )
    
    # Ejecutar limpieza completa
    cleaner.ejecutar_limpieza_completa()

if __name__ == "__main__":
    main()