"""
Módulo para limpieza y preparación inicial de datos

Este módulo contiene la clase DataCleaner que se encarga de:
    - Cargar datos raw desde data/raw/
    - Convertir strings a valores numéricos
    - Clasificar variables por tipo
    - Imputar valores faltantes
    - Detectar y corregir outliers
    - Guardar datos limpios en data/interim/
"""

from .eda_clean import DataCleaner

__all__ = ['DataCleaner']