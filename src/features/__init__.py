"""
Módulo para preprocesamiento y generación de features

Este módulo contiene la clase DataPreprocessor que se encarga de:
    - Cargar datos limpios desde data/interim/
    - Colapsar categorías poco frecuentes
    - Generar visualizaciones exploratorias
    - Aplicar One-Hot Encoding
    - Escalar variables numéricas
    - Aplicar SMOTE para balancear clases
"""

from .preprocessing import DataPreprocessor

__all__ = ['DataPreprocessor']