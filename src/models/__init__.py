"""
Módulo para entrenamiento y evaluación de modelos

Este módulo contiene la clase ModelTrainer que se encarga de:
    - Cargar datasets procesados
    - Entrenar múltiples modelos de clasificación
    - Evaluar modelos en train y test
    - Generar matrices de confusión
    - Ajustar umbral de decisión
"""

from .train_model import ModelTrainer

__all__ = ['ModelTrainer']