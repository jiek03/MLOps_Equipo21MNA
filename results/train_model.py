import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score ,confusion_matrix,ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

SRC_PATH = "data"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
OUT_DIR_GRAPHS = "graphs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)
os.makedirs(OUT_DIR_GRAPHS, exist_ok=True)

# -----------------------------------------------------------------
# 1) Leer datos de entrenamiento y test
# -----------------------------------------------------------------

data_x_train = pd.read_csv(SRC_PATH+'/X_train.csv', sep=",")
data_y_train = pd.read_csv(SRC_PATH+'/y_train.csv', sep=",")
data_x_test = pd.read_csv(SRC_PATH+'/X_test.csv', sep=",")
data_y_test = pd.read_csv(SRC_PATH+'/y_test.csv', sep=",")



print('Clase positiva (tiene poliza): %.1f%%' % (100 * (data_y_train.sum().iloc[0] / data_y_train.shape[0])))
print('Clase negativa (no tiene poliza ): %.1f%%' % (100 * (1 - data_y_train.sum().iloc[0] / data_y_train.shape[0])))

# -----------------------------------------------------------------
# 2) Función para entrenar y evaluar modelos
# -----------------------------------------------------------------

def evalua_modelos(modelos, X_train, y_train, X_test, y_test, average='binary'):
    """
    Función que entrena y evalúa múltiples modelos utilizando métricas comunes.

    Parametros
    ----------
    modelos : dict
       diccionario con nombre del modelo como clave y el estimador como valor.
        Ejemplo: {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier()}
    X_train, y_train, X_test, y_test.
    average : str, opcional (default='binary')
        metodo de promedio para problemas multiclase ('binary', 'macro', 'micro', 'weighted').

    Returns
    -------
    resultados_df_train: pd.DataFrame con las metricas del modelo sobre el conjunto de entrenamiento.
    resultados_df_test: pd.DataFrame con las metricas del modelo sobre el conjunto de prueba.
    modelos_entrenados: dict con los modelos entrenados.
    """

    resultados_train = []
    resultados_test = []
    modelos_entrenados = {}


    for name, model in modelos.items():
        # Ajusta el modelo
        model.fit(X_train, y_train)
        modelos_entrenados[name] = model

        # Realiza predicciones
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        # En caso de que el modelo no soporte predict_proba (e.g., SVM sin probabilidad)
        try:
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_train = roc_auc_score(y_train, y_prob_train)
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        # calcula métricas sobre entrenamiento y prueba
        metrics_train = {
            'model': name,
            'accuracy': round(accuracy_score(y_train, y_pred_train)*100 , 2),
            'precision': round(precision_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'recall': round(recall_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'f1': round(f1_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'roc_auc': round(auc_train*100,2) if not np.isnan(auc_train) else np.nan,
        }

        metrics_test = {
            'model': name,
            'accuracy': round(accuracy_score(y_test, y_pred)*100,2),
            'precision': round(precision_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'recall': round(recall_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'f1': round(f1_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'roc_auc': round(auc*100,2) if not np.isnan(auc_train) else np.nan,
        }

        resultados_train.append(metrics_train)
        resultados_test.append(metrics_test)

    # Creamos el DataFrame ordenamos por el nombre del modelo 
    resultados_df_train = pd.DataFrame(resultados_train).sort_values(by='model', ascending=False).reset_index(drop=True)
    resultados_df_test = pd.DataFrame(resultados_test).sort_values(by='model', ascending=False).reset_index(drop=True)
    return resultados_df_train,resultados_df_test,modelos_entrenados

# -----------------------------------------------------------------
# 3) Entrenar modelo de regresión logística
# -----------------------------------------------------------------

x_train=data_x_train.to_numpy()
y_train=data_y_train.to_numpy()

x_test=data_x_test.to_numpy()
y_test=data_y_test.to_numpy()

# -----------------------------------------------------------------
# 4) Entrenar y evaluar modelos 
# ----------------------------------------------------------------

modelos = {
    'Logistic Regression': LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'SVM': SVC(probability=True)
    }
resultados_train, resultados_test, modelos_entrenados = evalua_modelos(modelos, x_train, y_train, x_test, y_test, average='binary')

print("Resultados en entrenamiento:")
print(resultados_train)
print("\nResultados en prueba:")
print(resultados_test)  

resultados_train.to_json(os.path.join(OUT_DIR_RES, 'resultados_train.json'), orient='columns',indent=4)
resultados_test.to_json(os.path.join(OUT_DIR_RES, 'resultados_test.json'),orient='columns',indent=4)

#modelo = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=1)
#modelo.fit(x_train, np.ravel(y_train))

#acc = accuracy_score(np.ravel(y_test), modelo.predict(x_test)) * 100
#print(f"Logistic Regression model accuracy: {acc:.2f}%")

# -----------------------------------------------------------------
# 5) Matriz de confusión
# -----------------------------------------------------------------
OUT_DIR_GRAPHS = "graphs"
os.makedirs(OUT_DIR_GRAPHS, exist_ok=True)

for modelo in modelos.keys():
    y_true = np.ravel(y_test)
    y_pred = modelos_entrenados[modelo].predict(x_test)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No tiene póliza', 'Tiene póliza'])

    # Matriz de confusión - Conteos
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Matriz de confusión (conteos) - {modelo}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_GRAPHS, f"cm_counts_{modelo}.png"), dpi=200)
    plt.close(fig)
