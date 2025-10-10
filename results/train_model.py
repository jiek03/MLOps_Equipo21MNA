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
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

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
data_x_test  = pd.read_csv(SRC_PATH+'/X_test.csv',  sep=",")
data_y_test  = pd.read_csv(SRC_PATH+'/y_test.csv',  sep=",")

print('Clase positiva (tiene poliza): %.1f%%' % (100 * (data_y_train.sum().iloc[0] / data_y_train.shape[0])))
print('Clase negativa (no tiene poliza ): %.1f%%' % (100 * (1 - data_y_train.sum().iloc[0] / data_y_train.shape[0])))

# -----------------------------------------------------------------
# 2) Función para entrenar y evaluar modelos
# -----------------------------------------------------------------
def evalua_modelos(modelos, X_train, y_train, X_test, y_test, average='binary'):
    resultados_train = []
    resultados_test = []
    modelos_entrenados = {}

    for name, model in modelos.items():
        model.fit(X_train, y_train)
        modelos_entrenados[name] = model

        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        try:
            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_prob = model.predict_proba(X_test)[:, 1]
            auc_train = roc_auc_score(y_train, y_prob_train)
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan  # (dejamos como en tu código original)
            # auc_train no se usa si cae aquí en tus modelos actuales (todos tienen predict_proba)

        metrics_train = {
            'model': name,
            'accuracy': round(accuracy_score(y_train, y_pred_train)*100 , 2),
            'precision': round(precision_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'recall': round(recall_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'f1': round(f1_score(y_train, y_pred_train, average=average, zero_division=0)*100,2),
            'roc_auc': round(auc_train*100,2) if 'auc_train' in locals() else np.nan,
        }

        metrics_test = {
            'model': name,
            'accuracy': round(accuracy_score(y_test, y_pred)*100,2),
            'precision': round(precision_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'recall': round(recall_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'f1': round(f1_score(y_test, y_pred, average=average, zero_division=0)*100,2),
            'roc_auc': round(auc*100,2) if not np.isnan(auc) else np.nan,
        }

        resultados_train.append(metrics_train)
        resultados_test.append(metrics_test)

    resultados_df_train = pd.DataFrame(resultados_train).sort_values(by='model', ascending=False).reset_index(drop=True)
    resultados_df_test  = pd.DataFrame(resultados_test ).sort_values(by='model', ascending=False).reset_index(drop=True)
    return resultados_df_train, resultados_df_test, modelos_entrenados

# -----------------------------------------------------------------
# 3) Preparar matrices X/y
# -----------------------------------------------------------------
x_train = data_x_train.values
y_train = data_y_train.values.ravel()

x_test  = data_x_test.values
y_test  = data_y_test.values.ravel()

# -----------------------------------------------------------------
# 4) Baseline: entrenar y evaluar modelos
# -----------------------------------------------------------------
modelos = {
    'Logistic Regression': LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'SVM': SVC(probability=True)
}
resultados_train, resultados_test, modelos_entrenados = evalua_modelos(
    modelos, x_train, y_train, x_test, y_test, average='binary'
)

print("Resultados en entrenamiento:")
print(resultados_train)
print("\nResultados en prueba:")
print(resultados_test)

resultados_train.to_json(os.path.join(OUT_DIR_RES, 'resultados_train.json'), orient='columns', indent=4)
resultados_test.to_json(os.path.join(OUT_DIR_RES, 'resultados_test.json'),  orient='columns', indent=4)

# -----------------------------------------------------------------
# 4b) versiones con class_weight='balanced'
# -----------------------------------------------------------------
modelos_bal = {
    'LogReg (balanced)': LogisticRegression(
        penalty='l2', C=0.1, solver='lbfgs',
        max_iter=4000, class_weight='balanced',
        random_state=42
    ),
    'SVM (balanced)': SVC(probability=True, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
}
res_train_bal, res_test_bal, modelos_entrenados_bal = evalua_modelos(
    modelos_bal, x_train, y_train, x_test, y_test, average='binary'
)

print("\nResultados en prueba (class_weight='balanced'):")
print(res_test_bal)

# -----------------------------------------------------------------
# 5) Ajuste de umbral F1 para LogReg (balanced) y evaluación en TEST
# -----------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix

# 1) Split de validación desde TRAIN
X_tr, X_val, y_tr, y_val = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=42
)

# 2) Re-entrenar SOLO LogReg(balanced) con la misma config del bloque balanced
log_bal = LogisticRegression(
    penalty='l2', C=0.1, solver='lbfgs',
    max_iter=4000, class_weight='balanced',
    random_state=42
)
log_bal.fit(X_tr, y_tr)

# 3) Elegir umbral que maximiza F1 en VALID
probs_val = log_bal.predict_proba(X_val)[:, 1]
p, r, thr = precision_recall_curve(y_val, probs_val)
f1 = 2 * (p * r) / (p + r + 1e-12)
idx = f1.argmax()
best_thr = thr[max(idx - 1, 0)] if idx > 0 else 0.5
print(f"\n[Umbral F1] VALID -> thr={best_thr:.3f} | P={p[idx]:.2f} R={r[idx]:.2f} F1={f1[idx]:.2f}")

# 4) Evaluar en TEST con ese umbral + PR-AUC (métrica clave en desbalance)
probs_test = log_bal.predict_proba(x_test)[:, 1]
ap_test = average_precision_score(y_test, probs_test)
y_pred = (probs_test >= best_thr).astype(int)
print(f"[TEST] Average Precision (PR-AUC) = {ap_test:.3f}")
print("[TEST] classification_report (LogReg balanced con umbral F1)")
print(classification_report(y_test, y_pred, target_names=['No póliza','Sí póliza'], zero_division=0))
print("CM TEST [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, y_pred, labels=[0,1]))
