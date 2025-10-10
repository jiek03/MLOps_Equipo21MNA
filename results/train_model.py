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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
# 2) Entrenar modelo de regresión logística
# -----------------------------------------------------------------

x_train=data_x_train.to_numpy()
y_train=data_y_train.to_numpy()

x_test=data_x_test.to_numpy()
y_test=data_y_test.to_numpy()

# -----------------------------------------------------------------
# 3) Evaluar modelo
# -----------------------------------------------------------------

modelo = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=1)
modelo.fit(x_train, np.ravel(y_train))

acc = accuracy_score(np.ravel(y_test), modelo.predict(x_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")

# -----------------------------------------------------------------
# 4) Matriz de confusión
# -----------------------------------------------------------------
OUT_DIR_GRAPHS = "graphs"
os.makedirs(OUT_DIR_GRAPHS, exist_ok=True)

y_true = np.ravel(y_test)

y_pred = modelo.predict(x_test)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No tiene póliza', 'Tiene póliza'])

# Matriz de confusión - Conteos
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Matriz de confusión (conteos)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_GRAPHS, "cm_counts.png"), dpi=200)
plt.close(fig)
