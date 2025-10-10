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

SRC_PATH = "data"
OUT_DIR_DATA = "data"
OUT_DIR_RES = "results"
OUT_DIR_GRAPHS = "graphs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_RES, exist_ok=True)
os.makedirs(OUT_DIR_GRAPHS, exist_ok=True)


data_x_train = pd.read_csv(SRC_PATH+'/X_train.csv', sep=",")
data_y_train = pd.read_csv(SRC_PATH+'/y_train.csv', sep=",")
data_x_test = pd.read_csv(SRC_PATH+'/x_test.csv', sep=",")
data_y_test = pd.read_csv(SRC_PATH+'/y_test.csv', sep=",")



print('Clase positiva (tiene poliza): %.1f%%' % (100 * (data_y_train.sum() / data_y_train.shape[0])))
print('Clase negativa (no tiene poliza ): %.1f%%' % (100 * (1 - data_y_train.sum() / data_y_train.shape[0])))



x_train=data_x_train.to_numpy()
y_train=data_y_train.to_numpy()

x_test=data_x_test.to_numpy()
y_test=data_y_test.to_numpy()

modelo = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=1)
modelo.fit(x_train, np.ravel(y_train))

acc = accuracy_score(np.ravel(y_test), modelo.predict(x_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")

