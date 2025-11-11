# MLOps_Equipo21MNA

## 🧩 Descripción del Dataset

El proyecto utiliza el conjunto de datos **Insurance Company Benchmark (COIL 2000)**, 
publicado originalmente por *Sentient Machine Research* y disponible en el 
[Repositorio de Aprendizaje Automático de la UCI](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)).

Este dataset contiene información de **5,822 clientes** y **86 variables**, que describen tanto
características sociodemográficas como la posesión de distintos productos de seguros en los Países Bajos.
El objetivo principal es predecir si un cliente posee una **póliza de seguro para casas rodantes**
(`CARAVAN` = 1).

## 📘 Diccionario de Datos (COIL 2000)

El conjunto de datos **Insurance Company Benchmark (COIL 2000)** contiene información de clientes neerlandeses, combinando datos sociodemográficos (derivados del código postal) y datos de uso de productos de seguros.

Cada registro representa a un cliente, con **86 variables** descritas a continuación.

---

### 🏠 Variables principales

## 📘 Diccionario de Datos (COIL 2000)

| Nº | Nombre | Descripción | Dominio |
|----|---------|--------------|----------|
| 1 | MOSTYPE | Subtipo de cliente | Ver L0 |
| 2 | MAANTHUI | Número de viviendas | 1 – 10 |
| 3 | MGEMOMV | Tamaño promedio del hogar | 1 – 6 |
| 4 | MGEMLEEF | Edad promedio | Ver L1 |
| 5 | MOSHOOFD | Tipo principal de cliente | Ver L2 |
| 6 | MGODRK | Católico romano | Ver L3 |
| 7 | MGODPR | Protestante | ... |
| 8 | MGODOV | Otra religión | ... |
| 9 | MGODGE | Sin religión | ... |
| 10 | MRELGE | Casado | ... |
| 11 | MRELSA | Vive en pareja | ... |
| 12 | MRELOV | Otro tipo de relación | ... |
| 13 | MFALLEEN | Solteros | ... |
| 14 | MFGEKIND | Hogar sin hijos | ... |
| 15 | MFWEKIND | Hogar con hijos | ... |
| 16 | MOPLHOOG | Educación de nivel alto | ... |
| 17 | MOPLMIDD | Educación de nivel medio | ... |
| 18 | MOPLLAAG | Educación de nivel bajo | ... |
| 19 | MBERHOOG | Estatus alto | ... |
| 20 | MBERZELF | Emprendedor | ... |
| 21 | MBERBOER | Agricultor | ... |
| 22 | MBERMIDD | Gerencia media | ... |
| 23 | MBERARBG | Trabajadores calificados | ... |
| 24 | MBERARBO | Trabajadores no calificados | ... |
| 25 | MSKA | Clase social A | ... |
| 26 | MSKB1 | Clase social B1 | ... |
| 27 | MSKB2 | Clase social B2 | ... |
| 28 | MSKC | Clase social C | ... |
| 29 | MSKD | Clase social D | ... |
| 30 | MHHUUR | Vivienda rentada | ... |
| 31 | MHKOOP | Vivienda propia | ... |
| 32 | MAUT1 | 1 automóvil | ... |
| 33 | MAUT2 | 2 automóviles | ... |
| 34 | MAUT0 | Sin automóvil | ... |
| 35 | MZFONDS | Servicio Nacional de Salud | ... |
| 36 | MZPART | Seguro médico privado | ... |
| 37 | MINKM30 | Ingreso < 30,000 | ... |
| 38 | MINK3045 | Ingreso 30,000 – 45,000 | ... |
| 39 | MINK4575 | Ingreso 45,000 – 75,000 | ... |
| 40 | MINK7512 | Ingreso 75,000 – 122,000 | ... |
| 41 | MINK123M | Ingreso > 123,000 | ... |
| 42 | MINKGEM | Ingreso promedio | ... |
| 43 | MKOOPKLA | Clase de poder adquisitivo | ... |
| 44 | PWAPART | Contribución a seguros privados de terceros | Ver L4 |
| 45 | PWABEDR | Contribución a seguros de terceros (empresas) | ... |
| 46 | PWALAND | Contribución a seguros agrícolas | ... |
| 47 | PPERSAUT | Contribución a seguros de automóvil | ... |
| 48 | PBESAUT | Contribución a seguros de camionetas | ... |
| 49 | PMOTSCO | Contribución a seguros de motocicletas/scooters | ... |
| 50 | PVRAAUT | Contribución a seguros de camiones | ... |
| 51 | PAANHANG | Contribución a seguros de remolque | ... |
| 52 | PTRACTOR | Contribución a seguros de tractor | ... |
| 53 | PWERKT | Contribución a seguros de maquinaria agrícola | ... |
| 54 | PBROM | Contribución a seguros de ciclomotores | ... |
| 55 | PLEVEN | Contribución a seguros de vida | ... |
| 56 | PPERSONG | Contribución a seguros de accidentes personales | ... |
| 57 | PGEZONG | Contribución a seguros familiares de accidentes | ... |
| 58 | PWAOREG | Contribución a seguros por discapacidad | ... |
| 59 | PBRAND | Contribución a seguros contra incendios | ... |
| 60 | PZEILPL | Contribución a seguros de tabla de surf | ... |
| 61 | PPLEZIER | Contribución a seguros de barco | ... |
| 62 | PFIETS | Contribución a seguros de bicicleta | ... |
| 63 | PINBOED | Contribución a seguros de propiedad | ... |
| 64 | PBYSTAND | Contribución a seguros de seguridad social | ... |
| 65 | AWAPART | Número de seguros privados de terceros | 1 – 12 |
| 66 | AWABEDR | Número de seguros de terceros (empresas) | ... |
| 67 | AWALAND | Número de seguros agrícolas | ... |
| 68 | APERSAUT | Número de seguros de automóvil | ... |
| 69 | ABESAUT | Número de seguros de camionetas | ... |
| 70 | AMOTSCO | Número de seguros de motocicletas/scooters | ... |
| 71 | AVRAAUT | Número de seguros de camiones | ... |
| 72 | AAANHANG | Número de seguros de remolque | ... |
| 73 | ATRACTOR | Número de seguros de tractor | ... |
| 74 | AWERKT | Número de seguros de maquinaria agrícola | ... |
| 75 | ABROM | Número de seguros de ciclomotores | ... |
| 76 | ALEVEN | Número de seguros de vida | ... |
| 77 | APERSONG | Número de seguros de accidentes personales | ... |
| 78 | AGEZONG | Número de seguros familiares de accidentes | ... |
| 79 | AWAOREG | Número de seguros por discapacidad | ... |
| 80 | ABRAND | Número de seguros contra incendios | ... |
| 81 | AZEILPL | Número de seguros de tabla de surf | ... |
| 82 | APLEZIER | Número de seguros de barco | ... |
| 83 | AFIETS | Número de seguros de bicicleta | ... |
| 84 | AINBOED | Número de seguros de propiedad | ... |
| 85 | ABYSTAND | Número de seguros de seguridad social | ... |
| 86 | CARAVAN | Número de pólizas de casas rodantes | 0 – 1 |


---

### 🧩 Niveles jerárquicos (L0 – L4)

Estas categorías representan niveles de **agregación geográfica** de variables sociodemográficas basadas en el código postal del cliente.

#### 🏙️ L0 –
Agrupa a los clientes según características socioeconómicas y estilo de vida.

| Valor | Descripción |
|-------|--------------|
| 1 | Ingreso alto, familia con hijos costosos |
| 2 | Provinciales importantes |
| 3 | Adultos mayores de alto estatus |
| 4 | Adultos mayores acomodados |
| 5 | Adultos mayores mixtos |
| 6 | Carrera y cuidado infantil |
| 7 | “Dinkis” (doble ingreso, sin hijos) |
| 8 | Familias de clase media |
| 9 | Familias modernas y completas |
| 10 | Familias estables |
| 11 | Familias jóvenes en formación |
| 12 | Familias jóvenes acomodadas |
| 13 | Familias jóvenes tradicionales |
| 14 | Jóvenes cosmopolitas |
| 15 | Adultos mayores cosmopolitas |
| 16 | Estudiantes en apartamentos |
| 17 | Recién graduados urbanos |
| 18 | Juventud soltera |
| 19 | Jóvenes suburbanos |
| 20 | Diversidad étnica |
| 21 | Jóvenes urbanos de bajos recursos |
| 22–41 | Otras categorías de familias, adultos mayores y zonas rurales |

#### 👥 L1 – 
| Valor | Rango de edad |
|-------|----------------|
| 1 | 20–30 años |
| 2 | 30–40 años |
| 3 | 40–50 años |
| 4 | 50–60 años |
| 5 | 60–70 años |
| 6 | 70–80 años |

#### 💼 L2 – 
| Valor | Descripción |
|-------|--------------|
| 1 | Hedonistas exitosos |
| 2 | Crecimiento profesional |
| 3 | Familia promedio |
| 4 | Profesionales solitarios |
| 5 | Buena calidad de vida |
| 6 | Adultos mayores activos |
| 7 | Jubilados religiosos |
| 8 | Familias con hijos adultos |
| 9 | Familias conservadoras |
| 10 | Agricultores |

#### ⛪ L3 – 
| Valor | Rango (%) |
|-------|------------|
| 0 | 0% |
| 1 | 1 – 10% |
| 2 | 11 – 23% |
| 3 | 24 – 36% |
| 4 | 37 – 49% |
| 5 | 50 – 62% |
| 6 | 63 – 75% |
| 7 | 76 – 88% |
| 8 | 89 – 99% |
| 9 | 100% |

#### 💰 L4 –
| Valor | Intervalo aproximado (€) |
|-------|---------------------------|
| 0 | 0 |
| 1 | 1 – 49 |
| 2 | 50 – 99 |
| 3 | 100 – 199 |
| 4 | 200 – 499 |
| 5 | 500 – 999 |
| 6 | 1,000 – 4,999 |
| 7 | 5,000 – 9,999 |
| 8 | 10,000 – 19,999 |
| 9 | 20,000 o más |

---

📎 **Nota:**  
Las variables que comienzan con “M”, “P” o “A” provienen de **datos agregados por código postal** y describen porcentajes o conteos promedio en la zona del cliente.  
Por ejemplo, `MHHUUR` indica la proporción de viviendas rentadas en el área postal, y `MAUT1` el porcentaje de hogares con un automóvil.

equipo21mna_coil2000
==============================

Pipeline de ML COIL 2000

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── eda_clean.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── preprocessing.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
## FastAPI

El artefacto `pipeline.joblib` contiene tanto las transformaciones de datos como el clasificador final.
Este modelo fue registrado como la versión 1 (`src/models/pipeline.joblib`) en el flujo de trabajo MLOps y es utilizado por el contenedor `coil-2000:1.0` para ofrecer predicciones mediante la API REST.

## 🐳 Contenedor Docker

**Repositorio de imagen:** [https://hub.docker.com/r/jiek03/coil-2000](https://hub.docker.com/r/jiek03/coil-2000)

### Versiones publicadas
| Tag | Descripción | Fecha |
|-----|--------------|--------|
| `1.0` | Versión inicial del servicio FastAPI para el dataset COIL 2000. | 11/11/2025 |
| `latest` | Alias de la versión estable más reciente. | 11/11/2025 |

---

### 🧩 Ejecución local
Cualquier usuario puede ejecutar el contenedor sin necesidad de credenciales, solo debe tener **Docker Desktop instalado**.

```bash
# Descargar la imagen desde Docker Hub
docker pull jiek03/coil-2000:tagname

# Ejecutar el servicio FastAPI
docker run -p 8000:8000 jiek03/coil-2000:tagname

Luego abre en el navegador:

http://localhost:8000/docs

### 👥 Otros usuarios y etiquetas

Cualquier persona puede crear su propia copia y publicarla en su cuenta de Docker Hub:

# Crear una nueva etiqueta con su usuario
docker tag jiek03/coil-2000:tagname user/coil-2000:tagname

# Subir la imagen a su propio repositorio
docker push user/coil-2000:tagname


# Para actualizar versiones dentro del mismo repositorio:

docker tag jiek03/coil-2000:tagname jiek03/coil-2000:latest
docker push jiek03/coil-2000:latest


El alias latest puede actualizarse para apuntar a la versión más reciente.

