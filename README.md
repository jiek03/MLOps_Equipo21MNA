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

| Nº | Nombre | Descripción | Dominio / Rango |
|----|----------|--------------|------------------|
| 1 | MOSTYPE | Subtipo de cliente | Ver niveles L0 |
| 2 | MAANTHUI | Número de viviendas | 1 – 10 |
| 3 | MGEMOMV | Tamaño promedio del hogar | 1 – 6 |
| 4 | MGEMLEEF | Edad promedio | Ver niveles L1 |
| 5 | MOSHOOFD | Tipo principal de cliente | Ver niveles L2 |
| 6–9 | MGODRK – MGODGE | Religión (católica, protestante, otra o sin religión) | Ver niveles L3 |
| 10–15 | MRELGE – MFWEKIND | Estado civil y tipo de hogar | Binarias / proporciones |
| 16–18 | MOPLHOOG – MOPLLAAG | Nivel educativo (alto, medio, bajo) | Proporciones |
| 19–24 | MBERHOOG – MBERARBO | Ocupación o categoría laboral | Proporciones |
| 25–29 | MSKA – MSKD | Clase social (A–D) | Proporciones |
| 30–31 | MHHUUR / MHKOOP | Vivienda rentada / propia | 0–1 |
| 32–34 | MAUT1 – MAUT0 | Número de autos | 0–2 |
| 35–36 | MZFONDS / MZPART | Tipo de seguro médico | Binarias |
| 37–41 | MINKM30 – MINK123M | Nivel de ingreso | Intervalos (<30k, 30–45k, 45–75k, 75–122k, >123k) |
| 42 | MINKGEM | Ingreso promedio | Numérico |
| 43 | MKOOPKLA | Clase de poder adquisitivo | Escala ordinal |
| 44–64 | PWAPART – PBYSTAND | Contribución a tipos de seguros (auto, vida, incendio, etc.) | Ver niveles L4 |
| 65–85 | AWAPART – ABYSTAND | Número de pólizas de cada tipo | 1 – 12 |
| 86 | CARAVAN | Póliza de casa rodante (variable objetivo) | 0 = No, 1 = Sí |

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

