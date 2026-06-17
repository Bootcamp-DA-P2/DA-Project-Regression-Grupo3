# DA-Project-Regression-Grupo3

Machine Learning regression project that predicts an LGBT Acceptance Index across European countries using FRA survey data and socioeconomic indicators.

# 🏳️‍🌈 LGBTI Acceptance Index — Proyecto de Regresión

> *¿Pueden los indicadores socioeconómicos de un país predecir el nivel de aceptación percibida del colectivo LGBTI en Europa?*

Proyecto desarrollado en el marco del **Bootcamp Data Analyst (Junio 2026)** por Rita, Yasira y Romi, con motivo del mes del Orgullo LGBTIQ+.

---

## 📋 Descripción

Este proyecto aplica técnicas de **Machine Learning supervisado (regresión)** para analizar la relación entre indicadores socioeconómicos (PIB, educación, desigualdad, urbanización, desempleo) y el nivel de aceptación percibida del colectivo LGBTI en 30 países europeos, a partir de los datos de la encuesta oficial de la **Agencia de Derechos Fundamentales de la UE (FRA)**.

### Hipótesis
> El gasto en educación, la urbanización y el PIB per cápita predicen significativamente el nivel de aceptación LGBTI percibida por país.

**Resultado:** ✅ Confirmada — R² Test = 0.855, Overfitting = 3.0%

---

## 🗂️ Estructura del proyecto

```
DA-Project-Regression-Grupo3/
│
├── app/
│   └── app.py                          # Aplicación Streamlit
│
├── data/
│   ├── raw/
│   │   ├── 2012/                       # CSVs originales FRA 2012
│   │   └── 2019/                       # CSVs originales FRA 2019
│   └── processed/
│       ├── master_fra.csv              # Dataset unificado (Yasira)
│       ├── master_fra_limpio.csv       # Limpieza fase 1 (Rita)
│       ├── master_fra_verificado.csv   # Limpieza fase 2 (Romi)
│       ├── acceptance_index.csv        # Índice construido (Yasira)
│       ├── worldbank_indicators.csv    # Datos World Bank (Yasira)
│       └── dataset_regresion.csv      # Dataset final para ML
│
├── models/                             # Modelos entrenados (.pkl)
│
├── notebooks/
│   ├── 02_acceptance_index_design.ipynb  # Índice + EDA (Yasira)
│   ├── 2_limpieza_verificacion.ipynb     # Limpieza (Romi)
│   ├── 03_eda_regresion.ipynb            # EDA + Modelado (Yasira)
│   └── 04_overfitting_validacion.ipynb   # Validación (Romi)
│
├── src/
│   ├── build_master_fra.py             # Pipeline de datos
│   ├── discover_fra_files.py           # Exploración de archivos
│   └── inspect_csv_structure.py        # Inspección de estructura
│
├── config/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Datos

### Fuente principal
- **EU LGBTI Survey II** — FRA (European Union Agency for Fundamental Rights)
  - Oleada 2012: 102.342 filas
  - Oleada 2019: 168.201 filas
  - **Total:** 270.543 respuestas · 30 países europeos

### Variables externas (World Bank API)
| Variable | Código World Bank |
|---|---|
| PIB per cápita | `NY.GDP.PCAP.CD` |
| Índice de Gini | `SI.POV.GINI` |
| Gasto en educación (% PIB) | `SE.XPD.TOTL.GD.ZS` |
| Tasa de urbanización | `SP.URB.TOTL.IN.ZS` |
| Tasa de desempleo | `SL.UEM.TOTL.ZS` |

### Dataset final para ML
- **58 filas** (30 países × 2 años)
- **7 columnas** (year + 5 features + acceptance_index)

---

## 🧠 Acceptance Index

El **Acceptance Index** (0–100) mide el nivel de aceptación social percibida por el colectivo LGBTI en cada país. Se construye en 4 etapas:

1. **Clasificación con IA** — Groq LLaMA clasifica cada bloque de preguntas de la encuesta como relevante/irrelevante, con dirección positiva/negativa y categoría temática
2. **Correcciones manuales** — Revisión experta de la clasificación automática
3. **Ponderación por bloque** — Cada respuesta recibe un peso según su dirección; cada bloque pesa igual independientemente del número de preguntas
4. **Normalización 0–100** — Score final comparable entre países y años

---

## 🤖 Modelo

### Modelos evaluados

| Modelo | R² Train | R² Test | Overfitting |
|---|---|---|---|
| Regresión Lineal | 0.406 | 0.097 | 0.309 ❌ |
| Ridge (selección features) | 0.342 | −0.112 | 0.454 ❌ |
| Ridge + K-Fold | 0.860 | 0.801 | 0.059 ⚠️ |
| **Ridge + K-Fold + Year** ✅ | **0.884** | **0.855** | **0.030** ✅ |

### Modelo ganador: Ridge Regression + K-Fold CV

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=10))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### Métricas finales

| Métrica | Valor |
|---|---|
| R² Test | 0.855 |
| RMSE | 10.13 |
| MAE | ~8.5 |
| Overfitting | 0.030 ✅ (<5%) |

### Feature importance (coeficientes Ridge)

| Variable | Coeficiente | Interpretación |
|---|---|---|
| `gdp_per_capita` | +4.92 | A mayor PIB, mayor aceptación |
| `urbanization_rate` | +4.78 | Países más urbanizados = más aceptación |
| `education_spending` | +3.38 | Más gasto educativo = más aceptación |
| `gini_index` | −3.33 | Mayor desigualdad = menor aceptación |
| `unemployment_rate` | +1.11 | Efecto débil |

### ⚠️ Limitación metodológica: variable `year`

La variable `year` presenta un coeficiente de −19.55, muy superior al resto, debido a un artefacto de normalización min-max global del `acceptance_index` (2012 y 2019 juntos). El modelo aprende en parte "si es 2019, restar puntos", lo cual no representa una relación real. Los coeficientes socioeconómicos son los relevantes para la hipótesis.

---

## 🚀 Instalación y uso

### Requisitos previos
- Python 3.11+
- Git

### Setup

```bash
# Clonar el repositorio
git clone https://github.com/rnavea-r/DA-Project-Regression-Grupo3.git
cd DA-Project-Regression-Grupo3

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Activar entorno (Mac/Linux)
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar la aplicación Streamlit

```bash
cd app
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

---

## 📱 Aplicación Streamlit

La app incluye tres secciones:

### 📊 Dashboard analítico
- Mapa de Europa coloreado por Acceptance Index (interactivo, selector 2012/2019)
- Ranking de países
- Matriz de correlaciones entre variables
- Sección de rendimiento del modelo (R², RMSE, predicción vs real, residuos, feature importance)

### 🔮 Predictor
- El usuario introduce valores de PIB, Gini, educación, urbanización y desempleo
- El modelo predice el Acceptance Index esperado para ese perfil de país

### 📝 Sobre el proyecto
- Descripción de la hipótesis, datos y metodología

---

## 🛠️ Tecnologías

| Categoría | Tecnologías |
|---|---|
| Lenguaje | Python 3.11 |
| Datos | Pandas, NumPy, OpenPyXL |
| ML | scikit-learn, Ridge Regression |
| IA (clasificación) | Groq API (LLaMA 3.3 70B) |
| Datos externos | World Bank API (`wbgapi`) |
| Visualización | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Control de versiones | Git, GitHub |

---

## 👩‍💻 Equipo

| Integrante | Responsabilidad |
|---|---|
| **Yasira** | Pipeline de datos, construcción del Acceptance Index, EDA, modelado |
| **Rita** | Limpieza de datos (nulos, duplicados), documentación |
| **Romi** | Verificación de datos, validación del modelo (overfitting, residuos), Streamlit |

Kanban a través de [github](https://github.com/users/rnavea-r/projects/3)

---

## 📁 Notebooks

| Notebook | Contenido | Autora |
|---|---|---|
| `02_acceptance_index_design.ipynb` | Clasificación de preguntas, construcción del índice, merge con World Bank | Yasira |
| `2_limpieza_verificacion.ipynb` | Verificación de tipos, países y consistencia | Romi |
| `03_eda_regresion.ipynb` | EDA completo, comparativa de modelos, selección del ganador | Yasira |
| `04_overfitting_validacion.ipynb` | Validación cruzada, overfitting, residuos, feature importance | Romi |

---

## 📄 Licencia

Proyecto académico — Bootcamp Data Analyst, Junio 2026.

Los datos de la encuesta EU LGBTI Survey son propiedad de la **Agencia de Derechos Fundamentales de la Unión Europea (FRA)** y se utilizan con fines educativos.

---

*Desarrollado con 🏳️‍🌈 durante el mes del Orgullo LGBTIQ+ — Junio 2026*
