import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="LGBTI Acceptance Index",
    page_icon="🌈",
    layout="wide"
)

# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data
def load_data():
    path = Path(__file__).parent.parent / "data" / "processed" / "dataset_regresion.csv"
    return pd.read_csv(path)

df = load_data()

# ============================================================
# SIDEBAR — FILTROS GLOBALES
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Pride_Flag_of_the_United_Kingdom.svg/320px-Pride_Flag_of_the_United_Kingdom.svg.png", width=200)
st.sidebar.title("🌈 LGBTI Acceptance Index")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    ["📊 Dashboard", "📝 Sobre el proyecto"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtros")

# Selector de años
años = sorted(df["year"].unique().tolist())
año_seleccionado = st.sidebar.selectbox("Año", años, index=1)

# Selector de países
paises = sorted(df["CountryCode"].unique().tolist())
paises_seleccionados = st.sidebar.multiselect(
    "Países",
    paises,
    default=paises
)

# Selector de indicador
indicadores = {
    "PIB per cápita": "gdp_per_capita",
    "Índice Gini": "gini_index",
    "Gasto en educación": "education_spending",
    "Tasa de urbanización": "urbanization_rate",
    "Tasa de desempleo": "unemployment_rate"
}
indicador_label = st.sidebar.selectbox("Indicador", list(indicadores.keys()))
indicador = indicadores[indicador_label]

# Filtrar datos según selección
df_filtrado = df[
    (df["year"] == año_seleccionado) &
    (df["CountryCode"].isin(paises_seleccionados))
]

# ============================================================
# PÁGINAS
# ============================================================
if pagina == "📊 Dashboard":
    st.title("📊 Dashboard — Aceptación LGBTI en Europa")
    st.write("Contenido próximamente...")

elif pagina == "📝 Sobre el proyecto":
    st.title("📝 Sobre el proyecto")
    st.write("Contenido próximamente...")