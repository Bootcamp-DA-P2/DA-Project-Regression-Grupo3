## Creación datos de práctica

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---- CONFIGURACIÓN ----
st.set_page_config(
    page_title="LGBTI Acceptance Index",
    page_icon="🌈",
    layout="wide"
)

# ---- CARGA DE DATOS ----
@st.cache_data
def load_data():
    # Cuando esté listo el CSV final lo cargamos aquí
    df = pd.read_csv('../data/processed/master_fra_verificado.csv')
    return df

df = load_data()

# ---- NAVEGACIÓN ----
st.sidebar.title("🌈 LGBTI Acceptance")
pagina = st.sidebar.radio(
    "Navegación",
    ["📊 Dashboard", "🔮 Predictor", "📝 Sobre el proyecto"]
)

# ---- PÁGINAS ----
if pagina == "📊 Dashboard":
    st.title("📊 Dashboard — Aceptación LGBTI en Europa")
    
    # Métricas rápidas en la parte superior
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total países", df['CountryCode'].nunique())
    
    with col2:
        st.metric("Total respuestas", f"{len(df):,}")
    
    with col3:
        st.metric("Años disponibles", "2012 — 2019")
    
    # Tabla de datos
    st.subheader("Vista del dataset")
    st.dataframe(df.head(20))

elif pagina == "🔮 Predictor":
    st.title("🔮 Predictor de Aceptación LGBTI")
    st.info("🚧 Predictor en construcción — esperando modelo")

elif pagina == "📝 Sobre el proyecto":
    st.title("📝 Sobre el proyecto")
    st.markdown("""
    **Hipótesis:** ¿Pueden los indicadores socioeconómicos de un país 
    predecir el nivel de discriminación y violencia que sufre el colectivo LGBTI?
    
    **Datos:** EU LGBTI Survey (FRA) — 2012 y 2019
    
    **Variables externas:** PIB per cápita, educación, Gini, desempleo, democracia
    
    **Modelo:** Regression con Random Forest + XGBoost
    """)