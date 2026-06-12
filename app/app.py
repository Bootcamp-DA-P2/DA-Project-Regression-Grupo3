import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Imports de Machine Learning para el dashboard analítico
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- CONFIGURACIÓN ----
st.set_page_config(
    page_title="LGBTI Acceptance Index",
    page_icon="🌈",
    layout="wide"
)

# ---- CARGA DE DATOS ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_regresion.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Arreglo: unificar Czech Republic / Czechia
    df['CountryCode'] = df['CountryCode'].replace({'Czech Republic': 'Czechia'})
    df['CountryName'] = df['CountryName'].replace({'Czech Republic': 'Czechia'})
    df = df.drop_duplicates(subset=['CountryCode', 'year'])

    return df


@st.cache_resource
def train_model_dashboard(df_input):
    columnas_features = [
        "gdp_per_capita",
        "gini_index",
        "education_spending",
        "urbanization_rate",
        "unemployment_rate"
    ]

    X = df_input[columnas_features]
    y = df_input["acceptance_index"]

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(modelo, X, y, cv=cv)

    modelo.fit(X, y)

    importancia = pd.DataFrame({
        "feature": columnas_features,
        "coeficiente": modelo.named_steps["ridge"].coef_
    }).sort_values("coeficiente", key=abs, ascending=True)

    return {
        "modelo": modelo,
        "y_real": y,
        "y_pred": y_pred,
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "importancia": importancia
    }


df = load_data()
resultados_modelo = train_model_dashboard(df)

# ---- NAVEGACIÓN ----
st.sidebar.title("🌈 LGBTI Acceptance")
pagina = st.sidebar.radio(
    "Navegación",
    ["📊 Dashboard", "🔮 Predictor", "📝 Sobre el proyecto"]
)

# ---- PÁGINAS ----
if pagina == "📊 Dashboard":
    st.title("📊 Dashboard — Aceptación LGBTI en Europa")

    # ---- MÉTRICAS RÁPIDAS ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total países", df['CountryCode'].nunique())

    with col2:
        st.metric("Total respuestas", f"{len(df):,}")

    with col3:
        st.metric("Años disponibles", "2012 — 2019")

    # ---- TABLA DE DATOS ----
    st.subheader("Vista del dataset")
    st.dataframe(df.head(20))

    # ---- MAPA DE EUROPA ----
    st.subheader("🗺️ Acceptance Index por país")

    año_seleccionado = st.selectbox("Selecciona un año", sorted(df["year"].unique()))

    df_mapa = df[df["year"] == año_seleccionado]

    fig = px.choropleth(
        df_mapa,
        locations="CountryName",
        locationmode="country names",
        color="acceptance_index",
        scope="europe",
        color_continuous_scale="Viridis",
        range_color=(0, 100),
        title=f"Acceptance Index — {año_seleccionado}",
        hover_name="CountryName",
        hover_data={"acceptance_index": ":.1f", "CountryName": False},
        labels={"acceptance_index": "Acceptance Index"}
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.3)",
        showframe=False
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_x=0.5,
        title_font_size=20,
        coloraxis_colorbar=dict(
            title="Acceptance<br>Index",
            thickness=15,
            len=0.7
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- RANKING DE PAÍSES ----
    st.subheader("🏆 Ranking de países por Acceptance Index")

    df_ranking = df_mapa.sort_values("acceptance_index", ascending=True)

    fig_ranking = px.bar(
        df_ranking,
        x="acceptance_index",
        y="CountryName",
        orientation="h",
        color="acceptance_index",
        color_continuous_scale="RdYlGn",
        range_color=(0, 100),
        title=f"Ranking — {año_seleccionado}",
        labels={"acceptance_index": "Acceptance Index", "CountryName": "País"}
    )
    fig_ranking.update_layout(height=700)
    st.plotly_chart(fig_ranking, use_container_width=True)

    # ---- MATRIZ DE CORRELACIONES ----
    st.subheader("🔗 Matriz de correlaciones")

    columnas_numericas = [
        "acceptance_index", "gdp_per_capita", "gini_index",
        "education_spending", "urbanization_rate", "unemployment_rate"
    ]

    corr = df[columnas_numericas].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        range_color=(-1, 1),
        title="Correlación entre variables"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- RENDIMIENTO DEL MODELO ----
    st.subheader("🤖 Rendimiento del modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R²", f"{resultados_modelo['r2']:.3f}")

    with col2:
        st.metric("MAE", f"{resultados_modelo['mae']:.2f}")

    with col3:
        st.metric("RMSE", f"{resultados_modelo['rmse']:.2f}")

    # ---- PREDICCIÓN VS REAL ----
    st.subheader("📈 Valores reales vs predichos")

    df_pred = pd.DataFrame({
        "Real": resultados_modelo["y_real"],
        "Predicho": resultados_modelo["y_pred"]
    })

    fig_pred = px.scatter(
        df_pred,
        x="Real",
        y="Predicho",
        title="Comparación entre valores reales y predichos"
    )
    fig_pred.add_shape(
        type="line", x0=0, y0=0, x1=100, y1=100,
        line=dict(color="red", dash="dash")
    )
    fig_pred.update_layout(height=600)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---- GRÁFICO DE RESIDUOS ----
    st.subheader("📉 Análisis de residuos")

    residuos = resultados_modelo["y_real"] - resultados_modelo["y_pred"]

    df_residuos = pd.DataFrame({
        "Predicción": resultados_modelo["y_pred"],
        "Residuo": residuos
    })

    fig_res = px.scatter(
        df_residuos,
        x="Predicción",
        y="Residuo",
        title="Distribución de residuos"
    )
    fig_res.add_hline(y=0, line_dash="dash")
    fig_res.update_layout(height=600)
    st.plotly_chart(fig_res, use_container_width=True)

    # ---- FEATURE IMPORTANCE ----
    st.subheader("🧩 Importancia de las variables")

    fig_imp = px.bar(
        resultados_modelo["importancia"],
        x="coeficiente",
        y="feature",
        orientation="h",
        color="coeficiente",
        color_continuous_scale="RdBu",
        title="Coeficientes del modelo Ridge"
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)

elif pagina == "🔮 Predictor":
    st.title("🔮 Predictor de Aceptación LGBTI")
    st.info("🚧 Predictor en construcción — esperando modelo")

elif pagina == "📝 Sobre el proyecto":
    st.title("📝 Sobre el proyecto")

    st.write("""
    Proyecto desarrollado para analizar la relación entre
    factores socioeconómicos y el índice de aceptación LGBTI en Europa.

    **Hipótesis:** ¿Pueden los indicadores socioeconómicos de un país
    predecir el nivel de aceptación percibida del colectivo LGBTI?

    **Datos:** EU LGBTI Survey (FRA) — 2012 y 2019, combinados con
    indicadores del Banco Mundial (PIB, Gini, educación, urbanización, desempleo).

    **Modelo:** Regresión Ridge con validación cruzada K-Fold.
    """)