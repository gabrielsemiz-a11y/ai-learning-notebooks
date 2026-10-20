import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Laboratorio Integral de Machine Learning",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0f172a;
        padding: 10px 15px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px;
        color: #f8fafc;
        font-weight: 600;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Laboratorio Integral de Machine Learning")
st.markdown("Plataforma interactiva para **Clasificación** y **Regresión**: limpieza de datos, asociación estadística (Chi-cuadrado, Cramér, Spearman), entrenamiento reproducible, AutoML con Optuna y predicción en producción.")

# ---------------------------------------------------------
# ESTADO DE SESIÓN (INICIALMENTE LIMPIO DE DATASETS PRECARGADOS)
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Ninguno"
if "historial_clf" not in st.session_state:
    st.session_state.historial_clf = []
if "historial_reg" not in st.session_state:
    st.session_state.historial_reg = []
if "veredicto_llm" not in st.session_state:
    st.session_state.veredicto_llm = ""

MODEL_HELP = {
    "Regresión Logística": "📌 **Para qué sirve:** Clasificador lineal probabilístico basado en la función sigmoide. Es explicable, veloz y el estándar para calibrar umbrales de decisión.",
    "K-Vecinos Cercanos (KNN Clasificador)": "📌 **Para qué sirve:** Clasifica asignando la categoría mayoritaria entre los 'K' registros históricos matemáticamente más parecidos.",
    "Árbol de Decisión (Clasificador)": "📌 **Para qué sirve:** Divide el espacio de datos con reglas Si/Entonces jerárquicas muy transparentes para auditorías.",
    "Random Forest (Clasificador)": "📌 **Para qué sirve:** Combina decenas de árboles de decisión independientes (Bagging). Reduce drásticamente el sobreajuste y ofrece alta precisión.",
    "HistGradientBoosting (Clasificador)": "📌 **Para qué sirve:** Árboles secuenciales por potenciación de gradiente optimizados con histogramas. Altísima velocidad y precisión superior en tablas.",
    "Red Neuronal (MLP Classifier)": "📌 **Para qué sirve:** Perceptrón multicapa capaz de modelar relaciones no lineales intrincadas entre múltiples variables.",
    "Regresión Lineal Penalizada (Ridge)": "📌 **Para qué sirve:** Estima valores numéricos continuos aplicando regularización L2 para evitar coeficientes desmedidos o colinealidad.",
    "K-Vecinos Cercanos (KNN Regresor)": "📌 **Para qué sirve:** Estima un valor continuo promediando los valores reales de los 'K' registros más parecidos.",
    "Árbol de Decisión (Regresor)": "📌 **Para qué sirve:** Segmenta datos en hojas terminales y predice el promedio numérico del grupo correspondiente.",
    "Random Forest (Regresor)": "📌 **Para qué sirve:** Promedia las predicciones de múltiples árboles para estimar magnitudes con mínima varianza.",
    "HistGradientBoosting (Regresor)": "📌 **Para qué sirve:** Boosting de alta velocidad para estimación numérica continua de máxima precisión.",
    "Red Neuronal (MLP Regressor)": "📌 **Para qué sirve:** Red neuronal multicapa con salida lineal para ajustar curvas y funciones continuas complejas."
}

tabs = st.tabs([
    "1️⃣ Carga y Limpieza",
    "2️⃣ Tarea y Selección Estadística",
    "3️⃣ Entrenamiento y AutoML",
    "4️⃣ Evaluación y Veredicto LLM",
    "5️⃣ Producción",
    "📘 Guía de Modelos y Métricas"
])

# ==============================================================================
# PASO 1: CARGA Y LIMPIEZA DEL DATASET
# ==============================================================================
with tabs[0]:
    st.header("1️⃣ Carga, Imputación e Inspección de Valores Atípicos")
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader("📂 Sube cualquier archivo CSV para analizar:", type=["csv"])
        if uploaded_file is not None:
            if st.session_state.get("dataset_name") != uploaded_file.name or st.session_state["df"] is None:
                st.session_state["df"] = pd.read_csv(uploaded_file)
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["target_col"] = None
                st.session_state["selected_features"] = []
                st.session_state["df_clean"] = None
                st.session_state["trained_model"] = None
                st.session_state["test_outputs"] = None
                st.success(f"Archivo cargado correctamente: `{uploaded_file.name}`")
                st.rerun()

    with col_up2:
        st.write("")
        st.write("")
        if st.session_state["df"] is not None:
            if st.button("🗑️ Quitar Dataset Activo"):
                st.session_state["df"] = None
                st.session_state["dataset_name"] = "Ninguno"
                st.session_state["target_col"] = None
                st.session_state["selected_features"] = []
                st.session_state["df_clean"] = None
                st.session_state["trained_model"] = None
                st.session_state["test_outputs"] = None
                st.info("Dataset eliminado de memoria.")
                st.rerun()

    df = st.session_state["df"]

    if df is None:
        st.info("📂 Por favor, sube un archivo CSV en el cargador superior para comenzar a trabajar en el laboratorio.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset Activo", st.session_state.get("dataset_name", "CSV"))
        m2.metric("Total de Filas", f"{df.shape[0]:,}")
        m3.metric("Total de Columnas", f"{df.shape[1]}")
        m4.metric("Valores Faltantes (NaN)", f"{int(df.isnull().sum().sum())}")

        with st.expander("👀 Vista previa de las primeras filas y tipos de datos", expanded=False):
            st.dataframe(df.head(8), use_container_width=True)

        st.divider()

        st.subheader("A. Tratamiento de Valores Faltantes (NaN)")
        con_nulos = df.isnull().sum()
        con_nulos = con_nulos[con_nulos > 0]

        if len(con_nulos) == 0:
            st.success("✅ El dataset no tiene valores faltantes en ninguna columna.")
        else:
            df_missing = pd.DataFrame({
                "Columna": con_nulos.index,
                "Tipo de Dato": [str(df[c].dtype) for c in con_nulos.index],
                "Faltantes": con_nulos.values,
                "% del Total": [f"{(v / len(df) * 100):.2f}%" for v in con_nulos.values]
            })
            st.dataframe(df_missing, use_container_width=True)

            c_imp1, c_imp2, c_imp3 = st.columns(3)
            with c_imp1:
                cols_a_imputar = st.multiselect("Columnas a corregir:", list(con_nulos.index), default=list(con_nulos.index))
            with c_imp2:
                estrategia_num = st.selectbox("Para Numéricas:", ["Mediana (Recomendada)", "Promedio (Media)", "Moda", "Valor Cero (0)", "Eliminar filas"])
            with c_imp3:
                estrategia_cat = st.selectbox("Para Categóricas:", ["Moda (Más frecuente)", "Etiqueta 'DESCONOCIDO'", "Eliminar filas"])

            if st.button("✨ Aplicar Imputación de Faltantes", type="primary"):
                for col in cols_a_imputar:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if estrategia_num.startswith("Mediana"):
                            df[col] = df[col].fillna(df[col].median())
                        elif estrategia_num.startswith("Promedio"):
                            df[col] = df[col].fillna(df[col].mean())
                        elif estrategia_num.startswith("Moda"):
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                        elif estrategia_num.startswith("Valor Cero"):
                            df[col] = df[col].fillna(0)
                        else:
                            df = df.dropna(subset=[col])
                    else:
                        if estrategia_cat.startswith("Moda"):
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "DESCONOCIDO")
                        elif estrategia_cat.startswith("Etiqueta"):
                            df[col] = df[col].fillna("DESCONOCIDO")
                        else:
                            df = df.dropna(subset=[col])
                st.session_state["df"] = df
                st.success("¡Imputación aplicada y sincronizada en memoria!")
                st.rerun()

        st.divider()

        st.subheader("B. Inspección Visual y Tratamiento Selectivo de Valores Atípicos (Outliers)")
        cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if cols_numericas:
            col_outlier = st.selectbox("Selecciona una columna numérica para inspeccionar:", cols_numericas)
            
            q1 = float(df[col_outlier].quantile(0.25))
            q3 = float(df[col_outlier].quantile(0.75))
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            
            mascara_outliers = (df[col_outlier] < lim_inf) | (df[col_outlier] > lim_sup)
            cant_outliers = int(mascara_outliers.sum())

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Límite Inferior (IQR)", f"{lim_inf:,.2f}")
            o2.metric("Límite Superior (IQR)", f"{lim_sup:,.2f}")
            o3.metric("Casos Atípicos Detectados", f"{cant_outliers}")
            o4.metric("% de la Columna", f"{(cant_outliers / len(df) * 100):.2f}%")

            fig_out = px.box(
                df, x=col_outlier, points="outliers",
                title=f"Distribución Interactiva y Puntos Atípicos de '{col_outlier}' (Pasa el cursor para ver valores)",
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark"
            )
            fig_out.add_vline(x=lim_inf, line_dash="dash", line_color="#ef4444", annotation_text="Lím. Inf")
            fig_out.add_vline(x=lim_sup, line_dash="dash", line_color="#ef4444", annotation_text="Lím. Sup")
            st.plotly_chart(fig_out, use_container_width=True)

            if cant_outliers > 0:
                df_outliers = df[mascara_outliers].copy()
                df_outliers['Tipo Atípico'] = np.where(df_outliers[col_outlier] > lim_sup, '🔺 Superior (> máx)', '🔻 Inferior (< mín)')
                cols_ord = [col_outlier, 'Tipo Atípico'] + [c for c in df_outliers.columns if c not in [col_outlier, 'Tipo Atípico']]
                df_outliers = df_outliers[cols_ord]

                c_om1, c_om2 = st.columns(2)
                with c_om1:
                    modo_sel_out = st.radio(
                        "¿A qué registros deseas aplicar la solución?",
                        ["Todos los casos atípicos detectados", "Seleccionar manualmente casos específicos"]
                    )
                with c_om2:
                    accion_out = st.radio(
                        "Solución propuesta:",
                        ["Recortar a los límites IQR (Capping)", "Reemplazar con la Mediana", "Eliminar filas seleccionadas"]
                    )

                if modo_sel_out == "Todos los casos atípicos detectados":
                    indices_corregir = df_outliers.index.tolist()
                    st.dataframe(df_outliers, use_container_width=True)
                else:
                    todos_idx = df_outliers.index.tolist()
                    indices_corregir = st.multiselect(
                        "Selecciona las filas específicas a modificar:",
                        options=todos_idx,
                        default=todos_idx[:min(5, len(todos_idx))],
                        format_func=lambda idx: f"Fila #{idx} | {col_outlier} = {df.loc[idx, col_outlier]:,.2f} ({df_outliers.loc[idx, 'Tipo Atípico']})"
                    )
                    if indices_corregir:
                        st.dataframe(df_outliers.loc[indices_corregir], use_container_width=True)

                if st.button("🛠️ Aplicar Solución a los Casos Seleccionados", type="primary"):
                    if not indices_corregir:
                        st.warning("Selecciona al menos una fila.")
                    else:
                        if accion_out.startswith("Recortar"):
                            df.loc[indices_corregir, col_outlier] = df.loc[indices_corregir, col_outlier].clip(lower=lim_inf, upper=lim_sup)
                        elif accion_out.startswith("Reemplazar"):
                            df.loc[indices_corregir, col_outlier] = float(df[col_outlier].median())
                        elif accion_out.startswith("Eliminar"):
                            df = df.drop(index=indices_corregir)
                        st.session_state["df"] = df
                        st.success(f"¡Se corrigieron {len(indices_corregir)} registros exitosamente y se sincronizaron en memoria!")
                        st.rerun()

df = st.session_state.get("df")

# ==============================================================================
# PASO 2: TAREA Y SELECCIÓN ESTADÍSTICA
# ==============================================================================
with tabs[1]:
    st.header("2️⃣ Tipo de Problema (Clasificación vs. Regresión) y Selección Estadística")

    if df is None:
        st.warning("👈 Primero debes cargar un archivo CSV en la pestaña '1️⃣ Carga y Limpieza'.")
    else:
        columnas_totales = list(df.columns)
        def_target_idx = len(columnas_totales) - 1

        col_t1, col_t2, col_t3 = st.columns([1.2, 1.5, 1.3])
        with col_t1:
            target_col = st.selectbox("🎯 1. Elige la Variable Objetivo (Target):", columnas_totales, index=def_target_idx, key="sb_target_col")
            st.session_state["target_col"] = target_col
            df_clean = df.dropna(subset=[target_col]).copy()
            st.session_state["df_clean"] = df_clean
            n_unicos = df_clean[target_col].nunique()
            es_numerica_target = pd.api.types.is_numeric_dtype(df_clean[target_col])

            if not es_numerica_target or n_unicos <= 10:
                sugerencia_tarea = "Clasificación (Predecir Categoría / Clase)"
            else:
                sugerencia_tarea = "Regresión (Estimar un Valor Numérico Continuo)"

        with col_t2:
            opciones_tarea = [
                "Clasificación (Predecir Categoría / Clase)",
                "Regresión (Estimar un Valor Numérico Continuo)"
            ]
            tipo_problema = st.radio(
                "🧭 2. Tipo de Problema (Detectado automáticamente, puedes cambiarlo):",
                opciones_tarea,
                index=opciones_tarea.index(sugerencia_tarea),
                key="rb_tipo_problema"
            )
            es_clasificacion = tipo_problema.startswith("Clasificación")
            st.session_state["es_clasificacion"] = es_clasificacion

        with col_t3:
            if es_clasificacion:
                st.info(f"🏷️ **Modo Clasificación Activo**\n\nClases detectadas en `{target_col}`: **{n_unicos}**")
                if n_unicos > 20:
                    st.warning("⚠️ Esta columna tiene muchos valores distintos. ¿Seguro que no es un problema de Regresión?")
            else:
                if not es_numerica_target:
                    st.error("❌ Para Regresión (estimar un valor), la variable objetivo debe ser numérica.")
                    st.stop()
                st.info(f"📈 **Modo Regresión Activo**\n\nPromedio de `{target_col}`: **{df_clean[target_col].mean():,.2f}**")

        st.divider()

        candidatas_todas = [c for c in columnas_totales if c != target_col and c.lower() not in ['id', 'index', 'unnamed: 0']]
        if es_clasificacion:
            le_target = LabelEncoder()
            y_all = le_target.fit_transform(df_clean[target_col].astype(str))
            st.session_state["classes_labels"] = [str(c) for c in le_target.classes_]
            es_binaria = (len(le_target.classes_) == 2)
            st.session_state["es_binaria"] = es_binaria
            min_class_count = pd.Series(y_all).value_counts().min()
            strat = y_all if min_class_count >= 2 else None
            df_train_stat, _ = train_test_split(df_clean, test_size=0.40, stratify=strat, random_state=42)
            y_train_stat = le_target.transform(df_train_stat[target_col].astype(str))
        else:
            y_all = df_clean[target_col].values.astype(float)
            st.session_state["es_binaria"] = False
            df_train_stat, _ = train_test_split(df_clean, test_size=0.40, random_state=42)
            y_train_stat = df_train_stat[target_col].values.astype(float)

        num_cands = [c for c in candidatas_todas if pd.api.types.is_numeric_dtype(df_clean[c])]
        cat_cands = [c for c in candidatas_todas if c not in num_cands]

        st.subheader("3. Filtros Estadísticos de Asociación (Calculados en Entrenamiento)")
        col_stat_num, col_stat_cat = st.columns(2)

        with col_stat_num:
            st.markdown("#### 🔢 Variables Numéricas: Correlación de Spearman (rho)")
            if num_cands:
                temp_num = df_train_stat[num_cands].copy()
                temp_num["__TARGET__"] = y_train_stat
                rho_s = temp_num.corr(method="spearman")["__TARGET__"].drop("__TARGET__").dropna().sort_values()
                
                df_rho = pd.DataFrame({
                    "Variable": rho_s.index,
                    "Spearman (rho)": rho_s.values.round(3),
                    "Dirección": ["Positiva (+)" if v >= 0 else "Negativa (-)" for v in rho_s.values]
                })
                fig_rho = px.bar(
                    df_rho, x="Spearman (rho)", y="Variable", color="Dirección", orientation="h",
                    color_discrete_map={"Positiva (+)": "#f59e0b", "Negativa (-)": "#38bdf8"},
                    title=f"Asociación Monótona con '{target_col}'",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_rho, use_container_width=True)
            else:
                st.info("No hay variables predictoras numéricas.")

        with col_stat_cat:
            if es_clasificacion:
                st.markdown("#### 🏷️ Variables Categóricas: Chi-cuadrado y V de Cramér")
                if cat_cands:
                    res_cat = []
                    for col in cat_cands:
                        tab = pd.crosstab(df_train_stat[col].fillna("Sin dato"), y_train_stat)
                        chi2, p_val, _, esp = chi2_contingency(tab)
                        div = tab.to_numpy().sum() * min(tab.shape[0] - 1, tab.shape[1] - 1)
                        v_cram = np.sqrt(chi2 / div) if div > 0 else 0.0
                        res_cat.append({
                            "Variable": col,
                            "V de Cramér": round(float(v_cram), 3),
                            "p-valor": round(float(p_val), 4),
                            "% Esperadas < 5": f"{(esp < 5).mean():.1%}",
                            "Categorías": tab.shape[0]
                        })
                    df_cram = pd.DataFrame(res_cat).sort_values("V de Cramér", ascending=False)
                    st.dataframe(df_cram, use_container_width=True)

                    cat_inspect = st.selectbox("Ver proporción de la clase objetivo por categoría:", cat_cands)
                    if cat_inspect:
                        df_train_plot = df_train_stat.copy()
                        df_train_plot["Target_Num"] = y_train_stat
                        agg_cat = df_train_plot.groupby(cat_inspect)["Target_Num"].agg(Proporción="mean", Casos="count").reset_index().sort_values("Proporción")
                        fig_cat = px.bar(
                            agg_cat, x=cat_inspect, y="Proporción", text="Casos",
                            title=f"Tasa de '{target_col}' según {cat_inspect} (Texto = N° Casos)",
                            color_discrete_sequence=["#10b981"], template="plotly_dark"
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("No hay variables categóricas.")
            else:
                st.markdown("#### 🏷️ Variables Categóricas: ANOVA (F-Test) sobre el Valor")
                if cat_cands:
                    res_anova = []
                    for col in cat_cands:
                        grupos = [g.dropna().values for _, g in df_train_stat.groupby(col)[target_col] if len(g.dropna()) > 1]
                        f_stat, p_val = f_oneway(*grupos) if len(grupos) >= 2 else (0.0, 1.0)
                        res_anova.append({
                            "Variable Categórica": col,
                            "Estadístico F (ANOVA)": round(float(f_stat), 2),
                            "p-valor": round(float(p_val), 4),
                            "Categorías": df_train_stat[col].nunique()
                        })
                    st.dataframe(pd.DataFrame(res_anova).sort_values("Estadístico F (ANOVA)", ascending=False), use_container_width=True)

                    cat_inspect_reg = st.selectbox("Ver distribución del valor por categoría:", cat_cands)
                    if cat_inspect_reg:
                        fig_box_cat = px.box(
                            df_train_stat, x=cat_inspect_reg, y=target_col,
                            title=f"Distribución de '{target_col}' según {cat_inspect_reg}",
                            color_discrete_sequence=["#a855f7"], template="plotly_dark"
                        )
                        st.plotly_chart(fig_box_cat, use_container_width=True)
                else:
                    st.info("No hay variables categóricas.")

        st.divider()

        st.subheader("4. Selección de Variables Predictoras para los Modelos")
        
        default_cands = candidatas_todas[:min(5, len(candidatas_todas))]
        current_selection = st.session_state.get("selected_features", default_cands)
        current_selection = [c for c in current_selection if c in candidatas_todas]
        if not current_selection:
            current_selection = default_cands

        col_sel_f1, col_sel_f2 = st.columns([3, 1])
        with col_sel_f1:
            selected_features = st.multiselect(
                "✅ Elige las columnas predictoras (Features) para entrenar:",
                options=candidatas_todas,
                default=current_selection,
                key="ms_features_selector"
            )
            st.session_state["selected_features"] = selected_features

        with col_sel_f2:
            st.write("")
            st.write("")
            if st.button("🤖 Auto-Seleccionar por Estadística", help="Selecciona automáticamente las 5 variables con mayor correlación/asociación con el target"):
                mejores = []
                if num_cands:
                    temp_num_a = df_train_stat[num_cands].copy()
                    temp_num_a["__T__"] = y_train_stat
                    rho_abs = temp_num_a.corr(method="spearman")["__T__"].drop("__T__").abs().dropna().sort_values(ascending=False)
                    mejores.extend(list(rho_abs.head(4).index))
                if cat_cands:
                    mejores.append(cat_cands[0])
                mejores = [c for c in mejores if c in candidatas_todas][:5]
                if mejores:
                    st.session_state["selected_features"] = mejores
                    st.success(f"¡Seleccionadas automáticamente: {mejores}!")
                    st.rerun()

# ==============================================================================
# PASO 3: ENTRENAMIENTO MANUAL Y AUTOML (REPRODUCIBLE Y CON AYUDAS)
# ==============================================================================
with tabs[2]:
    st.header("3️⃣ Configuración del Modelo, Entrenamiento y AutoML")

    if df is None:
        st.warning("👈 Primero debes cargar un archivo CSV en la pestaña '1️⃣ Carga y Limpieza'.")
    else:
        selected_features = st.session_state.get("selected_features", [])
        if not selected_features:
            st.warning("⚠️ Selecciona al menos una variable predictora en la pestaña '2️⃣ Tarea y Selección Estadística'.")
        else:
            es_clasificacion = st.session_state.get("es_clasificacion", True)
            es_binaria = st.session_state.get("es_binaria", True)
            target_col = st.session_state.get("target_col", df.columns[-1])
            df_clean = st.session_state.get("df_clean", df.dropna(subset=[target_col]).copy())

            st.markdown(f"**Tarea:** `{'Clasificación' if es_clasificacion else 'Regresión'}` | **Target:** `{target_col}` | **Predictoras ({len(selected_features)}):** `{', '.join(selected_features)}`")

            if es_clasificacion:
                modelos_disponibles = [
                    "Regresión Logística",
                    "Random Forest (Clasificador)",
                    "HistGradientBoosting (Clasificador)",
                    "Red Neuronal (MLP Classifier)",
                    "K-Vecinos Cercanos (KNN Clasificador)",
                    "Árbol de Decisión (Clasificador)"
                ]
                if TF_AVAILABLE and es_binaria:
                    modelos_disponibles.append("Red Neuronal (TensorFlow/Keras Clasificador)")
            else:
                modelos_disponibles = [
                    "Regresión Lineal Penalizada (Ridge)",
                    "Random Forest (Regresor)",
                    "HistGradientBoosting (Regresor)",
                    "Red Neuronal (MLP Regressor)",
                    "K-Vecinos Cercanos (KNN Regresor)",
                    "Árbol de Decisión (Regresor)"
                ]
                if TF_AVAILABLE:
                    modelos_disponibles.append("Red Neuronal (TensorFlow/Keras Regresor)")

            col_mod1, col_mod2 = st.columns([1.3, 1.7])
            with col_mod1:
                tipo_modelo = st.selectbox("🤖 Selecciona el Algoritmo:", modelos_disponibles)
                st.info(MODEL_HELP.get(tipo_modelo, ""))
                split_train = st.slider("Porcentaje para Entrenamiento (%):", 50, 90, 80, 5, help="80% entrenamiento y 20% testeo con random_state=42 para reproducibilidad idéntica a VS Code.")

            with col_mod2:
                st.markdown("#### 🎛️ Hiperparámetros y Ayuda Contextual")
                params_info = {}

                if "MLP" in tipo_modelo:
                    params_info["neurons"] = st.slider("Neuronas en Capa Oculta:", 4, 64, 16, 4)
                    st.caption("💡 *Neuronas:* Mayor cantidad permite aprender patrones complejos, pero requiere regularización para no sobreajustar.")

                    params_info["activation"] = st.selectbox("Función de Activación:", ["relu", "logistic", "tanh", "identity"])
                    st.info("""📌 **Ayuda de Activación:**\n• **logistic (Sigmoide):** Produce curvas suaves entre 0 y 1. Ideal para estimar probabilidades.\n• **tanh:** Salidas centradas entre -1 y 1.\n• **relu:** Rápida y estándar en redes profundas.\n• **identity:** Transformación lineal directa.""")

                    params_info["solver"] = st.selectbox("Optimizador (Solver):", ["adam", "lbfgs", "sgd"])
                    st.info("""📌 **Ayuda de Optimizador:**\n• **lbfgs:** Método matemático cuasi-Newton. Converge muy rápido y con alta exactitud en tablas medianas (< 10.000 filas).\n• **adam:** Descenso adaptativo por lotes, estándar para grandes volúmenes.\n• **sgd:** Descenso de gradiente clásico.""")

                    params_info["alpha_reg"] = st.slider("Regularización L2 (alpha):", 0.0001, 0.1, 0.0001, 0.0005, format="%.4f")
                    params_info["epochs"] = st.slider("Épocas / Iteraciones Máximas:", 20, 300, 100, 10)
                    params_info["lr"] = st.slider("Tasa de Aprendizaje (Learning Rate):", 0.001, 0.050, 0.010, 0.001, format="%.3f")

                elif "HistGradientBoosting" in tipo_modelo:
                    params_info["learning_rate"] = st.slider("Learning Rate:", 0.01, 0.30, 0.10, 0.01)
                    params_info["max_iter"] = st.slider("Cantidad de Árboles (max_iter):", 20, 200, 100, 10)
                    params_info["max_depth"] = st.slider("Profundidad Máxima:", 2, 15, 6, 1)

                elif "Random Forest" in tipo_modelo:
                    params_info["n_estimators"] = st.slider("Cantidad de Árboles (n_estimators):", 10, 200, 100, 10)
                    params_info["max_depth"] = st.slider("Profundidad Máxima:", 2, 20, 6, 1)

                elif "Logística" in tipo_modelo:
                    params_info["C"] = st.slider("Fuerza de Regularización Inversa (C):", 0.01, 10.0, 1.0, 0.05, help="Valores menores aumentan la penalización; valores mayores permiten mayor ajuste.")

                elif "Ridge" in tipo_modelo:
                    params_info["alpha"] = st.slider("Penalización L2 (Alpha):", 0.01, 20.0, 1.0, 0.1)

                elif "KNN" in tipo_modelo:
                    params_info["K"] = st.slider("Cantidad de Vecinos (K):", 1, 25, 5, 2)

                elif "Árbol de Decisión" in tipo_modelo:
                    params_info["max_depth"] = st.slider("Profundidad Máxima:", 2, 20, 5, 1)

                elif "TensorFlow" in tipo_modelo:
                    params_info["neurons"] = st.slider("Neuronas Capa Oculta:", 4, 64, 16, 4)
                    params_info["epochs"] = st.slider("Épocas de Entrenamiento:", 20, 150, 80, 10)
                    params_info["lr"] = st.slider("Learning Rate (Adam):", 0.001, 0.050, 0.010, 0.001, format="%.3f")

            def construir_preprocesador(df_origen, fts):
                num_cols_sel = [c for c in fts if pd.api.types.is_numeric_dtype(df_origen[c])]
                cat_cols_sel = [c for c in fts if c not in num_cols_sel]
                transf = []
                if num_cols_sel:
                    transf.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_sel))
                if cat_cols_sel:
                    transf.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_sel))
                return ColumnTransformer(transformers=transf)

            if st.button("🚀 Entrenar Modelo Manual", type="primary"):
                with st.spinner(f"Entrenando {tipo_modelo} de forma reproducible (random_state=42)..."):
                    X = df_clean[selected_features]
                    
                    if es_clasificacion:
                        le = LabelEncoder()
                        y = le.fit_transform(df_clean[target_col].astype(str))
                        min_cnt = pd.Series(y).value_counts().min()
                        strat_split = y if min_cnt >= 2 else None
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=(100 - split_train)/100, random_state=42, stratify=strat_split)
                    else:
                        y = df_clean[target_col].values.astype(float)
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=(100 - split_train)/100, random_state=42)

                    prep = construir_preprocesador(df_clean, selected_features)
                    X_tr_p = prep.fit_transform(X_tr)
                    X_te_p = prep.transform(X_te)

                    if tipo_modelo == "Regresión Logística":
                        model = LogisticRegression(C=params_info.get("C", 1.0), max_iter=500, random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "Random Forest (Clasificador)":
                        model = RandomForestClassifier(n_estimators=params_info.get("n_estimators", 100), max_depth=params_info.get("max_depth", 6), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "HistGradientBoosting (Clasificador)":
                        model = HistGradientBoostingClassifier(learning_rate=params_info.get("learning_rate", 0.1), max_iter=params_info.get("max_iter", 100), max_depth=params_info.get("max_depth", 6), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "Red Neuronal (MLP Classifier)":
                        model = MLPClassifier(hidden_layer_sizes=(params_info.get("neurons", 16),), activation=params_info.get("activation", "relu"), solver=params_info.get("solver", "adam"), alpha=params_info.get("alpha_reg", 0.0001), max_iter=params_info.get("epochs", 100), learning_rate_init=params_info.get("lr", 0.01), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "K-Vecinos Cercanos (KNN Clasificador)":
                        model = KNeighborsClassifier(n_neighbors=params_info.get("K", 5))
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "Árbol de Decisión (Clasificador)":
                        model = DecisionTreeClassifier(max_depth=params_info.get("max_depth", 5), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict_proba(X_te_p)[:, 1] if es_binaria else model.predict(X_te_p)
                    elif tipo_modelo == "Red Neuronal (TensorFlow/Keras Clasificador)":
                        tf.random.set_seed(42)
                        model = keras.Sequential([layers.Input(shape=(X_tr_p.shape[1],)), layers.Dense(params_info.get("neurons", 16), activation='relu'), layers.Dense(1, activation='sigmoid')])
                        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params_info.get("lr", 0.01)), loss='binary_crossentropy')
                        model.fit(X_tr_p, y_tr, epochs=params_info.get("epochs", 80), batch_size=32, verbose=0)
                        preds_out = model.predict(X_te_p, verbose=0).ravel()
                    elif tipo_modelo == "Regresión Lineal Penalizada (Ridge)":
                        model = Ridge(alpha=params_info.get("alpha", 1.0), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "Random Forest (Regresor)":
                        model = RandomForestRegressor(n_estimators=params_info.get("n_estimators", 100), max_depth=params_info.get("max_depth", 6), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "HistGradientBoosting (Regresor)":
                        model = HistGradientBoostingRegressor(learning_rate=params_info.get("learning_rate", 0.1), max_iter=params_info.get("max_iter", 100), max_depth=params_info.get("max_depth", 6), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "Red Neuronal (MLP Regressor)":
                        model = MLPRegressor(hidden_layer_sizes=(params_info.get("neurons", 16),), activation=params_info.get("activation", "relu"), solver=params_info.get("solver", "adam"), alpha=params_info.get("alpha_reg", 0.0001), max_iter=params_info.get("epochs", 100), learning_rate_init=params_info.get("lr", 0.01), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "K-Vecinos Cercanos (KNN Regresor)":
                        model = KNeighborsRegressor(n_neighbors=params_info.get("K", 5))
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "Árbol de Decisión (Regresor)":
                        model = DecisionTreeRegressor(max_depth=params_info.get("max_depth", 5), random_state=42)
                        model.fit(X_tr_p, y_tr)
                        preds_out = model.predict(X_te_p)
                    elif tipo_modelo == "Red Neuronal (TensorFlow/Keras Regresor)":
                        tf.random.set_seed(42)
                        model = keras.Sequential([layers.Input(shape=(X_tr_p.shape[1],)), layers.Dense(params_info.get("neurons", 16), activation='relu'), layers.Dense(1)])
                        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params_info.get("lr", 0.01)), loss='mse')
                        model.fit(X_tr_p, y_tr, epochs=params_info.get("epochs", 80), batch_size=32, verbose=0)
                        preds_out = model.predict(X_te_p, verbose=0).ravel()

                    st.session_state["trained_model"] = model
                    st.session_state["preprocessor"] = prep
                    st.session_state["test_outputs"] = preds_out
                    st.session_state["y_test"] = y_te
                    st.session_state["last_params"] = {
                        "modelo": tipo_modelo,
                        "target": target_col,
                        "features": selected_features,
                        "config": str(params_info)
                    }
                    st.success(f"✅ ¡{tipo_modelo} entrenado exitosamente! Ve a la pestaña '4️⃣ Evaluación y Veredicto LLM'.")

            st.divider()

            st.subheader("⚡ Torneo AutoML con Búsqueda Automática (Optuna)")
            if not OPTUNA_AVAILABLE:
                st.warning("Optuna no está disponible en este entorno. Puedes entrenar manualmente los modelos arriba o instalarlo con `pip install optuna`.")
            else:
                c_au1, c_au2, c_au3 = st.columns(3)
                with c_au1:
                    n_trials = st.slider("Cantidad de Pruebas Automáticas:", 5, 50, 15, 5)
                with c_au2:
                    if es_clasificacion:
                        metrica_opt = st.selectbox("Métrica a Optimizar:", ["Recall", "F1-Score", "Precisión", "Accuracy"])
                    else:
                        metrica_opt = st.selectbox("Métrica a Optimizar:", ["Menor MAE", "Menor RMSE", "Mayor R2"])
                with c_au3:
                    if es_clasificacion and es_binaria:
                        umbral_fijo_auto = st.slider("Umbral Fijo para Evaluar Torneo:", 0.10, 0.60, 0.30, 0.05, help="Fijar un umbral uniforme evita que el optimizador elija umbrales irreales solo para inflar el Recall.")
                    else:
                        umbral_fijo_auto = 0.50

                if st.button("⚡ Ejecutar Torneo AutoML"):
                    with st.spinner(f"Optimizando {n_trials} ensayos con validación cruzada..."):
                        X_a = df_clean[selected_features]
                        prep_a = construir_preprocesador(df_clean, selected_features)
                        X_a_p = prep_a.fit_transform(X_a)
                        trials_records = []

                        if es_clasificacion:
                            le_a = LabelEncoder()
                            y_a = le_a.fit_transform(df_clean[target_col].astype(str))
                            avg_mode = "binary" if es_binaria else "macro"

                            def obj_clf(trial):
                                m_name = trial.suggest_categorical("model", ["LogisticRegression", "RandomForest", "HistGradientBoosting", "MLP", "KNN"])
                                if m_name == "LogisticRegression":
                                    c_val = trial.suggest_float("C", 0.01, 10.0, log=True)
                                    clf = LogisticRegression(C=c_val, max_iter=300, random_state=42)
                                    p_str = f"C={c_val:.3f}"
                                elif m_name == "RandomForest":
                                    n_e = trial.suggest_int("n_estimators", 20, 100, step=20)
                                    d_val = trial.suggest_int("max_depth", 3, 10)
                                    clf = RandomForestClassifier(n_estimators=n_e, max_depth=d_val, random_state=42)
                                    p_str = f"trees={n_e}, depth={d_val}"
                                elif m_name == "HistGradientBoosting":
                                    lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
                                    clf = HistGradientBoostingClassifier(learning_rate=lr, random_state=42)
                                    p_str = f"lr={lr:.3f}"
                                elif m_name == "MLP":
                                    neu = trial.suggest_int("neurons", 8, 32, step=8)
                                    act = trial.suggest_categorical("activation", ["relu", "logistic", "tanh"])
                                    clf = MLPClassifier(hidden_layer_sizes=(neu,), activation=act, max_iter=80, random_state=42)
                                    p_str = f"neurons={neu}, act={act}"
                                else:
                                    k_v = trial.suggest_int("k", 3, 15, step=2)
                                    clf = KNeighborsClassifier(n_neighbors=k_v)
                                    p_str = f"k={k_v}"

                                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                                recs, precs, f1s, accs = [], [], [], []
                                for tr_i, val_i in skf.split(X_a_p, y_a):
                                    clf.fit(X_a_p[tr_i], y_a[tr_i])
                                    if es_binaria:
                                        probs = clf.predict_proba(X_a_p[val_i])[:, 1]
                                        pr_c = (probs >= umbral_fijo_auto).astype(int)
                                    else:
                                        pr_c = clf.predict(X_a_p[val_i])
                                    recs.append(recall_score(y_a[val_i], pr_c, average=avg_mode, zero_division=0))
                                    precs.append(precision_score(y_a[val_i], pr_c, average=avg_mode, zero_division=0))
                                    f1s.append(f1_score(y_a[val_i], pr_c, average=avg_mode, zero_division=0))
                                    accs.append(accuracy_score(y_a[val_i], pr_c))

                                mr, mp, mf, ma = float(np.mean(recs)), float(np.mean(precs)), float(np.mean(f1s)), float(np.mean(accs))
                                score = mr if metrica_opt == "Recall" else (mp if metrica_opt == "Precisión" else (ma if metrica_opt == "Accuracy" else mf))
                                
                                trials_records.append({
                                    "#": len(st.session_state.historial_clf) + len(trials_records) + 1,
                                    "Modelo": f"[AutoML] {m_name}",
                                    "Umbral": umbral_fijo_auto if es_binaria else "N/A",
                                    "Recall": f"{mr:.1%}",
                                    "Precisión": f"{mp:.1%}",
                                    "F1": round(mf, 3),
                                    "Accuracy": f"{ma:.1%}",
                                    "Hiperparámetros": p_str,
                                    "Predictoras": ", ".join(selected_features),
                                    "_score": score
                                })
                                return score

                            study = optuna.create_study(direction="maximize")
                            study.optimize(obj_clf, n_trials=n_trials)
                            df_tr_res = pd.DataFrame(trials_records).sort_values("_score", ascending=False).drop(columns=["_score"])
                            st.dataframe(df_tr_res, use_container_width=True)
                            for r in trials_records:
                                r_copy = {k: v for k, v in r.items() if k != "_score"}
                                st.session_state.historial_clf.append(r_copy)
                            st.success(f"🏆 Torneo AutoML finalizado. Se agregaron {len(trials_records)} modelos al historial de Clasificación.")

                        else:
                            y_a = df_clean[target_col].values.astype(float)
                            direction = "maximize" if "R2" in metrica_opt else "minimize"

                            def obj_reg(trial):
                                m_name = trial.suggest_categorical("model", ["Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor", "MLPRegressor", "KNNRegressor"])
                                if m_name == "Ridge":
                                    alp = trial.suggest_float("alpha", 0.05, 20.0, log=True)
                                    reg = Ridge(alpha=alp, random_state=42)
                                    p_str = f"alpha={alp:.2f}"
                                elif m_name == "RandomForestRegressor":
                                    n_e = trial.suggest_int("n_estimators", 20, 100, step=20)
                                    d_val = trial.suggest_int("max_depth", 3, 10)
                                    reg = RandomForestRegressor(n_estimators=n_e, max_depth=d_val, random_state=42)
                                    p_str = f"trees={n_e}, depth={d_val}"
                                elif m_name == "HistGradientBoostingRegressor":
                                    lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
                                    reg = HistGradientBoostingRegressor(learning_rate=lr, random_state=42)
                                    p_str = f"lr={lr:.3f}"
                                elif m_name == "MLPRegressor":
                                    neu = trial.suggest_int("neurons", 8, 32, step=8)
                                    reg = MLPRegressor(hidden_layer_sizes=(neu,), max_iter=80, random_state=42)
                                    p_str = f"neurons={neu}"
                                else:
                                    k_v = trial.suggest_int("k", 3, 15, step=2)
                                    reg = KNeighborsRegressor(n_neighbors=k_v)
                                    p_str = f"k={k_v}"

                                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                                maes, rmses, r2s = [], [], []
                                for tr_i, val_i in kf.split(X_a_p):
                                    reg.fit(X_a_p[tr_i], y_a[tr_i])
                                    pr = reg.predict(X_a_p[val_i])
                                    maes.append(mean_absolute_error(y_a[val_i], pr))
                                    rmses.append(np.sqrt(mean_squared_error(y_a[val_i], pr)))
                                    r2s.append(r2_score(y_a[val_i], pr))

                                mmae, mrmse, mr2 = float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(r2s))
                                score = mmae if "MAE" in metrica_opt else (mrmse if "RMSE" in metrica_opt else mr2)
                                
                                trials_records.append({
                                    "#": len(st.session_state.historial_reg) + len(trials_records) + 1,
                                    "Modelo": f"[AutoML] {m_name}",
                                    "MAE": round(mmae, 3),
                                    "RMSE": round(mrmse, 3),
                                    "R2": round(mr2, 3),
                                    "Hiperparámetros": p_str,
                                    "Predictoras": ", ".join(selected_features),
                                    "_score": score
                                })
                                return score

                            study = optuna.create_study(direction=direction)
                            study.optimize(obj_reg, n_trials=n_trials)
                            asc_sort = False if direction == "maximize" else True
                            df_tr_res = pd.DataFrame(trials_records).sort_values("_score", ascending=asc_sort).drop(columns=["_score"])
                            st.dataframe(df_tr_res, use_container_width=True)
                            for r in trials_records:
                                r_copy = {k: v for k, v in r.items() if k != "_score"}
                                st.session_state.historial_reg.append(r_copy)
                            st.success(f"🏆 Torneo AutoML finalizado. Se agregaron {len(trials_records)} modelos al historial de Regresión.")

# ==============================================================================
# PASO 4: EVALUACIÓN Y VEREDICTO LLM
# ==============================================================================
with tabs[3]:
    st.header("4️⃣ Evaluación de Resultados, Gráficos Interactivos y Veredicto LLM")

    if df is None:
        st.warning("👈 Primero debes cargar un archivo CSV en la pestaña '1️⃣ Carga y Limpieza'.")
    elif "test_outputs" not in st.session_state or st.session_state["test_outputs"] is None:
        st.info("👈 Entrena primero un modelo en la pestaña '3️⃣ Entrenamiento y AutoML' para visualizar sus métricas interactivas.")
    else:
        es_clasificacion = st.session_state.get("es_clasificacion", True)
        es_binaria = st.session_state.get("es_binaria", True)
        y_test = st.session_state["y_test"]
        outputs = st.session_state["test_outputs"]
        info_mod = st.session_state.get("last_params", {})

        if es_clasificacion:
            st.subheader(f"📊 Evaluación de Clasificación: `{info_mod.get('modelo')}`")
            if es_binaria:
                c_th1, c_th2 = st.columns([2, 1])
                with c_th1:
                    threshold = st.slider("🎚️ Calibrar Umbral de Decisión en Vivo (Sin reentrenar):", 0.05, 0.95, 0.30, 0.05)
                with c_th2:
                    st.caption("Ajusta el corte de probabilidad para equilibrar Recall y Precisión.")
                y_pred = (outputs >= threshold).astype(int)
                avg_m = "binary"
            else:
                threshold = "Multiclase"
                y_pred = outputs.astype(int)
                avg_m = "macro"

            rec = recall_score(y_test, y_pred, average=avg_m, zero_division=0)
            prec = precision_score(y_test, y_pred, average=avg_m, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_m, zero_division=0)
            acc = accuracy_score(y_test, y_pred)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Recall (Sensibilidad)", f"{rec:.1%}")
            k2.metric("Precisión", f"{prec:.1%}")
            k3.metric("F1-Score", f"{f1:.3f}")
            k4.metric("Exactitud (Accuracy)", f"{acc:.1%}")

            col_g_clf1, col_g_clf2 = st.columns(2)
            with col_g_clf1:
                cm = confusion_matrix(y_test, y_pred)
                labels_c = st.session_state.get("classes_labels", [str(i) for i in range(cm.shape[0])])
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Predicción del Modelo", y="Clase Real", color="Casos"),
                    x=[f"Pred: {l}" for l in labels_c],
                    y=[f"Real: {l}" for l in labels_c],
                    color_continuous_scale="Blues",
                    title="Matriz de Confusión Interactiva (Plotly)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_g_clf2:
                if es_binaria:
                    p_c, r_c, t_c = precision_recall_curve(y_test, outputs)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=t_c, y=r_c[:-1], mode='lines', name='Recall', line=dict(color='#38bdf8', width=3)))
                    fig_pr.add_trace(go.Scatter(x=t_c, y=p_c[:-1], mode='lines', name='Precisión', line=dict(color='#f59e0b', width=3)))
                    fig_pr.add_vline(x=threshold, line_dash="dot", line_color="#10b981", annotation_text=f"Umbral ({threshold:.2f})")
                    fig_pr.update_layout(
                        title="Curva Interactiva Recall vs. Precisión",
                        xaxis_title="Umbral de Probabilidad",
                        yaxis_title="Valor de la Métrica",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

            if st.button("💾 Guardar Experimento en Historial de Clasificación", type="primary"):
                st.session_state.historial_clf.append({
                    "#": len(st.session_state.historial_clf) + 1,
                    "Modelo": info_mod.get("modelo"),
                    "Umbral": threshold,
                    "Recall": f"{rec:.1%}",
                    "Precisión": f"{prec:.1%}",
                    "F1": round(f1, 3),
                    "Accuracy": f"{acc:.1%}",
                    "Hiperparámetros": info_mod.get("config"),
                    "Predictoras": ", ".join(info_mod.get("features", []))
                })
                st.success("Guardado en el historial de Clasificación.")

        else:
            st.subheader(f"📈 Evaluación de Regresión: `{info_mod.get('modelo')}`")
            mae = mean_absolute_error(y_test, outputs)
            rmse = float(np.sqrt(mean_squared_error(y_test, outputs)))
            r2 = r2_score(y_test, outputs)
            mape = mean_absolute_percentage_error(y_test, outputs)

            r_k1, r_k2, r_k3, r_k4 = st.columns(4)
            r_k1.metric("MAE (Error Absoluto)", f"{mae:,.2f}")
            r_k2.metric("RMSE (Error Cuadrático)", f"{rmse:,.2f}")
            r_k3.metric("R² (Ajuste)", f"{r2:.3f}")
            r_k4.metric("MAPE (Porcentual)", f"{mape:.1%}")

            col_g_reg1, col_g_reg2 = st.columns(2)
            with col_g_reg1:
                df_pred_vs_real = pd.DataFrame({"Valor Real": y_test, "Valor Estimado": outputs, "Error Absoluto": np.abs(y_test - outputs)})
                fig_sc = px.scatter(
                    df_pred_vs_real, x="Valor Real", y="Valor Estimado", color="Error Absoluto",
                    title="Valores Reales vs. Valores Estimados",
                    color_continuous_scale="Viridis", template="plotly_dark"
                )
                min_v, max_v = float(min(y_test.min(), outputs.min())), float(max(y_test.max(), outputs.max()))
                fig_sc.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal", line=dict(dash="dash", color="red")))
                st.plotly_chart(fig_sc, use_container_width=True)

            with col_g_reg2:
                residuos = y_test - outputs
                fig_res = px.histogram(
                    x=residuos, nbins=30,
                    labels={"x": "Error Residual (Real - Estimado)"},
                    title="Distribución de Errores Residuales",
                    color_discrete_sequence=["#10b981"], template="plotly_dark"
                )
                fig_res.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

            if st.button("💾 Guardar Experimento en Historial de Regresión", type="primary"):
                st.session_state.historial_reg.append({
                    "#": len(st.session_state.historial_reg) + 1,
                    "Modelo": info_mod.get("modelo"),
                    "MAE": round(mae, 3),
                    "RMSE": round(rmse, 3),
                    "R2": round(r2, 3),
                    "Hiperparámetros": info_mod.get("config"),
                    "Predictoras": ", ".join(info_mod.get("features", []))
                })
                st.success("Guardado en el historial de Regresión.")

        st.divider()

        historial_activo = st.session_state.historial_clf if es_clasificacion else st.session_state.historial_reg
        st.subheader(f"📋 Historial Comparativo de {'Clasificación' if es_clasificacion else 'Regresión'}")
        if historial_activo:
            st.dataframe(pd.DataFrame(historial_activo), use_container_width=True)
            if st.button("🗑️ Limpiar Historial Activo"):
                if es_clasificacion:
                    st.session_state.historial_clf = []
                else:
                    st.session_state.historial_reg = []
                st.rerun()
        else:
            st.caption("Aún no hay experimentos guardados en el historial.")

        st.divider()

        st.subheader("🦙 Veredicto Técnico con LLM Local (Ollama)")
        c_ll1, c_ll2, c_ll3 = st.columns([1.5, 1.5, 1])
        with c_ll1:
            modelos_ollama = [
                "deepseek-r1:1.5b", "qwen3:4b", "gemma3:4b", "qwen2.5:7b",
                "llama3.1:8b", "petronetto/mimo-v2.6-9b", "phi4-mini", "qwen3:30b"
            ]
            modelo_ollama = st.selectbox("Modelo de Ollama:", modelos_ollama)
        with c_ll2:
            ollama_url = st.text_input("URL Servidor Ollama:", value="http://localhost:11434")
        with c_ll3:
            max_tokens_llm = st.slider("Máximo de Tokens:", 150, 500, 350, 50)

        if st.button("🧠 Emitir Veredicto sobre el Historial Activo", type="primary"):
            if not historial_activo:
                st.warning("Guarda al menos un experimento en el historial o corre AutoML antes de consultar al LLM.")
            else:
                with st.spinner(f"Analizando historial con {modelo_ollama}..."):
                    tabla_str = pd.DataFrame(historial_activo).to_string(index=False)
                    tarea_txt = "Clasificación" if es_clasificacion else "Regresión"
                    target_col = st.session_state.get("target_col", "Target")
                    prompt_llm = f"""Eres un especialista en Ciencia de Datos.
Tipo de problema: {tarea_txt} sobre la variable objetivo '{target_col}'.
Tabla de experimentos:
{tabla_str}

Instrucciones:
1. Compara los modelos de la tabla.
2. Declara cuál es el MODELO GANADOR citando sus métricas exactas.
3. Evalúa si es adecuado para tomar decisiones operativas.
Responde directo en español en menos de {max_tokens_llm} tokens."""
                    try:
                        resp = requests.post(
                            f"{ollama_url.rstrip('/')}/api/generate",
                            json={"model": modelo_ollama, "prompt": prompt_llm, "stream": False, "options": {"num_predict": max_tokens_llm, "temperature": 0.2}},
                            timeout=180
                        )
                        if resp.status_code == 200:
                            st.session_state.veredicto_llm = resp.json().get("response", "")
                        else:
                            st.error(f"Error HTTP {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"No se pudo conectar con Ollama en `{ollama_url}`: {e}")

        if st.session_state.veredicto_llm:
            st.info(f"**Veredicto de {modelo_ollama}:**\n\n{st.session_state.veredicto_llm}")

# ==============================================================================
# PASO 5: PRODUCCIÓN
# ==============================================================================
with tabs[4]:
    st.header("5️⃣ Poner en Producción: Evaluar Nuevo Caso en Vivo")
    
    if df is None:
        st.warning("👈 Primero debes cargar un archivo CSV en la pestaña '1️⃣ Carga y Limpieza'.")
    elif "trained_model" not in st.session_state or st.session_state["trained_model"] is None:
        st.info("👈 Entrena primero un modelo en la pestaña '3️⃣ Entrenamiento y AutoML' para habilitar el simulador de producción.")
    else:
        df_prod_active = st.session_state.get("df_clean", df)
        feats = st.session_state.get("selected_features", [])
        es_clf = st.session_state.get("es_clasificacion", True)
        info_m = st.session_state.get("last_params", {})

        st.markdown(f"**Modelo en Producción:** `{info_m.get('modelo')}` | **Target:** `{info_m.get('target')}`")

        with st.form("form_prod_viva"):
            vals = {}
            cols_f = st.columns(min(3, max(1, len(feats))))
            for i, fn in enumerate(feats):
                c_ui = cols_f[i % len(cols_f)]
                if pd.api.types.is_numeric_dtype(df_prod_active[fn]):
                    med_v = float(df_prod_active[fn].median())
                    vals[fn] = c_ui.number_input(f"{fn}:", value=med_v)
                else:
                    opts = list(df_prod_active[fn].dropna().unique())
                    vals[fn] = c_ui.selectbox(f"{fn}:", options=opts)
            sub = st.form_submit_button("🔮 Generar Predicción")

        if sub:
            df_n = pd.DataFrame([vals])
            X_np = st.session_state["preprocessor"].transform(df_n)
            mod = st.session_state["trained_model"]
            if es_clf:
                p = float(mod.predict_proba(X_np)[0, 1]) if hasattr(mod, "predict_proba") else float(mod.predict(X_np)[0])
                st.metric("Probabilidad Estimada (Clase Positiva)", f"{p:.1%}")
                if p >= 0.30:
                    st.error("⚠️ **ALERTA PREVENTIVA:** El caso supera el umbral de decisión (0.30).")
                else:
                    st.success("✅ **BAJO RIESGO / NORMAL:** El caso se encuentra dentro de los parámetros normales.")
            else:
                v = float(np.ravel(mod.predict(X_np))[0])
                st.metric("Valor Estimado", f"{v:,.2f}")

# ==============================================================================
# PASO 6: GUÍA CONCEPTUAL
# ==============================================================================
with tabs[5]:
    st.header("📘 Guía Práctica: Clasificación vs. Regresión y Catálogo de Algoritmos")

    col_guia1, col_guia2 = st.columns(2)
    with col_guia1:
        st.subheader("🏷️ 1. Problemas de Clasificación")
        st.markdown("""
        - **¿Cuándo se usa?** Cuando la etiqueta objetivo es una categoría o estado discreto (ej.: `Aprobado / Rechazado`, `Positivo / Negativo`, `Tipo de Cliente`).
        - **¿Cómo predice?** Calcula probabilidades continuas entre `0.0` y `1.0` y aplica un **Umbral de Decisión** para definir la clase.
        
        #### Métricas Exclusivas de Clasificación:
        - **Recall (Sensibilidad):** Porcentaje de casos positivos reales capturados. Vital cuando el costo de omitir un positivo (falso negativo) es alto.
        - **Precisión:** De las alertas emitidas por el modelo, cuántas eran reales.
        - **F1-Score:** Balance armónico entre Recall y Precisión.
        - **Matriz de Confusión:** Desglose de Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Verdaderos Negativos.
        """)

    with col_guia2:
        st.subheader("📈 2. Problemas de Regresión (Estimar un Valor)")
        st.markdown("""
        - **¿Cuándo se usa?** Cuando la etiqueta objetivo es una cantidad numérica continua (ej.: `precios`, `ventas`, `antigüedad`).
        - **¿Cómo predice?** Estima magnitudes continuas directas (no existen matrices de confusión ni umbrales de probabilidad).
        
        #### Métricas Exclusivas de Regresión:
        - **MAE (Error Absoluto Medio):** Promedio de desvío en las mismas unidades de la variable.
        - **RMSE (Raíz del Error Cuadrático Medio):** Penaliza fuertemente desvíos grandes.
        - **R² (Coeficiente de Determinación):** Proporción de la varianza explicada por el modelo (cercano a 1 es óptimo).
        """)

    st.divider()
    st.subheader("🤖 Catálogo de Algoritmos")
    df_catalogo = pd.DataFrame([
        {"Familia": "Modelos Lineales", "Clasificación": "Regresión Logística", "Regresión": "Regresión Ridge (L2)", "Fortaleza": "Velocidad, simplicidad y alta interpretabilidad."},
        {"Familia": "Basados en Vecindad", "Clasificación": "KNN Clasificador", "Regresión": "KNN Regresor", "Fortaleza": "Predice por similitud directa sin asumir linealidad."},
        {"Familia": "Árboles Simples", "Clasificación": "Árbol de Decisión", "Regresión": "Árbol de Decisión Regresor", "Fortaleza": "Reglas explicables transparentes."},
        {"Familia": "Ensambles Bagging", "Clasificación": "Random Forest", "Regresión": "Random Forest Regressor", "Fortaleza": "Muy resistente a sobreajuste, alta precisión en tablas."},
        {"Familia": "Ensambles Boosting", "Clasificación": "HistGradientBoosting", "Regresión": "HistGradientBoosting Regressor", "Fortaleza": "Velocidad extrema y precisión competitiva con LightGBM."},
        {"Familia": "Redes Neuronales", "Clasificación": "MLPClassifier / Keras", "Regresión": "MLPRegressor / Keras", "Fortaleza": "Modelado de relaciones no lineales profundas."}
    ])
    st.table(df_catalogo)