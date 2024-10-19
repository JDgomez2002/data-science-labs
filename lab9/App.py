import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
from plotly import graph_objs as go

color_palette = {
    'primary': '#FFFFFF',  # Corregido de '#FFFFF' a '#FFFFFF'
    'secondary': '#E5E5E5',
    'accent1': '#CDECAC',
    'accent2': '#2D9494',
    'accent3': '#CC5A49'
}

# Configuración de la página
st.set_page_config(layout="wide", page_title="Dashboard de Datos de Pacientes")

# Estilo CSS personalizado
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: """ + color_palette['accent2'] + """;
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        color: #FFFFFF;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {{
    color: #FFFFFF !important;
    font-weight: bold;
}}
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 3rem;
    }}
    h1, h2, h3, h4, h5, h6, p {{
        color: #CDECAC;
    }}
    .stButton>button {{
        color: {color_palette['primary']};
        background-color: {color_palette['accent2']};
        border-radius: 5px;
    }}
    .stSelectbox {{
        color: {color_palette['accent2']};
    }}
    </style>
    """, unsafe_allow_html=True)

# Título de la aplicación
st.title("Cuadro de Mando Interactivo - Datos de Pacientes del Hospital Universitario de Caracas")

# Cargar datos
@st.cache_data
def load_data():
    data = pd.read_csv('risk_factors_cervical_cancer.csv')
    return data

data = load_data()

# Reemplazar valores '?' por 0s en el conjunto de datos y convertir a numérico
data = data.replace('?', 0)
data = data.astype(float)

# Mostrar los datos en la aplicación
st.subheader("Muestra del conjunto de datos")
st.dataframe(data.head(), use_container_width=True)

# Sidebar para la selección de variables y configuración de gráficos
st.sidebar.title("Opciones de visualización")
selected_variable = st.sidebar.selectbox("Seleccione una variable para analizar", data.columns)

# Visualización básica de la variable seleccionada
st.subheader(f"Distribución de {selected_variable}")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data[selected_variable], kde=True, ax=ax, color=color_palette['accent2'])
ax.set_facecolor(color_palette['secondary'])
fig.patch.set_facecolor(color_palette['primary'])
st.pyplot(fig)

# División de datos para modelos predictivos
st.sidebar.subheader("Selección de modelo predictivo")
target_variable = 'Dx:Cancer'
X = data[['Age', 'Number of sexual partners', 'Smokes', 'STDs', 'STDs:HPV']]
y = data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Selección del modelo a entrenar
model_choice = st.sidebar.selectbox("Seleccione el modelo", ("Random Forest", "Regresión Logística", "K-Nearest Neighbors"))

# Entrenamiento y evaluación del modelo
if model_choice == 'Random Forest':
    model = RandomForestClassifier()
elif model_choice == 'Regresión Logística':
    model = LogisticRegression()
else:
    model = KNeighborsClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mostrar la matriz de confusión de forma interactiva
st.subheader(f"Matriz de Confusión Interactiva - {model_choice}")
# Mostrar la exactitud del modelo seleccionado
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Exactitud del modelo {model_choice}: {accuracy:.6f}")
conf_matrix = confusion_matrix(y_test, y_pred)

fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicción Negativa', 'Predicción Positiva'],
                y=['Valor Real Negativo', 'Valor Real Positivo'],
                colorscale=[color_palette['primary'], color_palette['accent1'], color_palette['accent2'], color_palette['accent3']],
                text=conf_matrix,
                texttemplate='%{text}',
                textfont={"size":20},
                hoverongaps=False))

fig.update_layout(
    title=f'Matriz de Confusión - {model_choice}',
    xaxis_title='Predicción',
    yaxis_title='Valor Real',
    height=500,
    width=700
)

st.plotly_chart(fig, use_container_width=True)

# Explicación de la matriz de confusión
st.write("""
Esta matriz de confusión muestra el rendimiento del modelo:
- Verdaderos Negativos (arriba izquierda): Casos negativos correctamente identificados
- Falsos Positivos (arriba derecha): Casos negativos incorrectamente identificados como positivos
- Falsos Negativos (abajo izquierda): Casos positivos incorrectamente identificados como negativos
- Verdaderos Positivos (abajo derecha): Casos positivos correctamente identificados

Un modelo perfecto tendría solo valores en la diagonal principal (arriba izquierda y abajo derecha).
""")

# Gráficos interactivos con Plotly
st.subheader("Visualización interactiva")
visual_type = st.selectbox("Seleccione el tipo de visualización", (
  "Box Plot",
  "Scatter Plot",
  ))

if visual_type == "Box Plot":
    fig = px.box(data, x='STDs', y='Age', color='Dx:Cancer',
                 labels={'Dx:Cancer': 'Diagnóstico de Cáncer'},
                 color_discrete_sequence=[color_palette['accent1'], color_palette['accent3']])
    st.plotly_chart(fig, use_container_width=True)
elif visual_type == "Scatter Plot":
    fig = px.scatter(data, x='Age', y='Dx:Cancer', color='Smokes',
                     labels={'Dx:Cancer': 'Diagnóstico de Cáncer'},
                     color_discrete_sequence=[color_palette['accent1'], color_palette['accent3']])
    st.plotly_chart(fig, use_container_width=True)


# Matriz de correlación
st.subheader("Matriz de Correlación Interactiva entre Variables Numéricas")
numeric = data[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
                'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)', 
                'STDs (number)', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis', 
                'STDs: Time since last diagnosis']]

correlation_matrix = numeric.corr()

fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale=[color_palette['primary'], color_palette['accent1'], color_palette['accent2'], color_palette['accent3']],
                zmin=-1, zmax=1,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size":10},
                hoverongaps=False))

fig.update_layout(height=1000, width=800)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Comparación de Modelos")

# Add model selection widget
available_models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

selected_models = st.multiselect(
    "Seleccione los modelos para comparar",
    options=list(available_models.keys()),
    default=list(available_models.keys())
)

if selected_models:
    accuracies = []
    for model_name in selected_models:
        model = available_models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    comparison_data = pd.DataFrame({
        'Model': selected_models,
        'Accuracy': accuracies
    })

    fig = px.bar(comparison_data, x='Model', y='Accuracy',
                 color='Model',
                 color_discrete_map={
                     'Random Forest': color_palette['accent1'],
                     'Logistic Regression': color_palette['accent2'],
                     'K-Nearest Neighbors': color_palette['accent3']
                 },
                 labels={'Accuracy': 'Precisión', 'Model': 'Modelo'})
    
    fig.update_layout(
        xaxis_title="Modelo",
        yaxis_title="Precisión",
        legend_title="Modelos"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Por favor, seleccione al menos un modelo para comparar.")

# Pie de página
st.markdown(f"""
    <div style='background-color: {color_palette['secondary']}; padding: 10px; border-radius: 5px; text-align: center;'>
        <p style='color: {color_palette['accent2']}; margin: 0;'>Aplicación desarrollada por Abner Ivan Garcia - 21285, Jose Daniel Gomez - 21429</p>
    </div>
    """, unsafe_allow_html=True)
