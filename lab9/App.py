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
st.write("Muestra del conjunto de datos", data.head())

st.write(data.columns)

# Sidebar para la selección de variables y configuración de gráficos
st.sidebar.title("Opciones de visualización")
selected_variable = st.sidebar.selectbox("Seleccione una variable para analizar", 
                                         data.columns)

# Visualización básica de la variable seleccionada
st.write(f"Distribución de {selected_variable}")
fig = plt.figure(figsize=(10, 6))
sns.histplot(data[selected_variable], kde=True)
st.pyplot(fig)

# División de datos para modelos predictivos
st.sidebar.subheader("Selección de modelo predictivo")
target_variable = 'Dx:Cancer'  # Variable objetivo: diagnóstico de cáncer
X = data[['Age', 'Number of sexual partners', 'Smokes', 'STDs', 'STDs:HPV']]  # Selección de variables explicativas
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
    model = KNeighborsClassifier()  # KNN agregado

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mostrar la exactitud del modelo seleccionado
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Exactitud del modelo {model_choice}: {accuracy:.2f}")

# Mostrar la matriz de confusión
st.subheader(f"Matriz de Confusión - {model_choice}")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Gráficos interactivos con Plotly
st.subheader("Visualización interactiva")
visual_type = st.selectbox("Seleccione el tipo de visualización", ("Scatter Plot", "Box Plot"))

if visual_type == "Scatter Plot":
    fig = px.scatter(data, x='Age', y='Dx:Cancer', color='Smokes',
                     labels={'Dx:Cancer': 'Diagnóstico de Cáncer'})
    st.plotly_chart(fig)

elif visual_type == "Box Plot":
    fig = px.box(data, x='STDs', y='Age', color='Dx:Cancer',
                 labels={'Dx:Cancer': 'Diagnóstico de Cáncer'})
    st.plotly_chart(fig)

# Gráficos enlazados - Filtrado interactivo de pacientes fumadores
st.subheader("Gráficos enlazados")
selected_smoking_status = st.selectbox("Filtrar por si fuma o no", data['Smokes'].unique())
filtered_data = data[data['Smokes'] == selected_smoking_status]

st.write(f"Datos filtrados para pacientes que {'fuman' if selected_smoking_status else 'no fuman'}")
st.write(filtered_data)

# Gráfico detallado basado en la selección
st.write(f"Visualización detallada para fumadores = {selected_smoking_status}")
fig = plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_data['STDs'], y=filtered_data['Age'])
st.pyplot(fig)

if st.sidebar.checkbox("Mostrar comparación de modelos"):
    model_rf = RandomForestClassifier()
    model_lr = LogisticRegression()
    model_knn = KNeighborsClassifier()

    model_rf.fit(X_train, y_train)
    model_lr.fit(X_train, y_train)
    model_knn.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)
    y_pred_lr = model_lr.predict(X_test)
    y_pred_knn = model_knn.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    comparison_data = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'K-Nearest Neighbors'],
        'Accuracy': [acc_rf, acc_lr, acc_knn]
    })

    st.write(comparison_data)

    # Mostrar las matrices de confusión de los tres modelos
    st.write("Matriz de confusión para Random Forest")
    st.write(confusion_matrix(y_test, y_pred_rf))

    st.write("Matriz de confusión para Regresión Logística")
    st.write(confusion_matrix(y_test, y_pred_lr))

    st.write("Matriz de confusión para K-Nearest Neighbors")
    st.write(confusion_matrix(y_test, y_pred_knn))

# Separar las variables numéricas para analizar la correlación
numeric = data[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
                'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)', 
                'STDs (number)', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis', 
                'STDs: Time since last diagnosis']]

# Convertir las variables a numéricas
for column in numeric.columns:
    numeric[column] = pd.to_numeric(numeric[column].replace('?', np.nan), errors='coerce')

# Reemplazar los valores faltantes con la media de la columna
for column in numeric:
    if column not in ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']:
        numeric[column] = numeric[column].fillna(numeric[column].mean())
    else:
        # Reemplazar los valores faltantes con 0
        numeric[column] = numeric[column].fillna(0).astype(int)

# Matriz de correlación entre variables numéricas    
correlation_matrix = numeric.corr()

# Crear la visualización interactiva de la matriz de correlación
st.subheader("Matriz de Correlación Interactiva entre Variables Numéricas")

fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu_r',  # Usar la escala RdBu invertida
                zmin=-1, zmax=1,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size":10},
                hoverongaps=False))

fig.update_layout(
    # title='Matriz de correlación entre variables numéricas',
    height=600,
    width=800,
)

st.plotly_chart(fig)

# Añadir una explicación
st.write("""
Esta matriz de correlación interactiva muestra la relación entre las diferentes variables numéricas del conjunto de datos. 
Los valores varían entre -1 y 1, donde:
- 1 indica una correlación positiva perfecta (rojo intenso)
- -1 indica una correlación negativa perfecta (azul intenso)
- 0 indica que no hay correlación lineal (blanco)

Colores más intensos indican correlaciones más fuertes, mientras que colores más claros indican correlaciones más débiles.
Puedes interactuar con la gráfica pasando el cursor sobre las celdas para ver los valores exactos.
""")

st.write(correlation_matrix)

# Comparación de modelos predictivos
st.subheader("Comparación de Modelos")

# Añadir pie de página o conclusiones
st.write("Aplicación desarrollada por Abner Ivan Garcia - 21285, Jose Daniel Gomez - 21429")
