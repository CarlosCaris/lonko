# Importar todas las librerias necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.preprocessing import StandardScaler
now = datetime.datetime.now()

#----------------------------#
# Layout de la pagina
st.set_page_config(page_title = 'LONKO', layout='wide')
#----------------------------#
st.write("""

# LONKO

Software/herramienta digital para la simulación, modelación y optimización del proceso de obtención del ingrediente funcional nootrópico a partir de materias primas previamente identificadas y caracterizadas, incorporando las etapas fundamentales del proceso (i.e. extracción, purificación, encapsulación) y cuantificando el efecto de los parámetros operacionales sobre la pureza, rendimiento y productividad final. Para este desafío, el objetivo estará centrado en desarrollar el prototipo mínimo viable de dicho software , considerando como base distintos procesos y equipamientos existentes, ya probados y estudiados para productos polifenólicos (tanto a escala experimental como piloto e industrial). El software incluirá distintas funcionalidades: análisis y selección de operaciones unitarias y su pertinencia para cada etapa procesamiento, con un foco en tecnologías verdes; modelación de fenómenos termodinámicos y químicos involucrados; y simulación y optimización para la correcta operación de la cadena productiva y la reducción de incertidumbre y costos asociados a las tradicionales pruebas de ensayo y error. Las funcionalidades estarán basadas en herramientas y algoritmos de búsqueda inteligente y simulación dinámica, las que permiten predecir la salida de un proceso para cada configuración que se desee, evitando aquellas dificultades asociadas a la arista económica del diseño de procesos. La simulación, acoplada a la valoración objetiva de términos como productividad, pureza y rendimiento, posibilitan la optimización de los parámetros de operación del proceso, la cual puede ser orientada de acuerdo a las diversas restricciones inherentes del proceso (e.g. condiciones iniciales de la materia prima, calidad mínima garantizada, etc.).

""")

st.sidebar.header('Ingreso del input por el usuario')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/CarlosCaris/lonko/blob/main/data_hackathon_sample.csv)
""")
## Para ingresar los datos al la plataforma
# Esta función permite subir un archivo o permite que el usuario las agregue datos manualmente
uploaded_file = st.sidebar.file_uploader("Subir archivo .csv", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        variedad = st.sidebar.selectbox('Variedad',('Cabernet Sauvignon','Cinsault','Merlot','Carmenere','Pinot Noir','Syrah','Malbec')) # Variedades disponibles para utilizar
        valle = st.sidebar.selectbox('Valle',('Maipo alto','Maule','Itata','Colchagua','Casablanca','Cachapoal','Maipo')) # valles disponebles para utilizar
        sistema_conduccion = st.sidebar.selectbox('Sistema de conducción', ('espaldera (1x3)','espaldera (1x2)','Parron (3x3)'))
        rendimiento = st.sidebar.slider('Rendimiento (ton/ha)', 8,22,10)
        riego = st.sidebar.number_input('Riego (mm3/ha)', 10)
        DGA = st.sidebar.slider('Grados día acumulados en cosecha(°C)', 1000,1700,1300)
        plantacion = st.sidebar.slider('Año de plantacion', 1990,2015,2000)
        cosecha = st.sidebar.slider('°Brix en cosecha', 19,26,20)
        nitrogeno = st.sidebar.slider('Unidades de nitrogeno totales aplicadas ', 3,55,25)
        maceracion = st.sidebar.slider('Días de maceración ', 3,30,20)
        temperatura_maceracion = st.sidebar.slider('Temperatura de maceración (°C)', 3,30,25)

        data = {'variedad': variedad,
                'valle': valle,
                'sistema_conduccion': sistema_conduccion,
                'rendimiento': rendimiento,
                'riego': riego,
                'DGA': DGA,
                'plantacion':plantacion,
                'cosecha': cosecha,
                'nitrogeno': nitrogeno,
                'maceracion': maceracion,
                'temperatura_maceracion': temperatura_maceracion}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Una vez que se tiene la data leida, se importa la data original, con la que se entrenó el modelo
# esta data solo se usa para la codificación
hackathon_raw = pd.read_csv('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/data_hackathon.csv', sep=";")
# elimino la columna target
hackathon_raw = hackathon_raw.drop(['cantidad_semillas','cantidad_hollejo','humedad_orujo','polifenoles_totales','taninos','flavanoles','acidos_fenolicos'], axis=1)
hackathon_raw = hackathon_raw.set_index('sample_id')
df = pd.concat([input_df,hackathon_raw])
df[['plantacion']] = now.year - df[['plantacion']]
# Ahora hay que hacer la codificación de la nueva instancia que se agregó
encode = ['variedad','valle','sistema_conduccion']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
df = df[:1] # Recupero solo la primera linea, esta contiene la nueva instancia

## Ya no se están agregando cosas a la barra lateral

st.subheader('Input ingrados por el usuario')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Esperando archivo .CSV subido. Actualmente se ven los datos ingresados por el usuario')
    st.write(df)

## Ahora vienen las predicciones, vamos a cargar los modelos entrenados
# Modelo cantidad de semillas
load_cantidad_semillas = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_semillas.pkl','rb'))
# Modelo cantidad cantidad_hollejo
load_cantidad_hollejo = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_hollejo.pkl','rb'))
# Modelo humedad horujo
load_humedad_horujo = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_humedad_orujo.pkl','rb'))
# Modelo polifenoles totales
load_polifenoles_totales = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_polifenoles_totales.pkl','rb'))
# Modelo taninos
load_taninos = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_taninos.pkl','rb'))
# Modelo flavanoles
load_flavanoles = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_flavanoles.pkl','rb'))
# Modelo acidos acidos_fenolicos
load_acidos_fenolicos = pickle.load(open('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/Fitted_models/model_acidos_fenolicos.pkl','rb'))

## Aplicamos el modelo para hacer predicciones
# Predicción de cantidad de semillas
prediction_semillas = load_cantidad_semillas.predict(df)
prediction_semillas = prediction_semillas*100
# Predicción de cantidad de hollejo
prediction_hollejo = load_cantidad_hollejo.predict(df)
prediction_hollejo = prediction_hollejo*100
# Predicción hotujo
prediction_humedad = 100-(prediction_semillas+prediction_hollejo)
# Predicción polifenoles totales
prediction_polifenoles_totales = load_polifenoles_totales.predict(df)
prediction_polifenoles_totales = prediction_polifenoles_totales*100
# Predicción taninos
prediction_taninos = load_polifenoles_totales.predict(df)
prediction_taninos = prediction_taninos*100
# Predicción flavanoles
prediction_flavanoles = load_flavanoles.predict(df)
prediction_flavanoles = prediction_flavanoles*100
# Predicción acidos fenolicos
prediction_acidos_fenolicos = load_acidos_fenolicos.predict(df)
prediction_acidos_fenolicos = prediction_acidos_fenolicos*100

# Crear data fram con las predicciones
df_predicciones = pd.DataFrame({'Cantidad_semillas': prediction_semillas,
                                'Cantidad_hollejo': prediction_hollejo,
                                'Humedad_orujo': prediction_humedad,
                                'Polifenoles_totales': prediction_polifenoles_totales,
                                'taninos': prediction_taninos,
                                'flavanoles': prediction_flavanoles ,
                                'acidos fenolicos': prediction_acidos_fenolicos})
# Se transponen las predicciones para gráficar
#df_predicciones_T = df_predicciones.T

st.subheader('Predicción')
st.write(df_predicciones)

st.subheader('Plot')
st.bar_chart(df_predicciones.T)
