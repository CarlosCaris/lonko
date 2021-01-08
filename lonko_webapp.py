# Importar todas las librerias necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from Proceso_Global import Proceso_Global
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
[Example CSV input file]('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/data_hackathon_sample.csv')
""")
## Para ingresar los datos al la plataforma
# Esta función permite subir un archivo o permite que el usuario las agregue datos manualmente
uploaded_file = st.sidebar.file_uploader("Subir archivo con datos de campo y proceso de vinificación .csv", type=["csv"])
if uploaded_file is not None:
    input_df_prediccion = pd.read_csv(uploaded_file)
else:
    input_df_prediccion = pd.read_csv('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/data_hackathon_sample.csv',sep=";")
    input_df_prediccion = input_df_prediccion.drop(['cantidad_semillas','cantidad_hollejo','humedad_orujo','polifenoles_totales','taninos','flavanoles','acidos_fenolicos'], axis=1)
    input_df_prediccion = input_df_prediccion.set_index('sample_id')

def user_input_features():
    # Proceso de secado
    masa = st.sidebar.slider('Masa(kg)',0,1000,500) # Masa en kilogramos
    # Hmedad inicial - Predicho
    T_sec = st.sidebar.slider('Temperatura de secado(K°)',60,85,75) # valles disponebles para utilizar
    #PTOS polifenoles totales de orujo seco (%) - Predicho
    # PPFS porcentaje peso PT en semillas (%) - Predicho
    # Proceso de extraccion
    te = st.sidebar.slider('Tiempo de extracción(min)', 10,240,120)
    temp_e = st.sidebar.slider('Temperatura de extracción(C°)', 40,100,75)
    ID_sol = st.sidebar.slider('Tipo de solvente utilizado\n [1]Etanol-Agua pH 2.0, 50% w/w [2]Etanol-Agua 50% w/w',1,2,1)
    ID_procesos = st.sidebar.slider('Tipo de solvente utilizado\n Pressurized Liquid Extracion (PLE)',0,1,1)
    # PP_Tan Porcentaje de los polifenoles totales asociados a taninos (%) - Predicho
    # PP_AF  : Porcentaje de los polifenoles totales asociados a acidos-fenolicos (%) - Predicho
    # PP_FLA : Porcentaje de los polifenoles totales asociados a flavanoides (%) - Predicho

    data = {'masa': masa,
            'T_sec': T_sec,
            'te': te,
            'temp_e': temp_e,
            'ID_sol': ID_sol,
            'ID_procesos':ID_procesos}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Una vez que se tiene la data leida, se importa la data original, con la que se entrenó el modelo
# esta data solo se usa para la codificación
hackathon_raw = pd.read_csv('C:/Users/carlo/OneDrive/Documentos/DataScience/portafolio/Hackathon/data_hackathon_v4.csv', sep=";")
# elimino la columna target
hackathon_raw = hackathon_raw.drop(['cantidad_semillas','cantidad_hollejo','humedad_orujo','polifenoles_totales','taninos','flavanoles','acidos_fenolicos'], axis=1)
hackathon_raw = hackathon_raw.set_index('sample_id')
df = pd.concat([input_df_prediccion,hackathon_raw])
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
# Predicción de cantidad de hollejo
prediction_hollejo = load_cantidad_hollejo.predict(df)
# Predicción hotujo
prediction_humedad = 1-(prediction_semillas+prediction_hollejo)
# Predicción polifenoles totales
prediction_polifenoles_totales = load_polifenoles_totales.predict(df)
# Predicción taninos
prediction_taninos = load_polifenoles_totales.predict(df)
# Predicción flavanoles
prediction_flavanoles = load_flavanoles.predict(df)
# Predicción acidos fenolicos
prediction_acidos_fenolicos = load_acidos_fenolicos.predict(df)

# Crear data fram con las predicciones
df_predicciones = pd.DataFrame({'Cantidad_semillas': prediction_semillas,
                                'Cantidad_hollejo': prediction_hollejo,
                                'Humedad_orujo': prediction_humedad,
                                'Polifenoles_totales': prediction_polifenoles_totales,
                                'taninos': prediction_taninos,
                                'flavanoles': prediction_flavanoles ,
                                'acidos fenolicos': prediction_acidos_fenolicos})
array_predicciones = df_predicciones.to_numpy()
array_input = input_df.to_numpy()
# Guardar los datos de entrada del proceso Proceso_Global
Masa_0 = array_input[0,0]
Hum_0 = array_predicciones[0,2]
T_Sec = array_input[0,1]+273
PTOS = array_predicciones[0,3]
PPFS = array_predicciones[0,0]*array_predicciones[0,3]
te = array_input[0,2]
Temp_e = array_input[0,3]
ID_Sol = array_input[0,4]
ID_Proceso = array_input[0,5]
PP_Tan = array_predicciones[0,4]
PP_AF = array_predicciones[0,6]
PP_FLA = array_predicciones[0,5]


# Se aplica la función de proceso global que nos permite simular el proceso y los resultados
retenciones = Proceso_Global(Masa_0,Hum_0,T_Sec,PTOS,PPFS,te,Temp_e,ID_Sol,ID_Proceso,PP_Tan,PP_AF,PP_FLA)
df_retenciones = pd.DataFrame(retenciones)

## Calculo de indicadores de desempeño
# Rendimiento
rendimiento_extraccion = df_retenciones[3:4:].to_numpy()/Masa_0
# Productividad
productividad = Masa_0/te+df_retenciones[2:3:].to_numpy()
# Eficiencia
eficiencia  = df_retenciones[0:1:].to_numpy()*df_retenciones[2:3:].to_numpy()*100
# Caracteristicas del producto
caracteristicas_producto = df_predicciones[['taninos','flavanoles','acidos fenolicos']]


st.subheader('Predicción')
st.write(df_predicciones)

st.subheader('Input Usuario')
st.write(input_df)

st.subheader('Resultado proceso global')
st.write(df_retenciones)

st.subheader('Plot')
st.bar_chart(caracteristicas_producto.T)

