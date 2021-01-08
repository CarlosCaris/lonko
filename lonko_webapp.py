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

Herramienta digital basada en modelos machine learning y fundamentos termodinámicos para :
1. Predecir características del residuo agroindustrial a usar como materia prima
2. Definir proceso óptimo para la producción de ingredientes funcionales a partir del residuo agroindustrial.

Funciona como un recomendador inteligente para la selección de residuos agroindustriales adecuados y la la definición de procesos costo-eficientes que apoyen la toma de decisiones.

""")

st.sidebar.header('Ingreso del input por el usuario')
st.sidebar.write("""Ingreso de datos para predecir condiciones iniciales del proceso """)

st.sidebar.markdown("""
[Archivo CSV de ejemplo](https://github.com/CarlosCaris/lonko/blob/main/data_hackathon_sample.csv)
""")
## Para ingresar los datos al la plataforma
# Esta función permite subir un archivo o permite que el usuario las agregue datos manualmente
uploaded_file = st.sidebar.file_uploader("Subir archivo con datos de campo y proceso de vinificación .csv", type=["csv"])
if uploaded_file is not None:
    input_df_prediccion = pd.read_csv(uploaded_file)
    input_df_prediccion = input_df_prediccion.drop(['cantidad_semillas','cantidad_hollejo','humedad_orujo','polifenoles_totales','taninos','flavanoles','acidos_fenolicos'], axis=1)
    input_df_prediccion = input_df_prediccion.set_index('sample_id')
else:
    input_df_prediccion = pd.read_csv('data_hackathon_sample.csv')
    input_df_prediccion = input_df_prediccion.drop(['cantidad_semillas','cantidad_hollejo','humedad_orujo','polifenoles_totales','taninos','flavanoles','acidos_fenolicos'], axis=1)
    input_df_prediccion = input_df_prediccion.set_index('sample_id')

st.sidebar.write("""Utilice los sliders para modificar las condiciones del proceso""")
def user_input_features():
    # Proceso de secado
    masa = st.sidebar.slider('Masa(kg)',0,1000,500) # Masa en kilogramos
    # Hmedad inicial - Predicho
    T_sec = st.sidebar.slider('Temperatura de secado(C°)',60,85,75) # valles disponebles para utilizar
    #PTOS polifenoles totales de orujo seco (%) - Predicho
    # PPFS porcentaje peso PT en semillas (%) - Predicho
    # Proceso de extraccion
    te = st.sidebar.slider('Tiempo de extracción(min)', 10,240,120)
    temp_e = st.sidebar.slider('Temperatura de extracción(C°)', 40,100,75)
    ID_sol = st.sidebar.slider('Tipo de solvente utilizado\n [1]Etanol-Agua pH 2.0, 50% w/w [2]Etanol-Agua 50% w/w',1,2,1)
    #ID_procesos = st.sidebar.slider('Tipo de solvente utilizado\n Pressurized Liquid Extracion (PLE)',0,1,1)
    # PP_Tan Porcentaje de los polifenoles totales asociados a taninos (%) - Predicho
    # PP_AF  : Porcentaje de los polifenoles totales asociados a acidos-fenolicos (%) - Predicho
    # PP_FLA : Porcentaje de los polifenoles totales asociados a flavanoides (%) - Predicho

    data = {'masa': masa,
            'T_sec': T_sec,
            'te': te,
            'temp_e': temp_e,
            'ID_sol': ID_sol,
            'ID_procesos':1}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Una vez que se tiene la data leida, se importa la data original, con la que se entrenó el modelo
# esta data solo se usa para la codificación
hackathon_raw = pd.read_csv('data_hackathon_v4.csv', sep=";")
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
## Ahora vienen las predicciones, vamos a cargar los modelos entrenados
# Modelo cantidad de semillas
load_cantidad_semillas = pickle.load(open('model_semillas.pkl','rb'))
# Modelo cantidad cantidad_hollejo
load_cantidad_hollejo = pickle.load(open('model_hollejo.pkl','rb'))
# Modelo humedad horujo
load_humedad_horujo = pickle.load(open('model_humedad_orujo.pkl','rb'))
# Modelo polifenoles totales
load_polifenoles_totales = pickle.load(open('model_polifenoles_totales.pkl','rb'))
# Modelo taninos
load_taninos = pickle.load(open('model_taninos.pkl','rb'))
# Modelo flavanoles
load_flavanoles = pickle.load(open('model_flavanoles.pkl','rb'))
# Modelo acidos acidos_fenolicos
load_acidos_fenolicos = pickle.load(open('model_acidos_fenolicos.pkl','rb'))

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

st.subheader('Perfil polifenólico del ingrediente MULLO (mg/mg polifenoles totales)')
st.bar_chart(caracteristicas_producto.T)

st.subheader('Rendimiento de producción de MULLO(mg polifenoles totales/kg de orujo)')
st.write(rendimiento_extraccion)

st.subheader('Capacidad de procesamiento (kg orujo húmedo/min)')
st.write(productividad)

st.subheader('Eficiencia de recuperación de polifenoles totales (%)')
st.write(eficiencia)

# Caracteristicas del producto
caracteristicas_producto = df_predicciones[['taninos','flavanoles','acidos fenolicos']]
st.subheader('Caracterización de materia prima orujo húmedo (%)')
st.write(df_predicciones[['Cantidad_semillas','Cantidad_hollejo','Humedad_orujo']]*100)
# Polifenoles totales del orujo seco
st.subheader('Polifenoles totales del orujo seco (%)')
st.write(df_predicciones[['Polifenoles_totales']]*100)
# Perfil polifenólico del orujo seco (%)
st.subheader('Perfil polifenólico del orujo seco (mg/mg polifenoles totales)')
st.write(df_predicciones[['taninos','flavanoles','acidos fenolicos']])

st.subheader('Input ingrados por el usuario')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Esperando archivo .CSV subido. Actualmente se ven los datos ingresados por el usuario')
    st.write(df)
