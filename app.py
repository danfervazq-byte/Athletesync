
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Panel Alto Rendimiento", layout="wide")
st.title("Panel de Alto Rendimiento - Vincula tu reloj Garmin")

# 1️⃣ Vincular Garmin (simulación de descarga)
st.subheader("1️⃣ Vincula tu reloj Garmin")
st.markdown("Haz clic para autorizar la app y descargar automáticamente los datos de tu reloj (simulado).")

datos = None
if st.button("Vincular Garmin"):
    st.success("Datos descargados desde Garmin (simulación)")
    # CSV simulado descargado automáticamente
    datos = pd.DataFrame({
        'Fecha': pd.to_datetime(['2025-08-01','2025-08-03','2025-08-05','2025-08-01','2025-08-03']),
        'Atleta': ['Dani','Dani','Dani','Antía','Antía'],
        'Duración_min': [60,45,70,50,55],
        'Distancia_km': [12,10,15,11,12],
        'FC_media': [150,145,155,148,150],
        'FC_max': [180,175,185,178,180],
        'RPE':[7,6,8,7,7],
        'HRV':[55,60,50,57,55],
        'Sueño_h':[7,8,6,7,8],
        'Tiempo_min':[60,58,65,59,60]
    })

if datos is not None:
    datos.sort_values(['Atleta','Fecha'], inplace=True)

    # Cálculo carga y fatiga
    datos['Carga'] = datos['Duración_min'] * datos['FC_media'] * datos['RPE'] / 100
    datos['Fatiga'] = datos['Carga'] / datos['HRV'] * (8 / datos['Sueño_h'])

    # Carga acumulada
    datos.set_index('Fecha', inplace=True)
    datos['Carga_semanal'] = datos.groupby('Atleta')['Carga'].rolling('7D').sum().reset_index(level=0, drop=True)
    datos['Carga_mensual'] = datos.groupby('Atleta')['Carga'].rolling('30D').sum().reset_index(level=0, drop=True)
    datos.reset_index(inplace=True)

    # Alertas semáforo
    def semaforo(fat, prom, std):
        if fat < prom: return 'Verde'
        elif fat < prom + std: return 'Amarillo'
        else: return 'Rojo'

    alerta_list = []
    for atleta in datos['Atleta'].unique():
        sub = datos[datos['Atleta']==atleta]
        prom = sub['Fatiga'].mean()
        std = sub['Fatiga'].std()
        alerta_list += [semaforo(f, prom, std) for f in sub['Fatiga']]
    datos['Alerta_fatiga'] = alerta_list

    st.subheader("Tabla de entrenamientos y alertas")
    st.dataframe(datos[['Atleta','Fecha','Carga','Fatiga','Alerta_fatiga','Carga_semanal','Carga_mensual']])

    # Gráficos interactivos
    st.subheader("Evolución de carga y fatiga")
    fig = px.line(datos, x='Fecha', y=['Carga','Fatiga','Carga_semanal','Carga_mensual'], color='Atleta', markers=True)
    st.plotly_chart(fig)

    st.subheader("Alertas de fatiga")
    fig2 = px.scatter(datos, x='Fecha', y='Fatiga', color='Alerta_fatiga', size='Carga', hover_data=['Atleta'])
    st.plotly_chart(fig2)

    # Predicción de rendimiento
    st.subheader("Predicción de rendimiento")
    for atleta in datos['Atleta'].unique():
        sub = datos[datos['Atleta']==atleta]
        X = sub[['Distancia_km','Carga']]
        y = sub['Tiempo_min']
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        dist = st.number_input(f"Distancia próxima prueba {atleta} (km):", min_value=1.0, value=10.0)
        carga_pred = st.number_input(f"Carga prevista {atleta}:", min_value=0.0, value=float(sub['Carga'].mean()))
        pred = modelo.predict([[dist,carga_pred]])[0]
        st.success(f"Tiempo estimado {atleta}: {pred:.2f} min")
