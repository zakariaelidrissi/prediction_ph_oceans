import folium as fl
from streamlit_folium import st_folium
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle

# with 1 variable  : ['TCO2']
# with 2 variables : ['TCO2', 'Temperature']
# with 3 variables : ['TCO2', 'Temperature', 'Phosphate']
# with 4 variables : ['TCO2', 'Temperature', 'Phosphate', 'Pressure']
# with 5 variables : ['TCO2', 'Temperature', 'Phosphate', 'Pressure', 'Alkalinity']

PATH_MODEL_XGB_1 = 'models/XGboost/GSXGboost_model_with1v.pkl'
PATH_MODEL_XGB_2 = 'models/XGboost/GSXGboost_model_with2v.pkl'
PATH_MODEL_XGB_3 = 'models/XGboost/GSXGboost_model_with3v.pkl'
PATH_MODEL_XGB_4 = 'models/XGboost/GSXGboost_model_with4v.pkl'
PATH_MODEL_XGB_5 = 'models/XGboost/GSXGboost_model_with5v.pkl'

PATH_MODEL_RF_1 = 'models/RF/GSRF_model_with1v.pkl'
PATH_MODEL_RF_2 = 'models/RF/GSRF_model_with2v.pkl'
PATH_MODEL_RF_3 = 'models/RF/GSRF_model_with3v.pkl'
PATH_MODEL_RF_4 = 'models/RF/GSRF_model_with4v.pkl'
PATH_MODEL_RF_5 = 'models/RF/GSRF_model_with5v.pkl'

PATH_MODEL_ANN_1 = 'models/ANN/ANN_model_with1v.h5'
PATH_MODEL_ANN_2 = 'models/ANN/ANN_model_with2v.h5'
PATH_MODEL_ANN_3 = 'models/ANN/ANN_model_with3v.h5'
PATH_MODEL_ANN_4 = 'models/ANN/ANN_model_with4v.h5'
PATH_MODEL_ANN_5 = 'models/ANN/ANN_model_with5v.h5'

st.title('PH Value prediction 🎈')

def get_pos(lng, lat):
    return lng, lat

def addMarker(m, pos):
    return m.add_child(
        fl.Marker(
            location=[pos[0], pos[1]],
            popup=
                "Latitude: " + str(pos[0]) + "<br>"
                + "Longitude: " + str(pos[1]) + "<br>"
                + "Depth: " + str(568) + "<br>"
                + "CO2: " + str(123) + "<br>"
                + "PH: " + str(7.71),
            # icon=fl.Icon(color="%s" % type_color),
        )
    )

m = fl.Map(tiles="Stamen Watercolor")
m.add_child(fl.LatLngPopup())
# fl.Marker(location=[-47.9899, 16.1719], popup='Lat : '+ str(-47.9899), tooltips='Click here').add_to(m)
# fl.Marker(location=[-12.5546, -111.7969], popup='Default popup marker', tooltips='Click here').add_to(m)
# fl.Marker(location=[-61.9390, -174.3750], popup='Default popup marker', tooltips='Click here').add_to(m)

addMarker(m, [-47.9899, 16.1719])
addMarker(m, [-12.5546, -111.7969])
addMarker(m, [-61.9390, -174.3750])
map = st_folium(m, height=400, width=700)
if map['last_clicked'] is not None:
    mapPos = get_pos(map['last_clicked']['lng'], map['last_clicked']['lat'])



depth = st.slider('Depth', 0, 7000, 25)

variables = ("1 Variable", "2 Variables", "3 Variables", "4 variables", "5 variables")
var_index = st.sidebar.selectbox("Select number of variables", range(
    len(variables)), format_func=lambda x: variables[x])

models = ("XGBoost", "Random Forest", "ANN")
models_index = st.sidebar.selectbox("Select a model", range(
    len(models)), format_func=lambda x: models[x])

col1, col2 = st.columns(2)

if var_index == 0: # TCO2
    is_valid = False
    
    tco2 = st.number_input('Insert TCO2 value')

    if tco2 != 0:
        is_valid = True
        if models_index == 0:
            model = pickle.load(open(PATH_MODEL_XGB_1, 'rb'))
            data = np.array([tco2])
        elif models_index == 1:
            model = pickle.load(open(PATH_MODEL_RF_1, 'rb'))
            data = np.array([tco2])
        else:
            model = load_model(PATH_MODEL_ANN_1)
            data = np.array([mapPos[0], mapPos[1], depth, tco2])
            data = data.reshape((1, data.shape[0]))
        # st.dataframe(data=data)
        y_pred = model.predict([data])

elif var_index == 1: # TCO2, Tempirateur
    is_valid = False
    
    with col1:
        # st.header("CO2")
        tco2 = st.number_input('Insert TCO2 value')

    with col2:
        # st.header("Tempirateur")
        tmp = st.number_input('Insert Tempirateur value')

    if tmp != 0 and tco2 != 0:
        is_valid = True
        if models_index == 0:
            model = pickle.load(open(PATH_MODEL_XGB_2, 'rb'))
            data = np.array([tco2, tmp])
        elif models_index == 1:
            model = pickle.load(open(PATH_MODEL_RF_2, 'rb'))
            data = np.array([tco2, tmp])
        else:
            model = load_model(PATH_MODEL_ANN_2)
            data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp])
            data = data.reshape((1, data.shape[0]))
            st.write(data.shape)
        # st.dataframe(data=data)
        y_pred = model.predict([data])

elif var_index == 2: # TCO2, Temperature, Phosphate
    is_valid = False

    with col1:
        # st.header("TCO2")
        tco2 = st.number_input('Insert TCO2 value')

    with col2:
        # st.header("Tempirateur")
        temp = st.number_input('Insert Tempirateur value')

    phos = st.number_input('Insert Phosphate value')

    if temp != 0 and tco2 != 0 and phos != 0:
        is_valid = True
        if models_index == 0:
            model = pickle.load(open(PATH_MODEL_XGB_3, 'rb'))
            data = np.array([tco2, temp, phos])
        elif models_index == 1:
            model = pickle.load(open(PATH_MODEL_RF_3, 'rb'))
            data = np.array([tco2, temp, phos])
        else:
            model = load_model(PATH_MODEL_ANN_3)
            data = np.array([mapPos[0], mapPos[1], depth, tco2, temp, phos])
            data = data.reshape((1, data.shape[0]))
            st.write(data.shape)
        y_pred = model.predict([data])

elif var_index == 3: # TCO2, Temperature, Phosphate, Pressure
    is_valid = False
    
    with col1:
        # st.header("TCO2")
        tco2 = st.number_input('Insert TCO2 value')
        phos = st.number_input('Insert Phosphate value')

    with col2:
        # st.header("Tempirateur")
        temp = st.number_input('Insert Tempirateur value')
        pressure = st.number_input('Insert Pressure value')

    if temp != 0 and tco2 != 0 and pressure != 0 and phos != 0:
        is_valid = True
        if models_index == 0:
            model = pickle.load(open(PATH_MODEL_XGB_4, 'rb'))
            data = np.array([tco2, temp, phos, pressure])
        elif models_index == 1:
            model = pickle.load(open(PATH_MODEL_RF_4, 'rb'))
            data = np.array([tco2, temp, phos, pressure])
        else:
            model = load_model(PATH_MODEL_ANN_4)
            data = np.array([mapPos[0], mapPos[1], depth, tco2, temp, phos, pressure])
            data = data.reshape((1, data.shape[0]))
            st.write(data.shape)
        y_pred = model.predict([data])

elif var_index == 4: # TCO2, Temperature, Phosphate, Pressure, Alkalinity,
    is_valid = False
    
    with col1:
        # st.header("TCO2")
        tco2 = st.number_input('Insert TCO2 value')
        phos = st.number_input('Insert Phosphate value')
        alkal = st.number_input('Insert Alkalinity value')

    with col2:
        # st.header("Tempirateur")
        temp = st.number_input('Insert Tempirateur value')
        pressure = st.number_input('Insert Pressure value')

    if temp != 0 and tco2 != 0 and alkal != 0 and phos != 0 and pressure != 0:
        is_valid = True
        if models_index == 0:
            model = pickle.load(open(PATH_MODEL_XGB_5, 'rb'))
            data = np.array([tco2, temp, phos, pressure, alkal])
        elif models_index == 1:
            model = pickle.load(open(PATH_MODEL_RF_5, 'rb'))
            data = np.array([tco2, temp, phos, pressure, alkal])
        else:
            model = load_model(PATH_MODEL_ANN_5)
            data = np.array([mapPos[0], mapPos[1], depth, tco2, temp, phos, pressure, alkal])
            data = data.reshape((1, data.shape[0]))
            st.write(data.shape)
        y_pred = model.predict([data])

if is_valid:
    pred = st.button('predict')
    if pred:
        print('predection : %.3f' % y_pred)
        if models_index and models_index == 2:
            st.text_input('PH value predicted', value=y_pred[0,0], disabled=True)
        else:
            st.text_input('PH value predicted', value=y_pred[0], disabled=True)
        # st.success(y_pred[0])
        # st.dataframe(data=data)