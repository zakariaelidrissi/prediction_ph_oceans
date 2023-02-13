import streamlit as st
import streamlit.components.v1 as components
import folium as fl
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from ph_classes.CalculpHfromTATC import CalculpHfromTATC

# with 1 variable  : ['Longitude', 'Latitude', 'Depth', 'TCO2']
# with 2 variables : ['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature']
# with 3 variables : ['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 'Phosphate']
# with 4 variables : ['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 'Phosphate', 'Pressure']
# with 5 variables : ['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 'Phosphate', 'Pressure', 'Alkalinity']

PATH_MODEL_XGB_1 = 'models2/XGboost/GSXGboost_model_with1v.pkl'
PATH_MODEL_XGB_2 = 'models2/XGboost/GSXGboost_model_with2v.pkl'
PATH_MODEL_XGB_3 = 'models2/XGboost/GSXGboost_model_with3v.pkl'
PATH_MODEL_XGB_4 = 'models2/XGboost/GSXGboost_model_with4v.pkl'
PATH_MODEL_XGB_5 = 'models2/XGboost/GSXGboost_model_with5v.pkl'

PATH_MODEL_RF_1 = 'models2/RF/GSRF_model_with1v.pkl'
PATH_MODEL_RF_2 = 'models2/RF/GSRF_model_with2v.pkl'
PATH_MODEL_RF_3 = 'models2/RF/GSRF_model_with3v.pkl'
PATH_MODEL_RF_4 = 'models2/RF/GSRF_model_with4v.pkl'
PATH_MODEL_RF_5 = 'models2/RF/GSRF_model_with5v.pkl'

PATH_MODEL_ANN_1 = 'models2/ANN/ANN_model_with1v.h5'
PATH_MODEL_ANN_2 = 'models2/ANN/ANN_model_with2v.h5'
PATH_MODEL_ANN_3 = 'models2/ANN/ANN_model_with3v.h5'
PATH_MODEL_ANN_4 = 'models2/ANN/ANN_model_with4v.h5'
PATH_MODEL_ANN_5 = 'models2/ANN/ANN_model_with5v.h5'

st.title('PH Value prediction 🎈')
st.write('\n')
st.write('\n')
def get_pos(lng, lat):
    return lat, lng

def addMarker(m, mapData):
    return m.add_child(
        fl.Marker(
            location=[mapData.Latitude, mapData.Longitude],
            popup=
                "Latitude: " + str(mapData.Latitude) + "<br>"
                + "Longitude: " + str(mapData.Longitude) + "<br>"
                + "Depth: " + str(mapData.Depth) + "<br>"
                + "TCO2: " + str(mapData.TCO2) + "<br>"
                + "Temperature: " + str(mapData.Temperature) + "<br>"
                + "Phosphate: " + str(mapData.Phosphate) + "<br>"
                + "Pressure: " + str(mapData.Pressure) + "<br>"
                + "Alkalinity: " + str(mapData.Alkalinity) + "<br>"
                + "PH: " + str(mapData.pHvalue)
        )
    )

m = fl.Map(tiles="Stamen Watercolor")
m.add_child(fl.LatLngPopup())
dataframe = pd.read_csv('dataset\carina_test.csv', sep=';')
carina_test = dataframe.filter(items=['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 
    'Phosphate', 'Pressure', 'Alkalinity', 'pHvalue'])
    
for i, row in carina_test.iterrows():
    addMarker(m, row)

map = st_folium(m, height=500, width=800)
if map['last_clicked'] is not None:
    mapPos = get_pos(map['last_clicked']['lng'], map['last_clicked']['lat'])


variables = ("1 Variable", "2 Variables", "3 Variables", "4 variables", "5 variables")
var_index = st.sidebar.selectbox("Select number of variables", range(
    len(variables)), format_func=lambda x: variables[x], key=0)

models = ("XGBoost", "Random Forest", "ANN")
models_index = st.sidebar.selectbox("Select a model", range(
    len(models)), format_func=lambda x: models[x], key=1)

uploaded_file = st.sidebar.file_uploader("Choose a file", key=2)

depth = st.slider('Depth', 0, 7000, 25)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file, sep=';')
    carina_test = dataframe.filter(items=['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 
     'Phosphate', 'Pressure', 'Alkalinity', 'pHvalue'])
        
    model = pickle.load(open(PATH_MODEL_RF_5, 'rb'))

    is_valid = False
    ypred = []
    pred = st.button('predict')
    ph = [CalculpHfromTATC(row.TCO2, row.Alkalinity, row.Salinity, 1, row.Temperature, 1, row.Pressure, row.Phosphate, 
        row.Silicate, 1, 2) - 0.27 for index, row in dataframe.iterrows()]
    if pred:
        for i, row in carina_test.iterrows():
            data = np.array([row.Longitude, row.Latitude, row.Depth, row.TCO2, row.Temperature, 
                        row.Phosphate, row.Pressure, row.Alkalinity])
            data = data.reshape((1, data.shape[0]))
            pr = model.predict(data)
            ypred.append(pr[0])
        carina_test["phPred"] = ypred
        new_data = pd.DataFrame()
        new_data['phcalc'] = ph
        new_data['phPred'] = ypred
        st.write(new_data)
        st.line_chart(new_data)

else :    
    if var_index == 0: # Longitude, Latitude, Depth, TCO2
        is_valid = False        
        with col1:
            lat = st.number_input('Insert The Latitude', format="%.4f")
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
        with col2:
            lng = st.number_input('Insert The Longitude', format="%.4f")        
        
        if tco2 != 0:
            is_valid = True
            if lat != 0 and lng != 0:                
                data = np.array([lng, lat, depth, tco2])
            elif mapPos != []: 
                data = np.array([mapPos[0], mapPos[1], depth, tco2])
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_1, 'rb'))
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_1, 'rb'))
            else:
                model = load_model(PATH_MODEL_ANN_1)
                data = data.reshape((1, data.shape[0]))

    elif var_index == 1: # Longitude, Latitude, Depth, TCO2, Temperature
        is_valid = False        
        with col1:
            lat = st.number_input('Insert The Latitude', format="%.4f")
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")

        with col2:
            lng = st.number_input('Insert The Longitude', format="%.4f")
            tmp = st.number_input('Insert Temperature value', format="%.4f")

        if tmp != 0 and tco2 != 0:
            is_valid = True
            if lat != 0 and lng != 0:                
                data = np.array([lng, lat, depth, tco2, tmp])
            elif mapPos != []: 
                data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp])
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_2, 'rb'))
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_2, 'rb'))
            else:
                model = load_model(PATH_MODEL_ANN_2)
                data = data.reshape((1, data.shape[0]))

    elif var_index == 2: # Longitude, Latitude, Depth, TCO2, Temperature, Phosphate
        is_valid = False    
        with col1:
            lat = st.number_input('Insert The Latitude', format="%.4f")
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
            phos = st.number_input('Insert Phosphate value', format="%.4f")

        with col2:
            lng = st.number_input('Insert The Longitude', format="%.4f")
            tmp = st.number_input('Insert Temperature value', format="%.4f")

        if tmp != 0 and tco2 != 0 and phos != 0:
            is_valid = True
            if lat != 0 and lng != 0:                
                data = np.array([lng, lat, depth, tco2, tmp, phos])
            elif mapPos != []: 
                data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp, phos])
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_3, 'rb'))
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_3, 'rb'))
            else:
                model = load_model(PATH_MODEL_ANN_3)
                data = data.reshape((1, data.shape[0]))

    elif var_index == 3: # Longitude, Latitude, Depth, TCO2, Temperature, Phosphate, Pressure
        is_valid = False
        with col1:
            lat = st.number_input('Insert The Latitude', format="%.4f")
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
            phos = st.number_input('Insert Phosphate value', format="%.4f")

        with col2:
            lng = st.number_input('Insert The Longitude', format="%.4f")
            tmp = st.number_input('Insert Temperature value', format="%.4f")
            pressure = st.number_input('Insert Pressure value', format="%.4f")

        if tmp != 0 and tco2 != 0 and pressure != 0 and phos != 0:
            is_valid = True
            if lat != 0 and lng != 0:                
                data = np.array([lng, lat, depth, tco2, tmp, phos, pressure])
            elif mapPos != []: 
                data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp, phos, pressure])
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_4, 'rb'))
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_4, 'rb'))
            else:
                model = load_model(PATH_MODEL_ANN_4)                
                data = data.reshape((1, data.shape[0]))

    elif var_index == 4: # Longitude, Latitude, Depth, TCO2, Temperature, Phosphate, Pressure, Alkalinity,
        is_valid = False        
        with col1:
            lat = st.number_input('Insert The Latitude', format="%.4f")
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
            phos = st.number_input('Insert Phosphate value', format="%.4f")
            alkal = st.number_input('Insert Alkalinity value', format="%.4f")

        with col2:
            lng = st.number_input('Insert The Longitude', format="%.4f")
            tmp = st.number_input('Insert Temperature value', format="%.4f")
            pressure = st.number_input('Insert Pressure value', format="%.4f")

        if tmp != 0 and tco2 != 0 and alkal != 0 and phos != 0 and pressure != 0:
            is_valid = True
            if lat != 0 and lng != 0:                
                data = np.array([lng, lat, depth, tco2, tmp, phos, pressure, alkal])
            elif mapPos != []: 
                data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp, phos, pressure, alkal])
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_5, 'rb'))
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_5, 'rb'))
            else:
                model = load_model(PATH_MODEL_ANN_5)
                data = data.reshape((1, data.shape[0]))

if is_valid:
    pred = st.button('predict')
    if pred and model is not None:
        y_pred = model.predict([data])
        # print('predection : %.3f' % y_pred)
        if models_index and models_index == 2:
            st.text_input('PH value predicted', value=y_pred[0,0], disabled=True)
        else:
            st.text_input('PH value predicted', value=y_pred[0], disabled=True)
        # st.success(y_pred[0])
