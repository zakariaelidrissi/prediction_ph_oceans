import streamlit as st
import streamlit.components.v1 as components
import folium as fl
from streamlit_folium import st_folium
from streamlit_modal import Modal
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
                + "PH: " + str(mapData.pHvalue),
            # icon=fl.Icon(color="%s" % type_color),
        )
    )

m = fl.Map(tiles="Stamen Watercolor")
# m.add_child(fl.LatLngPopup())
dataframe = pd.read_csv('dataset\carina_test.csv', sep=';')
carina_test = dataframe.filter(items=['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 
    'Phosphate', 'Pressure', 'Alkalinity', 'pHvalue'])
    
for i, row in carina_test.iterrows():
    # lat = carina_test.at[i, 'Latitude']
    # lng = carina_test.at[i, 'Longitude']
    # print(row.Longitude)
    
    # popup = 'Latitude : ' + str(lat) + '<br>' +'Longitude: ' + str(lng)
    # fl.Marker(location = [lat, lng], popup= popup, icon =fl.Icon(color='gren')).add_to(m)
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
# open_modal = st.sidebar.button('predict many items')

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:    
    dataframe = pd.read_csv(uploaded_file, sep=';')
    carina_test = dataframe.filter(items=['Longitude', 'Latitude', 'Depth', 'TCO2', 'Temperature', 
     'Phosphate', 'Pressure', 'Alkalinity', 'pHvalue'])
        
    # for i, row in carina_test.iterrows():        
    #     addMarker(m, row)    
    # st.write(carina_test)
    model = pickle.load(open(PATH_MODEL_RF_5, 'rb'))
    # model = load_model(PATH_MODEL_ANN_5)
    is_valid = False
    ypred = []
    pred = st.button('predict')
    if pred:
        for i, row in carina_test.iterrows():
            data = np.array([row.TCO2, row.Temperature, 
                        row.Phosphate, row.Pressure, row.Alkalinity])
            # st.write(data.shape)
            data = data.reshape((1, data.shape[0]))
            # st.write(data.shape)
            pr = model.predict(data)
            ypred.append(pr[0])
        carina_test["phPred"] = ypred
        st.write(carina_test)

else :
    depth = st.slider('Depth', 0, 7000, 25)
    if var_index == 0: # TCO2
        is_valid = False
        if models_index == 2:
            with col1:
                lat = st.number_input('Insert The Latitude', format="%.4f")
            with col2:
                long = st.number_input('Insert The Longitude', format="%.4f")        
        tco2 = st.number_input('Insert TCO2 value', format="%.4f")

        if tco2 != 0:
            is_valid = True
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_1, 'rb'))
                data = np.array([tco2])
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_1, 'rb'))
                data = np.array([tco2])
            elif lat != 0 and long != 0:
                model = load_model(PATH_MODEL_ANN_1)
                data = np.array([lat, long, depth, tco2])
                data = data.reshape((1, data.shape[0]))

    elif var_index == 1: # TCO2, Temperature
        is_valid = False
        if models_index == 2:        
            with col1:
                lat = st.number_input('Insert The Latitude', format="%.4f")
            with col2:
                long = st.number_input('Insert The Longitude', format="%.4f")
        with col1:
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")

        with col2:
            tmp = st.number_input('Insert Temperature value', format="%.4f")

        if tmp != 0 and tco2 != 0:
            is_valid = True
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_2, 'rb'))
                data = np.array([tco2, tmp])
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_2, 'rb'))
                data = np.array([tco2, tmp])
            elif lat != 0 and long != 0:
                model = load_model(PATH_MODEL_ANN_2)
                # data = np.array([mapPos[0], mapPos[1], depth, tco2, tmp])
                data = np.array([lat, long, depth, tco2, tmp])
                data = data.reshape((1, data.shape[0]))

    elif var_index == 2: # TCO2, Temperature, Phosphate
        is_valid = False
        if models_index == 2:        
            with col1:
                lat = st.number_input('Insert The Latitude', format="%.4f")
            with col2:
                long = st.number_input('Insert The Longitude', format="%.4f")
        with col1:
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")

        with col2:
            temp = st.number_input('Insert Temperature value', format="%.4f")

        phos = st.number_input('Insert Phosphate value', format="%.4f")

        if temp != 0 and tco2 != 0 and phos != 0:
            is_valid = True
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_3, 'rb'))
                data = np.array([tco2, temp, phos])
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_3, 'rb'))
                data = np.array([tco2, temp, phos])
            elif lat != 0 and long != 0:
                model = load_model(PATH_MODEL_ANN_3)
                data = np.array([lat, long, depth, tco2, temp, phos])
                data = data.reshape((1, data.shape[0]))

    elif var_index == 3: # TCO2, Temperature, Phosphate, Pressure
        is_valid = False
        if models_index == 2:        
            with col1:
                lat = st.number_input('Insert The Latitude', format="%.4f")
            with col2:
                long = st.number_input('Insert The Longitude', format="%.4f")
        with col1:
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
            phos = st.number_input('Insert Phosphate value', format="%.4f")

        with col2:
            temp = st.number_input('Insert Temperature value', format="%.4f")
            pressure = st.number_input('Insert Pressure value', format="%.4f")

        if temp != 0 and tco2 != 0 and pressure != 0 and phos != 0:
            is_valid = True
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_4, 'rb'))
                data = np.array([tco2, temp, phos, pressure])
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_4, 'rb'))
                data = np.array([tco2, temp, phos, pressure])
            elif lat != 0 and long != 0:
                model = load_model(PATH_MODEL_ANN_4)
                data = np.array([lat, long, depth, tco2, temp, phos, pressure])
                data = data.reshape((1, data.shape[0]))

    elif var_index == 4: # TCO2, Temperature, Phosphate, Pressure, Alkalinity,
        is_valid = False
        if models_index == 2:        
            with col1:
                lat = st.number_input('Insert The Latitude', format="%.4f")
            with col2:
                long = st.number_input('Insert The Longitude', format="%.4f")
        with col1:
            tco2 = st.number_input('Insert TCO2 value', format="%.4f")
            phos = st.number_input('Insert Phosphate value', format="%.4f")
            alkal = st.number_input('Insert Alkalinity value', format="%.4f")

        with col2:
            temp = st.number_input('Insert Temperature value', format="%.4f")
            pressure = st.number_input('Insert Pressure value', format="%.4f")

        if temp != 0 and tco2 != 0 and alkal != 0 and phos != 0 and pressure != 0:
            is_valid = True
            if models_index == 0:
                model = pickle.load(open(PATH_MODEL_XGB_5, 'rb'))
                data = np.array([tco2, temp, phos, pressure, alkal])
            elif models_index == 1:
                model = pickle.load(open(PATH_MODEL_RF_5, 'rb'))
                data = np.array([tco2, temp, phos, pressure, alkal])
            elif long != 0 and lat != 0:
                model = load_model(PATH_MODEL_ANN_5)
                data = np.array([lat, long, depth, tco2, temp, phos, pressure, alkal])
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
