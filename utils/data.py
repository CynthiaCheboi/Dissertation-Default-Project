import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os

import seaborn as sns
import plotly.graph_objects as go
import base64
import pickle

import lime
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular
from sklearn.model_selection import train_test_split

from pygwalker.api.streamlit import StreamlitRenderer

@st.cache_data
def  read_data():
    return pd.read_csv('https://raw.githubusercontent.com/CynthiaCheboi/Dissertation-Project/main/taxdefaultfinal.csv')


def get_pyg_renderer():

    df          = read_data()
    df['payment_time'] = np.where(df['payment_time']==0, 'Late payment', 'On-time payment')
    return StreamlitRenderer(df,default_tab='data',theme_key='vega',dark='dark')




#def predict_default(df):
#    with open('taxdefault_model.pkl', 'rb') as f:
#        loaded_model = pickle.load(f)
#    prediction = loaded_model.predict(df)
#    if prediction[-1] == 0:
#        results = 'The taxpayer is likely to be a defaulter'
#    else:
#        results = 'The taxpayer is likely to be a non-defaulter'
#    return results


#@st.cache_resource
def model_load():
    loaded_model  = load('taxdefault_model.pkl')
    return loaded_model 



#Function to help read searialized model
def predict_model(df):
    loaded_model = model_load()
    y_predict  = loaded_model.predict(df)
    y_predict_proba = loaded_model.predict_proba(df)[:, 1]  # Probability of positive class (class 1)

    return {'prediction': y_predict, 'probability': y_predict_proba[0]}


def model_category_using_prediction(predictions_dict,threshold):

    if predictions_dict['probability'] > float(threshold):
        return 'The taxpayer is likely to be a non-defaulter'
    else:
        return 'The taxpayer is likely to be a defaulter'

def model_category_using_y_preds(y_preds):

    if y_preds == 0:
        return 'The taxpayer is likely to be a defaulter'
    else:
        return 'The taxpayer is likely to be a non-defaulter'


# Create a button to download the model
def download_objects(model_path):
    
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Click to download",
        data=model_bytes,
        file_name=os.path.basename(model_path),
        mime="application/octet-stream"
    )
    

def train_test(df):
    X = df.drop(columns=['payment_time'])
    y = df['payment_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test, X


#def predict_fn(X):
#    loaded_models, loaded_model_results = model_load()
#    for model, result in zip(loaded_models, loaded_model_results):
#        if result['model_name']=='XGBClassifier':
#            model = model.predict_proba(X)
#    return model



#def predict_fn(X):
    loaded_model = model_load()
    predictions = None
    for model_info in loaded_model:
        if model_info['model_name'] == 'RandomForestClassifier':
            model = model_info['model']
            predictions = model.predict_proba(X)
            break
    if predictions is None:
        raise ValueError("RandomForestClassifier model not found.")
    return predictions


def predict_fn(X):
    loaded_model = model_load()
    if isinstance(loaded_model, dict):
        model = None
        for model_info in loaded_model:
            if model_info['model_name'] == 'RandomForestClassifier':
                model = model_info['model']
                break
        if model is None:
            raise ValueError("RandomForestClassifier model not found.")
    else:
        # If loaded_model is not in dict format, assume it's the model itself
        model = loaded_model

    predictions = model.predict_proba(X)
    return predictions




def lime_explainer(df,instance_index):

    X_train, X_test, y_train, y_test, X = train_test(df)

    explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(X_train),
                                                  feature_names=X.columns.tolist(),
                                                  class_names=['Non-defaulter', 'Defaulter'],
                                                  mode='classification',
                                                  random_state=42)
    
    instance   = X_test.iloc[[int(instance_index)]]
    explanation = explainer.explain_instance(instance.values[0], predict_fn, num_features=len(X.columns))
    html        = explanation.as_html()

    return html






