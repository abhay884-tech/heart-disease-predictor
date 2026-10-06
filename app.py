import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
FEATURES=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
NUMERIC=["age","trestbps","chol","thalach","oldpeak"]
CATEGORICAL=["sex","cp","fbs","restecg","exang","slope","ca","thal"]

@st.cache_data(show_spinner="Downloading and preparing the UCI dataset...")
def load_data():
    heart=fetch_ucirepo(id=45)
    X=heart.data.features.copy(); y=heart.data.targets.copy()
    X.columns=[str(c).strip().lower() for c in X.columns]
    X=X[FEATURES].replace(["?","nan","NaN",""],np.nan)
    for c in FEATURES: X[c]=pd.to_numeric(X[c],errors="coerce")
    target=pd.to_numeric(y.iloc[:,0],errors="coerce")
    valid=target.notna(); X=X.loc[valid].reset_index(drop=True)
    return X,(target.loc[valid].reset_index(drop=True)>0).astype(int)

@st.cache_resource(show_spinner="Training the machine-learning model...")
def train_model():
    X,y=load_data()
    prep=ColumnTransformer([("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),NUMERIC),("categorical",SimpleImputer(strategy="most_frequent"),CATEGORICAL)])
    model=Pipeline([("preprocess",prep),("classifier",LogisticRegression(max_iter=2000,random_state=42))])
    model.fit(X,y); return model

st.title("❤️ Heart Disease Prediction")
st.write("Machine-learning demonstration using the UCI Heart Disease dataset and Logistic Regression.")
st.warning("Educational project only. This is NOT a medical diagnostic tool. Do not use the prediction for medical decisions.")
try: model=train_model()
except Exception as exc:
    st.error("The application could not load the dataset or train the model."); st.code(str(exc)); st.stop()

with st.form("prediction_form"):
    age=st.number_input("Age",1,120,50); sex=st.selectbox("Sex",[0,1],format_func=lambda x:"Female (0)" if x==0 else "Male (1)")
    cp=st.selectbox("Chest pain type (cp)",[1,2,3,4]); trestbps=st.number_input("Resting blood pressure (mm Hg)",50,250,120)
    chol=st.number_input("Serum cholesterol (mg/dl)",50,700,200); fbs=st.selectbox("Fasting blood sugar > 120 mg/dl",[0,1])
    restecg=st.selectbox("Resting ECG result",[0,1,2]); thalach=st.number_input("Maximum heart rate achieved",50,250,150)
    exang=st.selectbox("Exercise-induced angina",[0,1]); oldpeak=st.number_input("ST depression (oldpeak)",0.0,10.0,1.0,step=0.1)
    slope=st.selectbox("Slope of peak exercise ST segment",[1,2,3]); ca=st.selectbox("Number of major vessels (ca)",[0,1,2,3]); thal=st.selectbox("Thal",[3,6,7])
    submitted=st.form_submit_button("Predict")
if submitted:
    row=pd.DataFrame([{ "age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal }])
    prediction=int(model.predict(row)[0]); probability=float(model.predict_proba(row)[0,1])
    st.subheader("Prediction")
    if prediction: st.error(f"Model output: positive class ({probability:.1%} model probability)")
    else: st.success(f"Model output: negative class ({probability:.1%} model probability)")
    st.caption("The probability is a model output for the dataset's positive class, not an individual's medical risk or diagnosis.")
st.divider(); st.caption("Dataset: UCI Heart Disease (Cleveland subset) | Model: Logistic Regression")
