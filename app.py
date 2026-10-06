import streamlit as st
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


@st.cache_data
def load_dataset():

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    X.columns = [
        str(column).strip().lower()
        for column in X.columns
    ]

    X = X[FEATURES]

    X = X.replace(
        ["?", "nan", "NaN", ""],
        np.nan
    )

    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    target = pd.to_numeric(
        y.iloc[:, 0],
        errors="coerce"
    )

    valid_rows = target.notna()

    X = X.loc[valid_rows].reset_index(drop=True)

    target = target.loc[valid_rows].reset_index(drop=True)

    y_binary = (target > 0).astype(int)

    return X, y_binary


@st.cache_resource
def train_model():

    X, y = load_dataset()

    preprocessing = ColumnTransformer(
        transformers=[

            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        )
                    ]
                ),
                NUMERIC_FEATURES
            ),

            (
                "categorical",
                SimpleImputer(
                    strategy="most_frequent"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model = Pipeline(
        steps=[

            (
                "preprocessing",
                preprocessing
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X, y)

    return model


st.title("❤️ Heart Disease Prediction")

st.write(
    """
    A machine-learning demonstration using
    the UCI Heart Disease dataset and
    Logistic Regression.
    """
)


st.warning(
    """
    This project is for educational purposes only.
    It is NOT a medical diagnostic system.
    Do not use the prediction for medical decisions.
    """
)


try:

    with st.spinner(
        "Loading dataset and training model..."
    ):

        model = train_model()

except Exception as error:

    st.error(
        "Unable to load the dataset or train the model."
    )

    st.code(str(error))

    st.stop()


st.subheader("Enter Demo Values")


with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }[x]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        [3, 6, 7],
        format_func=lambda x: {
            3: "Normal",
            6: "Fixed Defect",
            7: "Reversible Defect"
        }[x]
    )

    predict_button = st.form_submit_button(
        "Predict"
    )


if predict_button:

    input_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }]
    )

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"""
            Model Output: Positive Class

            Model probability:
            {probability:.1%}
            """
        )

    else:

        st.success(
            f"""
            Model Output: Negative Class

            Model probability:
            {probability:.1%}
            """
        )

    st.caption(
        """
        This probability is the model's output
        for the dataset's positive class.
        It is NOT an individual's medical risk
        or diagnosis.
        """
    )


st.divider()

st.caption(
    "Dataset: UCI Heart Disease | "
    "Algorithm: Logistic Regression"
)
