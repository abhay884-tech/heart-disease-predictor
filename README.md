# ❤️ Heart Disease Prediction using Machine Learning

A Streamlit machine-learning demonstration using the UCI Heart Disease dataset and Logistic Regression.

> Educational project only. This is not a medical diagnostic system.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload the files in this folder.
3. Open Streamlit Community Cloud and connect the repository.
4. Select `app.py` as the main file.
5. Deploy.

The app downloads the UCI dataset and trains the model automatically. No model file needs to be committed to GitHub.

## Workflow
Dataset → cleaning → missing-value handling → preprocessing → Logistic Regression → Streamlit prediction interface.

## Inputs
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

## Dataset target
Original UCI target: 0 = absence, 1–4 = presence. This project maps 1–4 to class 1.

## College explanation
**Problem:** Demonstrate supervised classification of heart-disease-related records.

**Algorithm:** Logistic Regression, chosen as a simple baseline for binary classification.

**Preprocessing:** Median imputation and standardization for numerical features; most-frequent imputation for categorical features.

**Application:** Streamlit provides a browser-based interface for entering demonstration values and viewing the model output.

## Limitations
This is a small historical dataset and the result is not a medical diagnosis. Real clinical use would require extensive validation, privacy controls, safety testing and professional oversight.

## Dataset citation
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.
