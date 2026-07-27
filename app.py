import os
import io
import base64
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
survmodel_path = os.path.join(BASE_DIR, 'survivemodel.pkl')
explainer_path = os.path.join(BASE_DIR, 'explainer.bz2')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(survmodel_path, 'rb') as f:
    survmodel = pickle.load(f)

explainer = joblib.load(explainer_path)


def positive_class_explanation(explainer, features):
    """Return SHAP values for class 1 across SHAP's supported output formats."""
    # Obtain raw SHAP output
    if callable(explainer):
        res = explainer(features)
    else:
        res = explainer.shap_values(features)

    # Extract base / expected value for positive class (class 1)
    raw_exp = getattr(explainer, 'expected_value', None)
    if hasattr(res, 'base_values'):
        raw_exp = res.base_values

    expected_values = np.asarray(raw_exp)
    if expected_values.ndim == 0:
        base_val = float(expected_values)
    elif expected_values.size == 1:
        base_val = float(expected_values.flat[0])
    elif expected_values.size >= 2:
        base_val = float(expected_values.flat[1])
    else:
        base_val = 0.0

    # Extract shap values array for positive class
    if hasattr(res, 'values'):
        raw_vals = res.values
    else:
        raw_vals = res

    if isinstance(raw_vals, list):
        if len(raw_vals) > 1:
            vals = np.asarray(raw_vals[1])
        else:
            vals = np.asarray(raw_vals[0])
        if vals.ndim >= 2:
            vals = vals[0]
        return base_val, vals.flatten()

    vals = np.asarray(raw_vals)

    if vals.ndim == 3:
        # Shape: (num_samples, num_features, num_classes) e.g. (1, 23, 2)
        return base_val, vals[0, :, 1]
    elif vals.ndim == 2:
        # Could be (1, num_features) or (num_features, num_classes)
        if vals.shape[0] == 1:
            return base_val, vals[0]
        elif vals.shape[1] == 2:
            return base_val, vals[:, 1]
        else:
            return base_val, vals.flatten()
    elif vals.ndim == 1:
        return base_val, vals
    else:
        return base_val, vals.flatten()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = 1 if str(request.form.get("gender", "0")) == "1" else 0
    SeniorCitizen = 1 if 'SeniorCitizen' in request.form else 0
    Partner = 1 if 'Partner' in request.form else 0
    Dependents = 1 if 'Dependents' in request.form else 0
    PaperlessBilling = 1 if 'PaperlessBilling' in request.form else 0

    MonthlyCharges = float(request.form.get("MonthlyCharges", 0))
    Tenure = float(request.form.get("Tenure", 0))
    TotalCharges = MonthlyCharges * Tenure

    PhoneService = 1 if 'PhoneService' in request.form else 0
    MultipleLines = 1 if ('MultipleLines' in request.form and PhoneService == 1) else 0

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    internet_val = str(request.form.get("InternetService", "1"))
    if internet_val == "0":
        InternetService_No = 1
    elif internet_val == "2":
        InternetService_Fiberoptic = 1

    OnlineSecurity = 1 if ('OnlineSecurity' in request.form and InternetService_No == 0) else 0
    OnlineBackup = 1 if ('OnlineBackup' in request.form and InternetService_No == 0) else 0
    DeviceProtection = 1 if ('DeviceProtection' in request.form and InternetService_No == 0) else 0
    TechSupport = 1 if ('TechSupport' in request.form and InternetService_No == 0) else 0
    StreamingTV = 1 if ('StreamingTV' in request.form and InternetService_No == 0) else 0
    StreamingMovies = 1 if ('StreamingMovies' in request.form and InternetService_No == 0) else 0

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    contract_val = str(request.form.get("Contract", "0"))
    if contract_val == "1":
        Contract_Oneyear = 1
    elif contract_val == "2":
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    payment_val = str(request.form.get("PaymentMethod", "0"))
    if payment_val == "1":
        PaymentMethod_CreditCard = 1
    elif payment_val == "2":
        PaymentMethod_ElectronicCheck = 1
    elif payment_val == "3":
        PaymentMethod_MailedCheck = 1

    features = [
        gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        PaperlessBilling, MonthlyCharges, TotalCharges, InternetService_Fiberoptic,
        InternetService_No, Contract_Oneyear, Contract_Twoyear, PaymentMethod_CreditCard,
        PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
    ]

    columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    final_features_arr = np.array([features], dtype=np.float64)
    final_features_df = pd.DataFrame(final_features_arr, columns=columns)

    prediction = model.predict_proba(final_features_df)
    output = float(prediction[0, 1])

    # Shap Values
    expected_value, shap_values = positive_class_explanation(explainer, final_features_df)
    shap_img = io.BytesIO()
    fig = shap.force_plot(expected_value, shap_values, features=final_features_arr[0], feature_names=columns, matplotlib=True, show=False)
    if fig is None or not hasattr(fig, 'savefig'):
        fig = plt.gcf()
    fig.savefig(shap_img, bbox_inches="tight", format='png')
    plt.close(fig)
    plt.close('all')
    shap_img.seek(0)
    shap_url = base64.b64encode(shap_img.getvalue()).decode()

    # Hazard and Survival Analysis
    surv_feats = np.array([[
        gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
        InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
        PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck
    ]], dtype=np.float64)

    hazard_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_cumulative_hazard(surv_feats).plot(ax=ax, color='red')
    ax.axvline(x=Tenure, color='blue', linestyle='--')
    ax.legend(labels=['Hazard', 'Current Position'])
    ax.set_xlabel('Tenure', fontsize=10)
    ax.set_ylabel('Cumulative Hazard', fontsize=10)
    ax.set_title('Cumulative Hazard Over Time')
    plt.savefig(hazard_img, format='png', bbox_inches='tight')
    plt.close(fig)
    hazard_img.seek(0)
    hazard_url = base64.b64encode(hazard_img.getvalue()).decode()

    surv_img = io.BytesIO()
    fig, ax = plt.subplots()
    survmodel.predict_survival_function(surv_feats).plot(ax=ax, color='red')
    ax.axvline(x=Tenure, color='blue', linestyle='--')
    ax.legend(labels=['Survival Function', 'Current Position'])
    ax.set_xlabel('Tenure', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title('Survival Probability Over Time')
    plt.savefig(surv_img, format='png', bbox_inches='tight')
    plt.close(fig)
    surv_img.seek(0)
    surv_url = base64.b64encode(surv_img.getvalue()).decode()

    life = survmodel.predict_survival_function(surv_feats).reset_index()
    life.columns = ['Tenure', 'Probability']
    filt = life.Tenure[life.Probability > 0.1]
    max_life = float(filt.max()) if not filt.empty else 0.0

    CLTV = max_life * MonthlyCharges

    # Gauge plot
    def degree_range(n):
        start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
        end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
        mid_points = start + ((end - start) / 2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) - np.radians(90))
        return rotation

    def gauge(labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
              colors=['#007A00', '#0063BF', '#FFCC00', '#ED1C24'], Probability=1, fname=False):

        colors = colors[::-1]
        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)
        labels = labels[::-1]

        patches = []
        for ang, c in zip(ang_range, colors):
            patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
            patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        for p in patches:
            ax.add_patch(p)

        for mid, lab in zip(mid_points, labels):
            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,
                    horizontalalignment='center', verticalalignment='center', fontsize=14,
                    fontweight='bold', rotation=rot_text(mid))

        r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        prob_str = f'{Probability:.2f}'
        ax.text(0, -0.05, 'Churn Probability ' + prob_str, horizontalalignment='center',
                verticalalignment='center', fontsize=22, fontweight='bold')

        pos = (1 - Probability) * 180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format='png', bbox_inches='tight')
        plt.close(fig)
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability=output)

    return render_template(
        'index.html',
        prediction_text='Churn probability is {} and Expected Life Time Value is ${}'.format(round(output, 2), CLTV),
        url_1=gauge_url,
        url_2=shap_url,
        url_3=hazard_url,
        url_4=surv_url
    )


if __name__ == "__main__":
    app.run(debug=True)

