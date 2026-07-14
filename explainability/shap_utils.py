import shap
import joblib

model = joblib.load("models/random_forest.pkl")

explainer = shap.TreeExplainer(model)

def get_shap_values(data):

    return explainer.shap_values(data)