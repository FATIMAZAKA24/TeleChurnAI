import pandas as pd
from catboost import CatBoostClassifier
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from api.ml_model.retention_strategies import generate_retention_strategy

print("1")
# Load trained classification model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")
print("CatBoost Classification Model Loaded!")

def single_predict(input_data):
    try:
        # Ensure input_data is a DataFrame
        input_data = pd.DataFrame([input_data])
        print(input_data)

        # Make predictions (Classification)
        prediction_probabilities = model.predict_proba(input_data)[:, 1]
        predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

        # Get SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        # Generate SHAP waterfall plot as base64 string
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()

        # Determine churn probability and cluster
        prob = float(prediction_probabilities[0])
        if prob < 0.25:
            cluster = "low"
        elif prob < 0.50:
            cluster = "medium"
        elif prob < 0.75:
            cluster = "medhigh"
        else:
            cluster = "high"

        # Extract top 3-4 features by absolute SHAP value
        shap_df = pd.DataFrame({
            "feature": input_data.columns,
            "shap_value": shap_values.values[0]
        })
        top_features = (
            shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
            ["feature"].tolist()[:4]
        )

        # Simple SHAP explanation text
        shap_explanation = (
            "This graph highlights the features that most influenced the churn prediction. "
            "Features with positive SHAP values increase churn risk, while negative values support retention."
        )

        # Generate retention strategy text
        retention_strategy = generate_retention_strategy(cluster, top_features)

        return {
            "predictions": predictions,
            "probabilities": prediction_probabilities.tolist(),
            "shap_plot": img_b64,  # base64 encoded image string of SHAP plot
            "shap_explanation": shap_explanation,
            "retention_strategy": retention_strategy
        }

    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

# import pandas as pd
# from catboost import CatBoostClassifier
# import shap
# import matplotlib.pyplot as plt
# import io  
# import uuid
# import os

# print("1")
# # Load trained classification model
# model = CatBoostClassifier()
# model.load_model("catboost_model.cbm")
# print("CatBoost Classification Model Loaded!")
        
# # Define the prediction function
# def single_predict(input_data):
#     try:
#         # Ensure input_data is a DataFrame
#         input_data = pd.DataFrame([input_data])
#         print(input_data)

#         # Make predictions (Classification)
#         prediction_probabilities = model.predict_proba(input_data)[:, 1]
#         predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

#         # Get SHAP values
#         explainer = shap.Explainer(model)
#         shap_values = explainer(input_data)

#         # Generate unique filename
#         filename = f"shap_plot_{uuid.uuid4().hex}.html"
#         filepath = os.path.join("static", "shap_plots", filename)  # Save to Django's static folder

#         # Ensure folder exists
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)

#         # Save the HTML plot
#         with open(filepath, "w") as f:
#             f.write("<html><head>")
#             f.write(shap.getjs())
#             f.write("</head><body>")
#             f.write(str(shap.plots.force(shap_values[0], matplotlib=False)))
#             f.write("</body></html>")

#         # Return relative path to HTML for frontend to load via iframe
#         return {
#             "predictions": predictions,
#             "probabilities": prediction_probabilities.tolist(),
#             "shap_html_path": f"/static/shap_plots/{filename}"
#         }
#         # # Create force plot and save as HTML string
#         # shap_html = f"""
#         # <head>{shap.getjs()}</head>
#         # <body>{shap.plots.force(shap_values[0], matplotlib=False)}</body>
#         # """
#         # Return path to frontend
#         # return {
#         #     "predictions": predictions,
#         #     "probabilities": prediction_probabilities.tolist(),
#         #     "shap_html": shap_html
#         # }
#     except Exception as e:
#         return {"error": f"Error during prediction: {str(e)}"}


# import pandas as pd
# from catboost import CatBoostClassifier

# # Define the prediction function
# def single_predict(input_data):
#     try:
#         print("1")
#         # Load trained model
#         model = CatBoostClassifier()
#         model.load_model("catboost_model.cbm")
#         print("CatBoost is working!")

#         # Ensure input_data is a DataFrame
#         input_data = pd.DataFrame([input_data])
#         print(input_data)

#         # 🚀 **Make predictions (FIXED)**
#         prediction_probabilities = model.predict_proba(input_data)[:, 1]
#         predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

#         return {"predictions": predictions, "probabilities": prediction_probabilities.tolist()}
    
#     except Exception as e:
#         return {"error": f"Error during prediction: {str(e)}"}


# import numpy as np
# import pandas as pd
# import joblib
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# def single_predict(input_data):
#     # Load trained model, encoder, and scaler
#     try:
#         print("1")
#         model = joblib.load("gradient_boosting_model.pkl")
#         scaler = joblib.load("scaler.pkl")  # Load trained StandardScaler
#         trained_features = joblib.load("trained_features.pkl")
#         encoder = joblib.load("encoder.pkl")
#         print("2")
#         # Convert input dictionary to DataFrame
#         df = pd.DataFrame([input_data])

#         # Convert numeric fields
#         numeric_fields = ["tenure", "monthlycharges", "totalcharges", "seniorcitizen","dependents"]
#         #df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')

#         # Identify categorical features
#         categorical_cols = df.select_dtypes(include=["object"]).columns
        
#         for col in categorical_cols:
#             df[col] = df[col].astype(str).fillna("unknown")

#         # Ensure input has correct categorical columns before encoding
#         df = df.reindex(columns=encoder.feature_names_in_, fill_value="No")  # Match categorical input with training

#         # tf_encoded = encoder.transform(trained_features)
#         # tf_encoded = pd.DataFrame(
#         #     tf_encoded,
#         # )

#         # Apply OneHotEncoder
#         df_encoded = encoder.transform(df)
#         df_encoded = pd.DataFrame(
#             df_encoded,
#             columns=encoder.get_feature_names_out(),  # Ensure correct feature names
#             index=df.index
#         )

#         # Debug: Print categorical feature names after encoding
#         print(f"📌 Categorical columns after encoding: {df_encoded.columns.tolist()}")

#         # Merge back encoded features
#         df = df.drop(columns=categorical_cols, errors="ignore")  # ✅ Prevent KeyError
#         df = pd.concat([df, df_encoded], axis=1)

#         # Keep only features that the model was trained on
#         df = df.reindex(columns=encoder.get_feature_names_out(), fill_value=0)  # Ensure missing columns are filled

#         df["monthlycharges"] = 10.1
#         df["paperlessbilling"] = "yes"
#         df["tenure"] = 1
#         df["totalcharges"] = 11.1 
#         df = df.drop(columns=["contract_Month-to-month", "contract_One year","contract_Two year", "deviceprotection_No", "deviceprotection_No internet service"], errors="ignore")
#         # Debug: Print final features passed to model
#         print(df.columns.tolist())
#         # Apply the **trained** scaler (instead of fitting a new one)
#         df_scaled = scaler.transform(df)
#         print("7")
#         # Make a prediction
#         prediction_prob = model.predict_proba(df_scaled)[:, 1]  # Probability of churn
#         print("8")
#         prediction = "Yes" if prediction_prob[0] >= 0.5 else "No"
#         return {"prediction": prediction, "probability": round(prediction_prob[0], 3)}

#     except Exception as e:
#         return {"error": f"Error during prediction: {str(e)}"}

# import pandas as pd
# from catboost import CatBoostClassifier

# # Define the prediction function
# def single_predict(input_data):
    # try:
    #     print("1")
    #     # Load trained model
    #     model = CatBoostClassifier()
    #     model.load_model("catboost_model (1).cbm")
    #     print("CatBoost is working!")

    #     # Ensure input_data is a DataFrame
    #     input_data = pd.DataFrame([input_data])
    #     print("2")

    #     categorical_features = [
    #         "contract", "internetservice", "onlinesecurity", "onlinebackup",
    #         "deviceprotection", "techsupport", "streamingtv", "streamingmovies",
    #         "paymentmethod", "paperlessbilling", "partner", "dependents", "seniorcitizen"]

    #     numeric_features = ["tenure", "monthlycharges", "totalcharges"]

    #     # ✅ Ensure numeric features are properly converted
    #     for col in numeric_features:
    #         if col in input_data.columns:
    #             input_data[col] = pd.to_numeric(input_data[col], errors="coerce")  # Convert properly

    #     # ✅ Ensure categorical features are **only** strings (No float issue)
    #     for col in categorical_features:
    #         if col in input_data.columns:
    #             input_data[col] = input_data[col].astype(str).str.strip()  # Remove spaces
    #             input_data[col] = input_data[col].replace({float("nan"): "Unknown"})  # Replace NaNs


    #     # 🚀 **Explicitly tell CatBoost which features are categorical**
    #     #input_data[categorical_features] = input_data[categorical_features].astype("category")

    #     print("Processed Input Data:\n", input_data.dtypes)
    #     print(input_data)

    #     # 🚀 **Make predictions (FIXED)**
    #     prediction_probabilities = model.predict_proba(input_data)[:, 1]
    #     predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

    #     return {"predictions": predictions, "probabilities": prediction_probabilities.tolist()}
    
    # except Exception as e:
    #     return {"error": f"Error during prediction: {str(e)}"}
# import pandas as pd
# from catboost import CatBoostClassifier
