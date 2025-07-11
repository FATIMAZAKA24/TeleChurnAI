# import joblib
# #from train_model import trainModel
# # Define the feature list used for training
# important_features = [
#     "tenure", "monthlycharges", "totalcharges", "contract", "internetservice",
#     "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport",
#     "streamingtv", "streamingmovies", "paymentmethod", "paperlessbilling",
#     "partner", "dependents", "seniorcitizen"
# ]

# # Save the feature list as 'trained_features.pkl'
# joblib.dump(important_features, "trained_features.pkl")

# print("✅ Feature list saved successfully as 'trained_features.pkl'")


# import joblib
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer

# def preprocess_and_load_predict(file):
#     # Load trained features and model
#     model = joblib.load("api/ml_model/gradient_boosting_model.pkl")
#     print(model)
#     print("Model Loaded")

#     trained_features = joblib.load("api/ml_model/trained_features.pkl")
#     print("Trained Features Loaded")

#     # Read CSV file into DataFrame
#     df = pd.read_csv(file)
#     print("CSV File Loaded")
    
#     # Handle categorical columns
#     categorical_cols = df.select_dtypes(include=["object"]).columns
#     print("Categorical Columns Identified:", categorical_cols)
#     for col in categorical_cols:
#         df[col] = df[col].astype(str).fillna("unknown")

#     # Handle missing values
#     imputer = SimpleImputer(strategy="most_frequent")
#     df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#     # Encode categorical columns
#     label_encoder = LabelEncoder()
#     for col in categorical_cols:
#         df[col] = label_encoder.fit_transform(df[col])

#     # Ensure correct column order
#     missing_features = set(trained_features) - set(df.columns)
#     for feature in missing_features:
#         df[feature] = 0

#     df = df[trained_features]

#     # Feature scaling
#     scaler = StandardScaler()
#     df = scaler.fit_transform(df)

#     # Predict churn probabilities
#     prediction_probabilities = model.predict_proba(df)[:, 1]
#     predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

#     return {"predictions": predictions, "probabilities": prediction_probabilities.tolist()}

from api.ml_model.retention_strategies import generate_retention_strategy

def generate_shap_explanation(df_cluster, shap_values_cluster, cluster_name, plot_type, top_n=3):
    """
    Generate a dynamic, unique natural language explanation for a given cluster and plot type,
    based on SHAP values and feature distributions.
    
    Args:
        df_cluster: DataFrame of the cluster data points.
        shap_values_cluster: SHAP values numpy array for the cluster (samples x features).
        cluster_name: str, e.g. 'Low', 'Medium', 'High'.
        plot_type: 'beeswarm', 'bar', or 'heatmap'.
        top_n: number of top features to analyze.
        
    Returns:
        explanation_text: str
    """
    import numpy as np
    
    # Mean absolute SHAP values per feature (importance)
    mean_abs_shap = np.mean(np.abs(shap_values_cluster), axis=0)
    
    # Get indices of top features by importance
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = df_cluster.columns[top_indices]
    top_shap_vals = mean_abs_shap[top_indices]
    
    # Describe feature impact direction on average (positive or negative)
    mean_shap_vals = np.mean(shap_values_cluster[:, top_indices], axis=0)
    
    # Summarize feature typical values in cluster
    summaries = []
    for i, feat in enumerate(top_features):
        # For numeric: describe mean +/- std, or quantiles
        if pd.api.types.is_numeric_dtype(df_cluster[feat]):
            mean_val = df_cluster[feat].mean()
            std_val = df_cluster[feat].std()
            direction = "increases" if mean_shap_vals[i] > 0 else "decreases"
            summaries.append(
                f"Feature '{feat}' with average value around {mean_val:.2f} {direction} churn risk"
            )
        else:
            # For categorical: most common value(s)
            mode_val = df_cluster[feat].mode().iloc[0]
            direction = "increases" if mean_shap_vals[i] > 0 else "decreases"
            summaries.append(
                f"Feature '{feat}' commonly set to '{mode_val}', which {direction} churn risk"
            )
    
    # Compose explanation based on plot type and cluster
    base_text = f"For the {cluster_name} churn risk cluster, "
    
    if plot_type == "beeswarm":
        explanation = base_text + "the beeswarm plot reveals how individual customers' feature values influence their churn risk. " + \
            "Key features include: " + "; ".join(summaries) + "."
    elif plot_type == "bar":
        explanation = base_text + "the bar chart ranks average feature importance for churn prediction. " + \
            "The most influential features are: " + "; ".join(summaries) + "."
    elif plot_type == "heatmap":
        explanation = base_text + "the heatmap visualizes SHAP value patterns across customers, highlighting feature impact consistency. " + \
            "Prominent features are: " + "; ".join(summaries) + "."
    else:
        explanation = base_text + "feature impacts vary; important features are: " + "; ".join(summaries) + "."
    
    return explanation,list(top_features)

import joblib
features_name = [
    "tenure", "monthlycharges", "totalcharges", "contract", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport",
    "streamingtv", "streamingmovies", "paymentmethod", "paperlessbilling",
    "partner", "dependents", "seniorcitizen"
]

# Save the feature list as 'trained_features.pkl'
joblib.dump(features_name, "trained_features.pkl")

print("✅ Feature list saved successfully as 'trained_features.pkl'")


import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load trained features and model
model = CatBoostClassifier()
# Load the saved model correctly
model.load_model("catboost_model1.cbm")
print("CatBoost is working!")


def preprocess_and_load_predict(file):

    # Read CSV file into DataFrame
    df = pd.read_csv(file)
    print("CSV File Loaded")
        # Convert all column names to lowercase to match the model's expectations
    df.columns = df.columns.str.lower()
    expected_features = set(features_name)
    uploaded_features = set(df.columns)

    missing = expected_features - uploaded_features
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {', '.join(missing)}")

    print("1")
    # Predict churn probabilities
    prediction_probabilities = model.predict_proba(df)[:, 1]
    predictions = ["Yes" if prob >= 0.5 else "No" for prob in prediction_probabilities]

    # Define custom clusters based on predicted probabilities
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ['Low', 'Medium', 'Medium-High', 'High']
    clusters = pd.cut(prediction_probabilities, bins=bins, labels=labels)

    # Create a DataFrame with the predicted probabilities and clusters
    predicted_labels = pd.DataFrame({'prediction': clusters})

    # Create a SHAP explainer
    explainer = shap.Explainer(model)

    # Get the SHAP values for the dataset
    shap_values = explainer.shap_values(df)

    # Create an Explanation object from the SHAP values
    shap_explanation = shap.Explanation(shap_values, base_values=explainer.expected_value, data=df)

    # Create SHAP sub-clusters based on custom clusters
    cluster_low_shap = shap_explanation[clusters == 'Low']
    cluster_medium_shap = shap_explanation[clusters == 'Medium']
    cluster_medhigh_shap = shap_explanation[clusters == 'Medium-High']
    cluster_high_shap = shap_explanation[clusters == 'High']

    # Beeswarm plot for each cluster
    def plot_beeswarm(shap_values_cluster, title):
        shap.summary_plot(shap_values_cluster.values, features=shap_values_cluster.data, 
                        feature_names=shap_values_cluster.feature_names, show=False)
        plt.title(title)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return img_b64

    # Beeswarm for all clusters
    img_low = plot_beeswarm(cluster_low_shap, 'SHAP values for Cluster Low')
    img_medium = plot_beeswarm(cluster_medium_shap, 'SHAP values for Cluster Medium')
    img_medhigh = plot_beeswarm(cluster_medhigh_shap, 'SHAP values for Cluster Medium-High')
    img_high = plot_beeswarm(cluster_high_shap, 'SHAP values for Cluster High')

    # Bar plot (mean SHAP values) for each cluster
    def plot_bar(shap_values_cluster, title):
        shap.summary_plot(shap_values_cluster.values, features=shap_values_cluster.data, 
                        feature_names=shap_values_cluster.feature_names, plot_type="bar", show=False)
        plt.title(title)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return img_b64

    # Bar plot (mean SHAP values) for all clusters
    img_bar_low = plot_bar(cluster_low_shap, 'Mean SHAP values for Cluster Low')
    img_bar_medium = plot_bar(cluster_medium_shap, 'Mean SHAP values for Cluster Medium')
    img_bar_medhigh = plot_bar(cluster_medhigh_shap, 'Mean SHAP values for Cluster Medium-High')
    img_bar_high = plot_bar(cluster_high_shap, 'Mean SHAP values for Cluster High')

    # Heatmap for each cluster
    def plot_heatmap(shap_values_cluster, title):
        sns.heatmap(shap_values_cluster.values, cmap='coolwarm')
        plt.title(title)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return img_b64

    # Heatmap for all clusters
    img_heatmap_low = plot_heatmap(cluster_low_shap, 'SHAP values for Cluster Low')
    img_heatmap_medium = plot_heatmap(cluster_medium_shap, 'SHAP values for Cluster Medium')
    img_heatmap_medhigh = plot_heatmap(cluster_medhigh_shap, 'SHAP values for Cluster Medium-High')
    img_heatmap_high = plot_heatmap(cluster_high_shap, 'SHAP values for Cluster High')


    # Extract dataframe subsets per cluster
    df_low = df[clusters == 'Low']
    df_medium = df[clusters == 'Medium']
    df_medhigh = df[clusters == 'Medium-High']
    df_high = df[clusters == 'High']

    # SHAP values numpy arrays per cluster
    shap_low_vals = cluster_low_shap.values
    shap_medium_vals = cluster_medium_shap.values
    shap_medhigh_vals = cluster_medhigh_shap.values
    shap_high_vals = cluster_high_shap.values

    # Generate explanations for all plot types and clusters
    explanation_low_beeswarm,top_low = generate_shap_explanation(df_low, shap_low_vals, "Low", "beeswarm")
    explanation_low_bar,top_low = generate_shap_explanation(df_low, shap_low_vals, "Low", "bar")
    explanation_low_heatmap,top_low = generate_shap_explanation(df_low, shap_low_vals, "Low", "heatmap")

    explanation_medium_beeswarm, top_medium = generate_shap_explanation(df_medium, shap_medium_vals, "Medium", "beeswarm")
    explanation_medium_bar, top_medium = generate_shap_explanation(df_medium, shap_medium_vals, "Medium", "bar")
    explanation_medium_heatmap, top_medium = generate_shap_explanation(df_medium, shap_medium_vals, "Medium", "heatmap")

    explanation_medhigh_beeswarm,top_medhigh = generate_shap_explanation(df_medhigh, shap_medhigh_vals, "Medium-High", "beeswarm")
    explanation_medhigh_bar,top_medhigh = generate_shap_explanation(df_medhigh, shap_medhigh_vals, "Medium-High", "bar")
    explanation_medhigh_heatmap,top_medhigh = generate_shap_explanation(df_medhigh, shap_medhigh_vals, "Medium-High", "heatmap")

    explanation_high_beeswarm,top_high = generate_shap_explanation(df_high, shap_high_vals, "High", "beeswarm")
    explanation_high_bar,top_high = generate_shap_explanation(df_high, shap_high_vals, "High", "bar")
    explanation_high_heatmap,top_high = generate_shap_explanation(df_high, shap_high_vals, "High", "heatmap")

    strategy_low = generate_retention_strategy("low", top_low)
    strategy_medium = generate_retention_strategy("medium", top_medium)
    strategy_medhigh = generate_retention_strategy("medium-high", top_medhigh)
    strategy_high = generate_retention_strategy("high", top_high)

    # Return the results with all generated images
    return {
        "predictions": predictions, 
        "probabilities": prediction_probabilities.tolist(),
        'img_low': img_low,
        'img_medium': img_medium,
        'img_medhigh': img_medhigh,
        'img_high': img_high,
        'img_bar_low': img_bar_low,
        'img_bar_medium': img_bar_medium,
        'img_bar_medhigh': img_bar_medhigh,
        'img_bar_high': img_bar_high,
        'img_heatmap_low': img_heatmap_low,
        'img_heatmap_medium': img_heatmap_medium,
        'img_heatmap_medhigh': img_heatmap_medhigh,
        'img_heatmap_high': img_heatmap_high,
        'explanations': {
        'low': {
            'beeswarm': explanation_low_beeswarm,
            'bar': explanation_low_bar,
            'heatmap': explanation_low_heatmap
        },
        'medium': {
            'beeswarm': explanation_medium_beeswarm,
            'bar': explanation_medium_bar,
            'heatmap': explanation_medium_heatmap
        },
        'medhigh': {
            'beeswarm': explanation_medhigh_beeswarm,
            'bar': explanation_medhigh_bar,
            'heatmap': explanation_medhigh_heatmap
        },
        'high': {
            'beeswarm': explanation_high_beeswarm,
            'bar': explanation_high_bar,
            'heatmap': explanation_high_heatmap
        }
    },
        'retention_strategies': {
        'low': strategy_low,
        'medium': strategy_medium,
        'medhigh': strategy_medhigh,
        'high': strategy_high
        }
    }
