from django.urls import path
from .views import single_predict_view, get_shap_explanation, batch_predict, single_input_page, batch_input_page, shiny_dashboard, Home, get_img_base64, shap_explanation_page
urlpatterns = [
    path("single-predict/", single_predict_view, name="single_predict"),
    path("batch-predict/", batch_predict, name="batch_predict"),
    path("get-shap-explanation/", get_shap_explanation, name="get_shap_explanation"),
    path("single-input/", single_input_page, name="single_input"), #html
    path("dataset-predict/", batch_input_page, name="dataset_predict"),
    path("Home/", Home, name="Home"), #html
    path("shiny_dashboard/", shiny_dashboard, name="shiny_dashboard"),  # New Dashboard URL
    path("get_img_base64/", get_img_base64, name="get_img_base64"),
    path("shap-explanation/", shap_explanation_page, name="shap_explanation"),
]
