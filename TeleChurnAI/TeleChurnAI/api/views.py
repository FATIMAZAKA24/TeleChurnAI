from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from .ml_model.single_predict import single_predict
from .ml_model.dataset_predict import preprocess_and_load_predict
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
import pandas as pd
import copy
import base64
import uuid

def get_img_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@api_view(["POST"])
def single_predict_view(request):
    try:
        # Extract input data from the POST request
        input_data = request.data
        print(request)
        # Call the prediction function
        result = single_predict(input_data)
        return Response(result)
       # return JsonResponse(result, safe=False) 

    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

def generate_unique_session_id():
    # Generate a random unique string as session ID
    return str(uuid.uuid4())

@api_view(['POST'])
@csrf_exempt
def batch_predict(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    csv_file = request.FILES['file']

    try:
        import io
        csv_string = csv_file.read().decode('utf-8')
        df_buffer = io.StringIO(csv_string)

        predictions = preprocess_and_load_predict(df_buffer)

        session_id = generate_unique_session_id()

        # Store in Django cache (e.g., Redis)
        cache.set(session_id, {
            "dataset": csv_string,
            "results": predictions,
        }, timeout=60 * 60)  # Optional: 1 hour expiration

        return JsonResponse({'predictions': predictions, 'session_id': session_id})

    except Exception as e:
        print("Error during batch prediction:", str(e))
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def get_shap_explanation(request):
    session_id = request.GET.get('session_id')
    cluster = request.GET.get('cluster')  # e.g., 'low', 'medium', etc.
    plot_type = request.GET.get('plot_type')  # 'beeswarm', 'bar', 'heatmap'

    session_data = cache.get(session_id)

    if session_data is None:
        return JsonResponse({'error': 'Invalid session'}, status=400)

    results = session_data.get("results")

    # Determine the correct image key
    image_key = f'img_{cluster}'
    if plot_type == 'bar':
        image_key = f'img_bar_{cluster}'
    elif plot_type == 'heatmap':
        image_key = f'img_heatmap_{cluster}'

    plot_base64 = results.get(image_key)
    explanation = results.get('explanations', {}).get(cluster, {}).get(plot_type)

    # 💡 New: Get retention strategy for this cluster
    strategy = results.get('retention_strategies', {}).get(cluster)

    if not plot_base64 or not explanation:
        return JsonResponse({'error': 'Plot or explanation not found'}, status=404)

    return JsonResponse({
        'plot_base64': plot_base64,
        'explanation': explanation,
        'strategy': strategy or "No strategy available."
    })

def Home(request):
    return render(request, "Home.html")

def single_input_page(request):
    return render(request, "single_predict.html")

def batch_input_page(request):
    return render(request, "dataset_predict.html")

def shap_explanation_page(request):
    return render(request, 'shap_explanation.html')

def shiny_dashboard(request):
    return render(request, "shiny_dashboard.html")
