from flask import Blueprint, jsonify, request
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.main.utils import calculate_bmr, calculate_tdee

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    # get request
    return 'Hello World!'

@main.route('/predict', methods=['POST'])
def predict():
    # post request
    data = request.get_json()

    # json body should include height, weight, age, gender, and activity_factor
    # data validation
    if 'height' not in data:
        return serialize_error('height is required'), 400
    if 'weight' not in data:
        return serialize_error('weight is required'), 400
    if 'age' not in data:
        return serialize_error('age is required'), 400
    if 'gender' not in data:
        return serialize_error('gender is required'), 400
    if 'activity_factor' not in data:
        return serialize_error('activity_factor is required'), 400
    

    # get data
    height = data['height']
    weight = data['weight']
    age = data['age']
    gender = data['gender']
    activity_factor = data['activity_factor']

    # Kalori Excedded calculation (The first user use the app)
    estimated_kalori = calculate_bmr(weight, height, age, gender)
    kalori_user = calculate_tdee(estimated_kalori, activity_factor)

    kalori_excess = kalori_user - estimated_kalori

    # Machine Learning
    models_dir = os.path.join(os.getcwd(), 'models')
    data_file_path = os.path.join(models_dir, 'Exercise_Output_5.csv')

    # Load the ensemble models
    loaded_models = []
    num_models = 3
    for i in range(num_models):
        model_path = f'Model/ensemble_model2_{i + 1}.h5'
        model_path = os.path.join(models_dir, f'ensemble_model2_{i + 1}.h5')
        loaded_model = load_model(model_path)
        loaded_models.append(loaded_model)

    dataset=pd.read_csv(data_file_path,index_col=0)

    # Reshape the input to (1, 1)
    kalori_input = np.array([[kalori_excess]])

    # Initialize ensemble probabilities
    ensemble_probabilities = np.zeros((1, 20))
    # Make predictions using each loaded model
    for model in loaded_models:
        predictions = model.predict(kalori_input)
        ensemble_probabilities += predictions
    flat_data = ensemble_probabilities[0]
    max_index = np.argmax(flat_data)
    value = np.random.rand(20)
    value[max_index] = 1.0

    # Create a list of tuples containing label and probability
    label_prob_tuples = [(label, probability) for label, probability in enumerate(value)]


    # Sorting
    # Sort the list based on probabilities in descending order
    sorted_label_prob_tuples = sorted(label_prob_tuples, key=lambda x: x[1], reverse=True)

    # Extract only the labels from the sorted list of tuples
    sorted_labels = [label for label, _ in sorted_label_prob_tuples]
    # label_5=sorted_labels[:5]
    # Convert label_5 to a pandas Series for comparison
    label_5_series = pd.Series(sorted_labels[0:5])

    # Use loc to filter rows where 'label' is in label_5_series
    selected_rows = dataset.loc[dataset['label'].isin(label_5_series), ['label', 'activity']]
    selected_rows.sort_values(by='label',ascending=True,inplace=True)

    result = selected_rows.drop_duplicates(subset=['label'])

    return result.to_json(orient='records'), 200, {'Content-Type':'application/json'}


def serialize_error(error: str):
    return {
        'error': error
    }