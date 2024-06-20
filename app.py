from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import threading
import time
import warnings
import joblib
import os
import base64
import io
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

app = Flask(__name__)

# Initialisation de Mediapipe Holistic et des utilitaires de dessin
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Variables globales
data = []
trained_model = None
is_predicting = False
is_capturing = False
current_class = ""
num_samples = 0
samples_captured = 0
capturing_complete = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    global data, is_capturing, num_samples, samples_captured, current_class, is_predicting, trained_model, capturing_complete

    try:
        # Récupérer les données de l'image depuis la requête POST
        image_data = request.json['image']
        image_data = image_data.split(',')[1]  # Supprimer le préfixe de l'URL data:image/jpeg;base64,
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)

        # Convertir l'image en RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            results = holistic.process(frame_rgb)

            if results.right_hand_landmarks or results.left_hand_landmarks:
                landmarks = []

                if results.right_hand_landmarks:
                    for landmark in results.right_hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                if results.left_hand_landmarks:
                    for landmark in results.left_hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                if is_capturing and samples_captured < num_samples:
                    data.append([current_class] + landmarks)
                    samples_captured += 1

                if is_predicting and trained_model:
                    columns = [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
                    input_data = pd.DataFrame([landmarks], columns=columns)
                    prediction = trained_model.predict(input_data)[0]
                    return jsonify({'prediction': prediction})

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error processing video frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global is_capturing, current_class, num_samples, samples_captured, data, capturing_complete

    try:
        capture_info = request.get_json()
        num_samples = int(capture_info['num_samples'])
        class_names = capture_info['class_names']

        for class_name in class_names:
            current_class = class_name
            samples_captured = 0
            is_capturing = True
            while samples_captured < num_samples:
                time.sleep(0.1)
            is_capturing = False

        return jsonify({'message': 'Capture completed.', 'success': True})
    except Exception as e:
        print(f"Error in start_capture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    global trained_model

    # Convertir les données en DataFrame et entraîner le modèle directement à partir de data
    columns = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    X = df.drop('label', axis=1)
    y = df['label']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    trained_model = model
    
    # Enregistrer le modèle dans un fichier .h5
    joblib.dump(trained_model, 'modele_decision_tree.h5')

    return jsonify({'message': 'Model trained successfully.', 'success': True})

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    global is_predicting
    is_predicting = True
    return jsonify({'message': 'Prediction started.', 'success': True})

@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    global is_predicting
    is_predicting = False
    return jsonify({'message': 'Prediction stopped.', 'success': True})

@app.route('/download_data', methods=['GET'])
def download_data():
    columns = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    file_path = 'hand_gestures.csv'
    df.to_csv(file_path, index=False)

    return send_file(file_path, mimetype='text/csv', download_name='hand_gestures.csv', as_attachment=True)

@app.route('/download_model', methods=['GET'])
def download_model():
    return send_file('modele_decision_tree.h5', mimetype='application/octet-stream', download_name='modele_decision_tree.h5', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
