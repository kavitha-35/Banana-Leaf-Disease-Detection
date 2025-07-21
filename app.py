from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

img_size = (128, 128)
model = load_model('model/Banana Disease Recognition_model.h5')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    classes = [
        'Banana Black Sigatoka Disease',
        'Banana Bract Mosaic Virus Disease',
        'Banana Healthy Leaf',
        'Banana Insect Pest Disease',
        'Banana Moko Disease',
        'Banana Panama Disease',
        'Banana Yellow Sigatoka Disease'
    ]
    class_name = classes[class_index]

    classes_list = [
        'Black Sigatoka ',
        'Bract Mosaic Virus',
        'Healthy Leaf',
        ' Insect Pest ',
        ' Moko ',
        ' Panama ',
        ' Yellow Sigatoka '
    ]

    return jsonify({
        'Class_names': class_name,
        'prediction': prediction[0].tolist(),
        'labels': classes_list,
    })

if __name__ == '__main__':
    app.run(debug=True)
