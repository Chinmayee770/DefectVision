import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load models
custom_cnn_model = load_model('casting_product_detection_normal.hdf5')
vgg16_model = load_model('casting_product_detection_vgg16.hdf5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')  # Save to static/uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224)) / 255.0
    return img.reshape(1, 224, 224, 3)

def create_confidence_graph(custom_confidence, vgg16_confidence):
    fig, ax = plt.subplots(figsize=(5, 3))
    models = ['Custom CNN', 'VGG16']
    confidence_values = [custom_confidence, vgg16_confidence]
    ax.bar(models, confidence_values, color=['blue', 'green'])
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence Comparison')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Prepare the image for prediction
            img = prepare_image(filepath)

            # Get predictions from both models
            custom_cnn_pred = custom_cnn_model.predict(img)[0][0]
            vgg16_pred = vgg16_model.predict(img)[0][0]

            # Calculate confidence for each model
            custom_confidence = custom_cnn_pred * 100 if custom_cnn_pred >= 0.5 else (1 - custom_cnn_pred) * 100
            vgg16_confidence = vgg16_pred * 100 if vgg16_pred >= 0.5 else (1 - vgg16_pred) * 100

            # Generate a graph for confidence
            confidence_graph = create_confidence_graph(custom_confidence, vgg16_confidence)

            # Determine final classification based on both models
            final_result = "Non-Defective" if vgg16_pred >= 0.5 else "Defective"

            # Render the result template with predictions and confidence scores
            return render_template('result.html', 
                                   image_url=url_for('uploaded_file', filename=filename),
                                   final_result=final_result,
                                   custom_confidence=custom_confidence,
                                   vgg16_confidence=vgg16_confidence,
                                   confidence_graph=confidence_graph)

    return render_template('upload.html')


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)
