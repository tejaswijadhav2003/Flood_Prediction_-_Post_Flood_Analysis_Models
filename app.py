from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import cv2
import numpy as np
import os

app = Flask(__name__)


# Load the trained model
model = keras.models.load_model('./model/flood_detection_trained_model.h5')

# Define the image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['image']
        
        # Save the image temporarily
        image_path = 'temp_image.jpg'
        file.save(image_path)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

         # Make prediction
        logits = model.predict(processed_image)
        probability = 1 / (1 + np.exp(-logits))

        # Clean up temporary image file
        os.remove(image_path)

        # Determine the prediction based on the probability
        prediction = "Flooded" if probability >= 0.5 else "Not Flooded"

        # Return the prediction as JSON
        return jsonify({'prediction': prediction, 'probability': float(probability), 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failure'})


    

   

#  for local
if __name__ == "__main__":
    app.run(debug=True,port=8080)

#  for cloud
# if __name__ == "__main__":
#     app.run(host = '0.0.0.0',port=8080)