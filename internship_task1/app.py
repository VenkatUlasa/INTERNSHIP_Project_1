import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import json
import PIL

# Initialize Flask app
app = Flask(__name__)

# Load the trained VGG16 model
MODEL_PATH = "VGG16_model.h5"  # Ensure correct path
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Create a folder for uploaded images
UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to process image and make prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Resize as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get class label
    class_name = class_labels[predicted_class]

    return class_name


# Flask route for image upload and prediction
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Get prediction
        predicted_label = predict_image(file_path)

        return render_template("index.html", prediction=predicted_label, image_path=file_path)

    return render_template("index.html")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

