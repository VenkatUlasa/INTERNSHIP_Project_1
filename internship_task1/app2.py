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

# Define paths for the models
MODEL_PATHS = {
    "VGG16": "VGG16_model.h5",
    "ResNet50": "ResNet_model.h5",
    "CustomModel": "custom_model.h5"
}

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Load model performance metrics
with open("metrics.json", "r") as f:
    model_metrics = json.load(f)

# Load food properties (nutritional information)
with open("Food_data.json", "r") as f:
    food_properties = json.load(f)

# Create a folder for uploaded images
UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to load model dynamically
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path and os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None


# Function to process image and make prediction
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Resize as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize

    # Predict using the selected model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get class label
    class_name = class_labels[predicted_class]

    return class_name


# Flask route for image upload and prediction
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        model_name = request.form.get("model_name")  # Get selected model
        if model_name not in MODEL_PATHS:
            return render_template("index.html", error="Invalid model selected", model_metrics=model_metrics)

        # Load selected model
        model = load_model(model_name)
        if model is None:
            return render_template("index.html", error="Model not found", model_metrics=model_metrics)

        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded", model_metrics=model_metrics)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file", model_metrics=model_metrics)

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Get prediction
        predicted_label = predict_image(model, file_path)

        # Get food properties
        food_info = food_properties.get(predicted_label, {})

        return render_template("index.html", prediction=predicted_label, image_path=file_path, food_info=food_info,
                               model_metrics=model_metrics)

    return render_template("index.html", model_metrics=model_metrics)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
