# Food Classification Using CNN

## 📌 Project Objective
The goal of this project is to build a deep learning-based food classification system that can accurately classify food images into 34 categories using multiple deep learning models.

## 🚀 Key Highlights
- Dataset balancing and preprocessing.
- Training three different models: **Custom CNN, VGG16, and ResNet**.
- Model validation using accuracy, precision, recall, and F1-score.
- **Flask-based web application** for real-time predictions.
- JSON-based nutritional information storage for food items (`Food_data.json`).
- Well-structured GitHub repository and project documentation.

---

## 📂 Dataset Details
- **Total Classes:** 34 food categories.
- **Dataset Source:** [Food Image Classification Dataset] (Add link if available).
- **Preprocessing Steps:**
  - Balance the dataset by selecting an equal number of images per class.
  - Use **200 images per class** to ensure uniform distribution.
  - Upload selected images to Google Drive for easy access.

---

## 🏗️ Model Development & Training
We train three different deep learning models and evaluate their performance:

### 1️⃣ **Custom Deep Learning Model**
- **Architecture:** Uses pre-trained **ResNet** with fine-tuning.
- **Training Parameters:**
  - **Epochs:** 30  
  - **Optimizer:** Adam  
- **Performance Metrics:**  
  - Accuracy, Precision, Recall, F1-Score  
- **Validation Report:** `CustomModel_validation_report.json`

### 2️⃣ **VGG16 Model (Transfer Learning)**
- **Architecture:** Pre-trained VGG16 with fine-tuning.
- **Training Parameters:**
  - **Epochs:** 30  
  - **Optimizer:** Adam  
- **Performance Metrics:**  
  - Accuracy, Precision, Recall, F1-Score  
- **Validation Report:** `VGG16_validation_report.json`

### 3️⃣ **ResNet Model (Transfer Learning)**
- **Architecture:** Pre-trained ResNet with fine-tuning.
- **Training Parameters:**
  - **Epochs:** 30  
  - **Optimizer:** Adam  
- **Performance Metrics:**  
  - Accuracy, Precision, Recall, F1-Score  
- **Validation Report:** `ResNet_validation_report.json`

---

## 🛠️ Coding Standards & Best Practices
- **Object-Oriented Programming (OOP):** Classes, Objects, and Constructors for structured code.
- **Exception Handling:** `try-except` blocks for smooth error handling.
- **Code Readability & Efficiency:** Follow Python best practices.
- **Development Environment:** PyCharm IDE.

---

## 📊 Data Collection & JSON Creation
- **Nutritional Information Attributes:**
  - Protein (g)
  - Fiber (g)
  - Calories (kcal)
  - Carbohydrates (g)
  - Fat (g)
- **Storage:** All data is stored in `Food_data.json`.

---

## 🌍 Model Deployment (Flask Web App)
### 🎨 **Frontend Features**
- **User Input:**
  - Upload an image for classification.
  - Select a trained model (**Custom CNN, VGG16, or ResNet**).

### ⚙️ **Backend Processing (Flask API)**
- Dynamically loads the selected model.
- Processes the uploaded image and classifies it.
- Returns the classification results.

### 📊 **Output Display**
- **Predicted Food Class**
- **Nutritional Information:**
  - Protein, Fiber, Calories, Carbohydrates, Fat.

---

## 💻 Installation & Usage
### 🔧 **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/Food-Classification-CNN.git
   cd Food-Classification-CNN


## 📞 Contact Me
📧 Email: josh8008venkat139@gmail.com
