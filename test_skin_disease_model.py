import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("skin_disease_model.h5")  # Make sure the model is saved after training

# Define class labels (update based on your dataset)
class_labels = ["Eczema", "Psoriasis", "Melanoma", "Acne", "Vitiligo", "Ringworm", "Normal Skin"]

def preprocess_image(img_path):
    """Loads and preprocesses an image for model prediction"""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize (if needed)
    return img_array

def predict_image(img_path):
    """Predict the disease for a single image"""
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get index of highest probability
    predicted_disease = class_labels[predicted_class]
    
    print(f"Image: {img_path} → Predicted Disease: {predicted_disease}")
    return predicted_disease

# **🔹 TEST ON A SINGLE IMAGE**
test_image = r"E:\Mini Project\test_images\test_eczema1.jpg"  # Change to your image path
predict_image(test_image)

# **🔹 TEST ON MULTIPLE IMAGES IN A FOLDER**
test_folder = r"E:\Mini Project\test_images"  # Folder containing test images

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    if img_path.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image files
        predict_image(img_path)
