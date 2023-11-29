# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the saved model
model = load_model('tea_leaf_disease_model.h5')

# Define class names
class_names = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

# Define image preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    return img

# Define image classification function
def classify_image(image):
    # Preprocess the image
    img_preprocessed = preprocess_image(image)
    img_4d = img_preprocessed.reshape(-1, 224, 224, 3)  # Add an extra dimension for batch size

    # Ensure the model output is a probability distribution using softmax
    predictions = model.predict(img_4d)[0]
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

    # Normalize probabilities to ensure they sum up to 100%
    probabilities /= np.sum(probabilities)

    # Map probabilities to class names
    predictions_dict = {class_names[i]: round(float(probabilities[i]), 4) for i in range(len(class_names))}

    return predictions_dict

# Streamlit app
def main():
    st.title("Tea Leaf Disease Classification App")
    st.write("Upload an image of a tea leaf to classify its disease.")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imread(uploaded_file.name)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Classify the image
        predictions = classify_image(image)

        # Display the predictions
        st.subheader("Predictions:")
        for disease, probability in predictions.items():
            st.write(f"{disease}: {probability}")

if __name__ == "__main__":
    main()
