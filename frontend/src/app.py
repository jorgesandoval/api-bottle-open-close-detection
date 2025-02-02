# frontend/src/app.py
import streamlit as st
import requests
import time
from PIL import Image
import io

# API Configuration
BACKEND_URL = "http://backend:5000"  # Using Docker service name


def check_health():
    """Check backend health status"""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            st.success("Backend service is healthy! ✅")
        else:
            st.error("Backend service is not responding correctly! ❌")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend service! ❌")


def classify_image(image_file):
    """Send image to backend for classification"""
    try:
        # Create files dictionary for request
        files = {'image': ('image.jpg', image_file.getvalue(), 'image/jpeg')}

        # Make request to backend
        with st.spinner('Classifying image...'):
            response = requests.post(f"{BACKEND_URL}/api/classify", files=files)

        if response.status_code == 200:
            result = response.json()

            # Display results
            st.success("Classification Complete!")

            # Create two columns
            col1, col2 = st.columns(2)

            # Display results in columns
            with col1:
                st.metric("Bottle Status", result['Bottle_Status'])
            with col2:
                st.metric("Confidence", f"{result['Confidence']:.2%}")

        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


def main():
    st.title("🍾 Bottle Status Classifier")
    st.write("Upload an image to check if a bottle is open or closed!")

    # Add health check button
    if st.button("Check Backend Health"):
        check_health()

    # Add separator
    st.divider()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter here

        # Add classify button
        if st.button('Classify Image'):
            classify_image(uploaded_file)


if __name__ == "__main__":
    main()