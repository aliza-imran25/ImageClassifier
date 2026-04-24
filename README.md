AI Image Classifier using MobileNetV2 and Streamlit

This mini project implements an AI-based image classification web application that can automatically recognize objects in images. The system uses a pre-trained deep learning model (MobileNetV2) from the TensorFlow / Keras framework, which has been trained on the ImageNet dataset containing thousands of object categories. The application is deployed through an interactive web interface built using Streamlit, allowing users to easily upload images and obtain predictions in real time.

The system works by first loading the MobileNetV2 deep learning model, which is optimized for fast and efficient image classification. When a user uploads an image, the program preprocesses it by resizing it to 224 × 224 pixels, converting it into a numerical array, and applying normalization using the model’s preprocessing function. The processed image is then passed to the neural network, which analyzes visual patterns and features in the image.

After the model processes the image, it generates prediction probabilities for different object categories. These predictions are decoded into human-readable labels using the ImageNet class labels, and the application displays the top three most likely predictions along with their confidence scores. To improve performance, the model is cached so that it does not reload every time the application runs.

The user interface is simple and interactive. Users can upload images in JPG or PNG format, preview the uploaded image, and click the “Classify Image” button to start the analysis. The system then shows the predicted object categories with corresponding confidence percentages.

This project demonstrates the integration of computer vision, deep learning, and web application development. It highlights how pre-trained neural networks can be used to quickly build intelligent systems capable of understanding visual data.
