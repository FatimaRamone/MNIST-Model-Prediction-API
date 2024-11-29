Handwritten Digit Prediction API with TensorFlow
Overview

This project is a real-time API that predicts handwritten digits (0-9) using a deep learning model built with TensorFlow. The model is trained on the famous MNIST dataset and exposed via an HTTP server. It receives POST requests with digit images, processes them, and returns the predicted digit class. Perfect for learning and integrating AI into web applications!
Features

    Deep Learning Model: Trained on the MNIST dataset using a Convolutional Neural Network (CNN).
    Real-Time Prediction: API processes POST requests with images of digits and returns predictions instantly.
    Simple HTTP Server: Built with Python’s http.server, ready to handle web requests.
    Normalized Input: Images are preprocessed to match the MNIST format for accurate predictions.

Requirements

Before running the project, ensure you have the following Python libraries installed:

    TensorFlow: For building and training the model.
    TensorFlow Datasets: For loading the MNIST dataset.
    NumPy: For data manipulation.
    Matplotlib: For visualizing data.
    Python 3.x

Install the necessary dependencies with:

pip install tensorflow tensorflow-datasets numpy matplotlib

Setup & Usage
1. Train the Model

The model is trained using the MNIST dataset. The training steps load and normalize the dataset, then train a simple CNN model to classify digits.

# Training the model
model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE)
)

2. Run the Server

To start the HTTP server that serves the prediction API, simply run:

python app.py

This will start the server on http://localhost:8000.
3. API Endpoints

    POST /predict: Send an image of size 28x28 pixels in a JSON format to get the predicted digit.

Example POST request:

{
  "data": [[0, 0, 0, ..., 0, 0, 0], ...]
}

Response:

{
  "prediction": "5"
}

    GET /: This serves the index.html page when accessed via the browser.

4. Predicting Digits

To send an image for prediction, send a POST request to /predict with an image in the proper format. The server will respond with the predicted digit class.
Folder Structure

/handwritten-digit-prediction-api
│
├── app.py               # Main script for training and API server
├── index.html           # Web page served on GET request
├── requirements.txt     # Project dependencies
└── model/               # Trained model files (saved during training)

Contributing

Feel free to fork this project and submit pull requests. Contributions to improve the model, API, or documentation are welcome!
License

This project is licensed under the MIT License. See LICENSE for more details.
Acknowledgments

    TensorFlow for providing the tools to build and train deep learning models.
    MNIST dataset for providing the benchmark of handwritten digits.
