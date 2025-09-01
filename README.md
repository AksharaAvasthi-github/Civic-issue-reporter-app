# Civic Issue Reporter App

This is a full-stack web application built using Flask (backend) and Jinja (frontend) to help users report civic issues such as garbage or potholes. It leverages machine learning to classify issues based on user input.

## Features

- User-friendly interface to submit civic issue reports.
- Backend APIs to handle data processing and storage.
- Machine learning model to classify images of garbage and potholes (model file excluded due to size).
- Data visualization and dashboard for admins (if applicable).

## Tech Stack

- **Backend:** Flask
- **Frontend:** Jinja templating
- **Machine Learning:** Python (TensorFlow/Keras)
- **Others:** HTML, CSS, JavaScript

## Why the Model File is Excluded

The trained model file (`garbage_vs_pothole_cnn.h5`) is approximately 137 MB, which exceeds GitHub's file size limit of 100 MB. Therefore, it is not included in this repository to keep it lightweight and easy to clone.

If you'd like to run the app with the model, please contact me or download the model from [provide alternative link if available], and place it inside the `/ml` directory.

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/AksharaAvasthi-github/Civic-issue-reporter-app.git
