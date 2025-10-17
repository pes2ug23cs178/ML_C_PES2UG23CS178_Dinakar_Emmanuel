# Speech Emotion Recognition using Deep Learning
###  Team No. 26
###  Collaborators: Cheruku Manas Ram (PES2UG23CS147), Dinakar Emmanuel (PES2UG23CS147)

A project to classify human emotions from audio speech signals using various deep learning architectures. This repository contains the scripts for feature extraction, model training, and a web application for live prediction.

## Key Features

- **Multi-Feature Extraction:** Extracts a set of acoustic features including MFCCs, Chroma, ZCR, RMS Energy, and Spectral Shape Features.
- **Data Augmentation:** Implements noise injection, pitch shifting, and time stretching to create a robust training dataset.
- **Multiple Model Architectures:** Includes implementations for:
    - 1D Convolutional Neural Networks (CNN)
    - Long Short-Term Memory (LSTM) networks
    - Hybrid CNN-LSTM models
- **Speaker-Independent Evaluation:** Utilizes a strict actor-independent train/test split to ensure the model generalises to unseen speakers.
- **Web UI:** A simple web application built with Streamlit for uploading an audio file and getting a real-time emotion prediction.

## Dataset

The project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

- **Total Files:** 1,440 speech files (+1,012 song files, if used).
- **Actors:** 24 professional actors (12 male, 12 female).
- **Emotions (8):** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
- **Train/Test Split:** A speaker-independent split was used, with 20 actors for training and 4 actors for testing.

You can download the datasets from https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio and https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-song-audio.

## Installation and Setup

## Usage


## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras:** For building and training deep learning models.
- **Librosa:** For audio processing and feature extraction.
- **Scikit-learn:** For data preprocessing (scaling, encoding) and evaluation metrics.
- **NumPy & Pandas:** For data manipulation.
- **Matplotlib & Seaborn:** For data visualization.
