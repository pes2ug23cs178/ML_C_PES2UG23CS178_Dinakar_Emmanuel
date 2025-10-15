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

Follow these steps to set up the project environment.
**1. Clone the Repository and go to the folder**

**2. Create a Python Virtual Environment**
It's highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install Dependencies**
Install all the required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Setup the Dataset**
- Download the RAVDESS dataset.
- Unzip the `Audio_Speech_Actors_01-24.zip` file.
- Download the RAVDESS song dataset.
- Unzip the `Audio_Speech_01-24.zip` file.
- Place the unzipped folders inside a `datasets` directory in the root of the project.

## Usage

The project is divided into three main steps: feature extraction, model training, and running the UI.

**Step 1: Generate Features**
Run the feature generation script. This will process the raw audio files and create the `.npy` feature files needed for training.
```bash
python3 feature_extraction_augmentation.py
```

**Step 2: Train a Model**
Choose one of the training scripts to run. For example, to train the 1D CNN model:
Go to the cnn.ipynb and run the latest code in the notebook. 

This will train the model and save the final `.h5` model file, along with the `scaler.pkl` and `encoder_classes.npy` files required for the UI.

**Step 3: Launch the UI**
Make sure the latest `.h5`, `.pkl`, and `.npy` files are in the same directory as `app.py`. Then, run the following command:
```bash
streamlit run app.py
```
This will open a web page in your browser where you can upload a `.wav` file and test the model.

## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras:** For building and training deep learning models.
- **Streamlit:** For creating the interactive web UI.
- **Librosa:** For audio processing and feature extraction.
- **Scikit-learn:** For data preprocessing (scaling, encoding) and evaluation metrics.
- **NumPy & Pandas:** For data manipulation.
- **Matplotlib & Seaborn:** For data visualization.
