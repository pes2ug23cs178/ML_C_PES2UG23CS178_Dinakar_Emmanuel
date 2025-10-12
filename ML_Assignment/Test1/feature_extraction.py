import os
import glob
import shutil
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_PATH = '/home/manas/Desktop/ML_SpeechEmotion/datasets/audio_speech_actors_01-24'
GENERATE_SEQUENCES = True     # 1D CNN / LSTM

# Output Directory
SEQUENCE_PATH = 'sequences' # .npy files

# Feature Parameters
# For Sequences
N_MFCC = 13
N_CHROMA = 12
MAX_PAD_LEN = 174


def load_and_split_data(dataset_path):
    if not os.path.exists(dataset_path): raise ValueError(f"Dataset path not found at '{dataset_path}'")
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    file_details = []
    for file_path in glob.glob(os.path.join(dataset_path, 'Actor_*', '*.wav')):
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('.')[0].split('-')
            emotion_code = parts[2]; actor_id = int(parts[6])
            file_details.append({'file_path': file_path, 'emotion_label': emotion_map[emotion_code], 'actor_id': actor_id})
        except (IndexError, KeyError): continue
    df = pd.DataFrame(file_details)
    all_actors = sorted(df['actor_id'].unique())
    train_actors, test_actors = all_actors[:-4], all_actors[-4:]
    return df[df['actor_id'].isin(train_actors)], df[df['actor_id'].isin(test_actors)]

def extract_mfcc_chroma_sequence(file_path, max_pad_len=MAX_PAD_LEN):
    """Extracts MFCC and Chroma features, combines, and pads them."""
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)
        combined = np.vstack([mfccs, chroma])
        
        if combined.shape[1] > max_pad_len: combined = combined[:, :max_pad_len]
        else: combined = np.pad(combined, ((0, 0), (0, max_pad_len - combined.shape[1])), mode='constant')
        
        return combined.T # Transpose to (time_steps, features)
    except Exception as e:
        print(f"Error extracting sequence for {file_path}: {e}")
        return None

# 4. MAIN EXECUTION
if __name__ == "__main__":
    train_df, test_df = load_and_split_data(DATASET_PATH)

    # --- PATH B: SEQUENCE GENERATION ---
    if GENERATE_SEQUENCES:
        print("\n--- Generating MFCC+Chroma Sequences ---")
        os.makedirs(SEQUENCE_PATH, exist_ok=True)
        
        X_train, y_train = [], []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Sequences"):
            seq = extract_mfcc_chroma_sequence(row['file_path'])
            if seq is not None:
                X_train.append(seq)
                y_train.append(row['emotion_label'])

        X_test, y_test = [], []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Sequences"):
            seq = extract_mfcc_chroma_sequence(row['file_path'])
            if seq is not None:
                X_test.append(seq)
                y_test.append(row['emotion_label'])

        # Save to .npy files
        np.save(os.path.join(SEQUENCE_PATH, 'X_train_seq.npy'), np.array(X_train))
        np.save(os.path.join(SEQUENCE_PATH, 'y_train_seq.npy'), np.array(y_train))
        np.save(os.path.join(SEQUENCE_PATH, 'X_test_seq.npy'), np.array(X_test))
        np.save(os.path.join(SEQUENCE_PATH, 'y_test_seq.npy'), np.array(y_test))
        print("Sequence generation complete.")