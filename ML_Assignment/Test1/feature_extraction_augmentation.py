# import os
# import glob
# import pandas as pd
# import numpy as np
# import librosa
# from tqdm import tqdm

# DATASET_PATH = '/home/manas/Desktop/ML_SpeechEmotion/datasets/audio_speech_actors_01-24'
# SEQUENCE_PATH = 'sequences'
# N_MFCC = 13
# N_CHROMA = 12
# MAX_PAD_LEN = 174

# def load_and_split_data(dataset_path):
#     if not os.path.exists(dataset_path): raise ValueError(f"Dataset path not found at '{dataset_path}'")
#     emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
#     file_details = []
#     for file_path in glob.glob(os.path.join(dataset_path, 'Actor_*', '*.wav')):
#         try:
#             filename = os.path.basename(file_path)
#             parts = filename.split('.')[0].split('-')
#             emotion_code = parts[2]; actor_id = int(parts[6])
#             file_details.append({'file_path': file_path, 'emotion_label': emotion_map[emotion_code], 'actor_id': actor_id})
#         except (IndexError, KeyError): continue
#     df = pd.DataFrame(file_details)
#     all_actors = sorted(df['actor_id'].unique())
#     train_actors, test_actors = all_actors[:-4], all_actors[-4:]
#     return df[df['actor_id'].isin(train_actors)], df[df['actor_id'].isin(test_actors)]

# def add_noise(data, noise_factor=0.005):
#     """Adds random Gaussian noise to the audio signal."""
#     noise = np.random.randn(len(data))
#     return data + noise_factor * noise

# def pitch_shift(data, sr, n_steps):
#     """Shifts the pitch of the audio signal."""
#     return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

# def time_stretch(data, rate):
#     """Stretches the time of the audio signal."""
#     return librosa.effects.time_stretch(y=data, rate=rate)

# def extract_features(file_path, use_augmentation=False, max_pad_len=MAX_PAD_LEN):
#     """
#     Extracts a rich set of features: MFCC, Chroma, ZCR, and RMS Energy.
#     Applies random augmentation if specified.
#     """
#     try:
#         audio, sr = librosa.load(file_path, res_type='kaiser_fast')
      
#         if use_augmentation:
#             # Apply one of the augmentations randomly
#             choice = np.random.randint(0, 4)
#             if choice == 1:
#                 audio = add_noise(audio)
#             elif choice == 2:
#                 n_steps = np.random.randint(-2, 3) 
#                 audio = pitch_shift(audio, sr, n_steps)
#             elif choice == 3:
#                 rate = np.random.uniform(0.8, 1.2)
#                 audio = time_stretch(audio, rate)

#         mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)
#         zcr = librosa.feature.zero_crossing_rate(y=audio)
#         rms = librosa.feature.rms(y=audio)
      
#         # Stack all features vertically
#         combined_features = np.vstack([mfccs, chroma, zcr, rms])
      
#         # Pad or truncate to a fixed length
#         if combined_features.shape[1] > max_pad_len:
#             combined_features = combined_features[:, :max_pad_len]
#         else:
#             pad_width = max_pad_len - combined_features.shape[1]
#             combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
      
#         # Transpose to get the shape (time_steps, features)
#         return combined_features.T
  
#     except Exception as e:
#         print(f"Error extracting features for {file_path}: {e}")
#         return None

# if __name__ == "__main__":
#     train_df, test_df = load_and_split_data(DATASET_PATH)
#     os.makedirs(SEQUENCE_PATH, exist_ok=True)
  
#     # Process training data with AUGMENTATION
#     print("\n--- Generating Augmented Training Sequences ---")
#     X_train, y_train = [], []
#     for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training Set"):
#         seq = extract_features(row['file_path'], use_augmentation=True)
#         if seq is not None:
#             X_train.append(seq)
#             y_train.append(row['emotion_label'])
#     # Process testing data WITHOUT AUGMENTATION
#     print("\n--- Generating Clean Testing Sequences ---")
#     X_test, y_test = [], []
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing Set"):
#         seq = extract_features(row['file_path'], use_augmentation=False) # NO AUGMENTATION
#         if seq is not None:
#             X_test.append(seq)
#             y_test.append(row['emotion_label'])
          
#     # Save to .npy files
#     np.save(os.path.join(SEQUENCE_PATH, 'X_train_seq.npy'), np.array(X_train))
#     np.save(os.path.join(SEQUENCE_PATH, 'y_train_seq.npy'), np.array(y_train))
#     np.save(os.path.join(SEQUENCE_PATH, 'X_test_seq.npy'), np.array(X_test))
#     np.save(os.path.join(SEQUENCE_PATH, 'y_test_seq.npy'), np.array(y_test))
#     print("\nSequence generation with augmentation complete.")





import os
import glob
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

DATASET_PATH = '/home/manas/Desktop/ML_SpeechEmotion (Copy)/datasets'
SEQUENCE_PATH = 'sequences'
N_MFCC = 13
N_CHROMA = 12
MAX_PAD_LEN = 174

def load_and_split_data(base_path):
    """
    Scans both speech and song directories, parses filenames, and returns a DataFrame.
    """
    subfolders = ['audio_speech_actors_01-24', 'audio_song_actors_01-24']
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    file_details = []
    print(f"Scanning for audio files in: {base_path}")
    for folder in subfolders:
        full_path = os.path.join(base_path, folder)
        if not os.path.exists(full_path):
            print(f"Warning: Directory not found at '{full_path}'. Skipping.")
            continue
        
        print(f" -> Processing folder: {folder}")
        for file_path in glob.glob(os.path.join(full_path, 'Actor_*', '*.wav')):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('.')[0].split('-')
                emotion_code = parts[2]
                actor_id = int(parts[6])

                # The 'calm' emotion (02) only exists in the speech dataset.
                # Skip it if found in the song folder to maintain consistency.
                if folder == 'audio_song_actors_01-24' and emotion_code == '02':
                    continue
                
                file_details.append({'file_path': file_path, 'emotion_label': emotion_map[emotion_code], 'actor_id': actor_id})
            except (IndexError, KeyError):
                continue

    if not file_details:
        raise ValueError("No audio files found. Check your DATASET_PATH and ensure subfolders exist.")
        
    df = pd.DataFrame(file_details)
    all_actors = sorted(df['actor_id'].unique())
    train_actors, test_actors = all_actors[:-4], all_actors[-4:]
    return df[df['actor_id'].isin(train_actors)], df[df['actor_id'].isin(test_actors)]

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def pitch_shift(data, sr, n_steps):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate):
    return librosa.effects.time_stretch(y=data, rate=rate)

# def extract_features(file_path, use_augmentation=False, max_pad_len=MAX_PAD_LEN):
#     try:
#         audio, sr = librosa.load(file_path, res_type='kaiser_fast')
#         if use_augmentation:
#             choice = np.random.randint(0, 4)
#             if choice == 1: audio = add_noise(audio)
#             elif choice == 2: audio = pitch_shift(audio, sr, np.random.randint(-2, 3))
#             elif choice == 3: audio = time_stretch(audio, np.random.uniform(0.8, 1.2))

#         mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)
#         zcr = librosa.feature.zero_crossing_rate(y=audio)
#         rms = librosa.feature.rms(y=audio)
#         combined_features = np.vstack([mfccs, chroma, zcr, rms])
        
#         if combined_features.shape[1] > max_pad_len: combined_features = combined_features[:, :max_pad_len]
#         else: combined_features = np.pad(combined_features, ((0, 0), (0, max_pad_len - combined_features.shape[1])), mode='constant')
        
#         return combined_features.T
#     except Exception as e:
#         print(f"Error extracting features for {file_path}: {e}")
#         return None

def extract_features(file_path, use_augmentation=False, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        if use_augmentation:
            # (Your augmentation logic remains here)
            choice = np.random.randint(0, 4)
            if choice == 1: audio = add_noise(audio)
            elif choice == 2: audio = pitch_shift(audio, sr, np.random.randint(-2, 3))
            elif choice == 3: audio = time_stretch(audio, np.random.uniform(0.8, 1.2))

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # Stack all features vertically
        combined_features = np.vstack([mfccs, chroma, zcr, rms, spec_cent, spec_bw, spec_rolloff])
        
        # Pad / truncate
        if combined_features.shape[1] > max_pad_len:
            combined_features = combined_features[:, :max_pad_len]
        else:
            pad_width = max_pad_len - combined_features.shape[1]
            combined_features = np.pad(combined_features, ((0, 0), (0, pad_width)), mode='constant')
        
        # Transpose to get the shape (time_steps, features)
        return combined_features.T
    
    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None

if __name__ == "__main__":
    train_df, test_df = load_and_split_data(DATASET_PATH)
    print(f"\nTotal files found and split: {len(train_df)} train, {len(test_df)} test.")
    os.makedirs(SEQUENCE_PATH, exist_ok=True)
  
    print("\n--- Generating Augmented Training Sequences (Speech + Song) ---")
    X_train, y_train = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training Set"):
        seq = extract_features(row['file_path'], use_augmentation=True)
        if seq is not None:
            X_train.append(seq)
            y_train.append(row['emotion_label'])
            
    print("\n--- Generating Clean Testing Sequences (Speech + Song) ---")
    X_test, y_test = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing Set"):
        seq = extract_features(row['file_path'], use_augmentation=False)
        if seq is not None:
            X_test.append(seq)
            y_test.append(row['emotion_label'])
          
    np.save(os.path.join(SEQUENCE_PATH, 'X_train_seq.npy'), np.array(X_train))
    np.save(os.path.join(SEQUENCE_PATH, 'y_train_seq.npy'), np.array(y_train))
    np.save(os.path.join(SEQUENCE_PATH, 'X_test_seq.npy'), np.array(X_test))
    np.save(os.path.join(SEQUENCE_PATH, 'y_test_seq.npy'), np.array(y_test))
    print("\nCombined sequence generation with augmentation complete.")