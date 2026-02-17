import os
import cv2
import csv
import mediapipe as mp
import numpy as np
from core.feature_extraction import extract_features

# --- CONFIGURATION ---
DATASET_PATH = r"D:\CODING\SignLanguageTranslator\archive\dataset"# UPDATE THIS!
OUTPUT_FILE = "data.csv"

# --- INIT MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# --- SETUP CSV ---
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'pixel{i}' for i in range(42)])

# --- PROCESSING LOOP ---
print(f"Scanning {DATASET_PATH}...")
success_count = 0

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    
    if os.path.isdir(folder_path):
        print(f"Processing Class: {folder_name}")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Read Image
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Process with MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convert to list format [id, x, y]
                    h, w, c = img.shape
                    lm_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])
                    
                    # Extract Features (using your existing core logic)
                    features = extract_features(lm_list)
                    
                    if features:
                        with open(OUTPUT_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([folder_name] + features)
                        success_count += 1

print(f"Done! Successfully converted {success_count} images to CSV.")