import cv2
import csv
import os
import numpy as np
from core.hand_tracking import HandDetector
from core.feature_extraction import extract_features
from utils.config import CAMERA_ID

# 1. Setup Data File
DATA_FILE = 'data.csv'

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Columns: label, pixel0, pixel1 ... pixel41
        writer.writerow(['label'] + [f'pixel{i}' for i in range(42)])
        print(f"Created new data file: {DATA_FILE}")

# 2. Initialize Camera & Detector
cap = cv2.VideoCapture(CAMERA_ID)
detector = HandDetector()

print("========================================")
print("  DATA COLLECTION MODE")
print("  1. Enter a label name in the terminal.")
print("  2. Press 's' to save a frame.")
print("  3. Press 'q' to quit.")
print("========================================")

current_label = input("Enter the label to collect (e.g., 'A', 'Hello'): ").strip()
print(f"Collecting data for: '{current_label}'. Show your hand and press 's'...")

counter = 0

while True:
    success, img = cap.read()
    if not success: break
    
    # Detect Hand
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    
    if len(lm_list) != 0:
        # Extract Features
        features = extract_features(lm_list)
        
        # Draw Visual Feedback
        cv2.putText(img, f"Label: {current_label}", (10, 50), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Saved: {counter}", (10, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('s'):
            with open(DATA_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + features)
            counter += 1
            print(f"Saved sample #{counter} for {current_label}")

    cv2.imshow("Data Collector", img)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()