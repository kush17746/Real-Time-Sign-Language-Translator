import cv2
import mediapipe as mp
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {mp.__version__}")
print(f"Numpy Version: {np.__version__}")

# Test Camera Access
cap = cv2.VideoCapture(0) # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Success: Webcam detected! Press 'q' to exit the window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Setup Test', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()