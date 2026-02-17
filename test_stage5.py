import cv2
from core.hand_tracking import HandDetector
from core.feature_extraction import extract_features
from utils.config import CAMERA_ID

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    detector = HandDetector()

    print("Show your hand. Watching for normalized features...")
    
    while True:
        success, img = cap.read()
        if not success: break

        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            features = extract_features(lm_list)
            # Print the first 4 features to show they are small numbers (approx -1 to 1)
            print(f"Features: {features[:4]} ... Total Len: {len(features)}")

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()