import cv2
import mediapipe as mp
import time
from utils.config import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

class HandDetector:
    def __init__(self, mode=False, max_hands=MAX_NUM_HANDS, detection_con=MIN_DETECTION_CONFIDENCE, track_con=MIN_TRACKING_CONFIDENCE):
        """
        Initializes the MediaPipe Hands module.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        """
        Processes the image to find hands and draws landmarks if requested.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_lms, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0):
        """
        Returns landmarks for a specific hand number (0 or 1).
        """
        lm_list = []
        if self.results.multi_hand_landmarks:
            # Check if the requested hand exists (e.g., if hand_no is 1 but only 1 hand is visible)
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    
        return lm_list