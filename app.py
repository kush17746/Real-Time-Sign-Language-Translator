import streamlit as st
import cv2
import time
import numpy as np

# Import our modules
from core.hand_tracking import HandDetector
from core.feature_extraction import extract_features
from core.model_inference import SignClassifier
from core.text_to_speech import VoiceEngine
from utils.config import CAMERA_ID

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sign Language Translator", layout="wide")

st.title("🤖 AI Sign Language Sentence Builder")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
run_app = st.sidebar.checkbox("Start Camera", value=False)
enable_voice = st.sidebar.checkbox("Enable Voice", value=True)

if st.sidebar.button("Clear Sentence"):
    st.session_state['sentence'] = []

# --- INITIALIZE STATE ---
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = []

# --- INITIALIZE MODULES ---
@st.cache_resource
def load_modules():
    detector = HandDetector()
    classifier = SignClassifier()
    try:
        voice = VoiceEngine()
    except:
        voice = None 
    return detector, classifier, voice

detector, classifier, voice = load_modules()

# --- LAYOUT ---
SENTENCE_PLACEHOLDER = st.empty()
col1, col2 = st.columns([2, 1])
with col1:
    FRAME_WINDOW = st.empty()
with col2:
    TEXT_PLACEHOLDER = st.empty()

# --- VARIABLES ---
last_prediction = ""
consecutive_frames = 0
CONFIDENCE_THRESHOLD = 0.7

# --- CAMERA LOOP ---
if run_app:
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while run_app:
            success, frame = cap.read()
            if not success: break
            
            # 1. Detect All Hands
            # This draws the landmarks on both hands if visible
            frame = detector.find_hands(frame)
            
            # We will try to detect up to 2 hands (Hand 0 and Hand 1)
            # hand_data stores the info to display on screen
            detected_hands_info = [] 

            for hand_idx in range(2): # Check for Hand 0 and Hand 1
                lm_list = detector.find_position(frame, hand_no=hand_idx)
                
                if len(lm_list) != 0:
                    features = extract_features(lm_list)
                    sign, conf = classifier.predict(features)
                    
                    if conf > CONFIDENCE_THRESHOLD:
                        # Get coordinate of the wrist (point 0) to put text label
                        wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]
                        
                        # Draw label on video (Visual Feedback for BOTH hands)
                        label_color = (255, 0, 0) if hand_idx == 0 else (0, 0, 255) # Blue for H1, Red for H2
                        cv2.putText(frame, f"{sign} ({int(conf*100)}%)", (wrist_x, wrist_y - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
                        
                        detected_hands_info.append(sign)

                        # --- SENTENCE LOGIC (PRIMARY HAND ONLY) ---
                        # We only use Hand 0 to control the sentence/voice to avoid double-typing
                        if hand_idx == 0:
                            if sign != last_prediction:
                                consecutive_frames = 0
                                last_prediction = sign
                            else:
                                consecutive_frames += 1
                                
                            # If sign is held for 15 frames (STABLE)
                            if consecutive_frames == 15:
                                if sign == "SPACE":
                                    st.session_state['sentence'].append(" ")
                                elif sign == "CLEAR":
                                    st.session_state['sentence'] = []
                                else:
                                    # Prevent duplicate words
                                    if not st.session_state['sentence'] or st.session_state['sentence'][-1] != sign:
                                        st.session_state['sentence'].append(sign)
                                        if enable_voice and voice:
                                            voice.speak(sign)
                                consecutive_frames = 0

            # 2. Update Streamlit UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")
            
            # Display Current Detection Text
            if detected_hands_info:
                current_text = " + ".join(detected_hands_info)
                TEXT_PLACEHOLDER.markdown(f"### Detected: **{current_text}**")
            else:
                TEXT_PLACEHOLDER.markdown("### Detected: ...")
            
            # Display Full Sentence
            full_sentence = "".join(st.session_state['sentence'])
            SENTENCE_PLACEHOLDER.info(f"📝 **Sentence:** {full_sentence}")

            time.sleep(0.05) 
            
            if not run_app:
                break
            
        cap.release()
else:
    st.info("Check 'Start Camera' to begin.")