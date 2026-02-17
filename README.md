# 🗣️ Real-Time Sign Language Translator

**A Computer Vision & AI project that translates hand sign gestures into text and speech in real-time.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## 📌 Project Overview
This project addresses the communication gap for the deaf and mute community. It uses **MediaPipe** for hand tracking and a **Support Vector Machine (SVM)** to classify gestures. The output is displayed on a **Streamlit** dashboard and converted to speech using **pyttsx3**.

### ✨ Features
- **Real-Time Detection:** Zero lag hand tracking using MediaPipe.
- **Two-Hand Support:** Detects and classifies gestures from both hands.
- **Text-to-Speech:** Vocalizes the detected sign for better communication.
- **Sentence Formation:** Smart logic to construct sentences from individual signs.
- **Customizable:** Easily train new signs using the built-in data collector.

## 🛠️ Tech Stack
- **OpenCV:** Image capturing and processing.
- **MediaPipe:** Hand landmark extraction (21 points).
- **Scikit-Learn:** SVM Model for classification.
- **Streamlit:** Web-based GUI for the dashboard.
- **Pyttsx3:** Text-to-speech conversion.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Real-Time-Sign-Language-Translator.git](https://github.com/YOUR_USERNAME/Real-Time-Sign-Language-Translator.git)
   cd Real-Time-Sign-Language-Translator

2. **Install Dependencies**

Bash
pip install -r requirements.txt

3. **Run the App**

Bash
streamlit run app.py

4. **Project strucutre**
📂 Project Structure
├── core/                # Core modules (Camera, AI, Speech)
├── models/              # Trained ML models (.pkl)
├── utils/               # Configuration settings
├── app.py               # Main application (Streamlit)
├── train_model.py       # Script to train the AI
└── collect_data.py      # Script to add new signs


👨‍💻 Author
kush hohel
3rd Year AI & Data Science Student


---

### **STEP 5: Final Update**

After saving your `README.md`, run these commands one last time to update GitHub:

```bash
git add .
git commit -m "Added documentation"
git push