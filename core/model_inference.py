import pickle
import numpy as np
import os

class SignClassifier:
    def __init__(self, model_path='models/model.pkl'):
        """
        Loads the pre-trained model from the pickle file.
        """
        if not os.path.exists(model_path):
            # Fallback if model doesn't exist yet
            self.model = None
            print(f"Warning: {model_path} not found. Prediction will verify only.")
            return

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            print("Model loaded successfully.")

    def predict(self, features):
        """
        Predicts the sign based on features.
        Args:
            features: List of 42 normalized coordinates
        Returns:
            label (str): The predicted sign (e.g., 'A')
            confidence (float): Probability score (0.0 to 1.0)
        """
        if self.model is None:
            return "No Model", 0.0

        # Reshape to (1, 42) because sklearn expects a batch
        features_np = np.array([features])
        
        # Predict Class
        prediction = self.model.predict(features_np)[0]
        
        # Predict Probability
        confidence = np.max(self.model.predict_proba(features_np))
        
        return prediction, confidence