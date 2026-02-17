import numpy as np

def extract_features(lm_list):
    """
    Converts a list of 21 landmarks into a normalized feature vector.
    Args:
        lm_list: List of [id, x, y] from HandDetector.
    Returns:
        np.array: A flat array of 42 values (21 points * (x, y)) normalized.
        Returns None if input is empty.
    """
    if len(lm_list) == 0:
        return None

    # 1. Convert to NumPy array (discard the index ID, keep only x and y)
    # lm_list structure is [[id, x, y], ...]. We want [[x, y], ...]
    landmarks = np.array(lm_list)[:, 1:] 

    # 2. Origin Translation (Make Wrist (0,0))
    # Point 0 is the Wrist. We subtract its coordinates from all points.
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    
    # Subtract wrist x from all x's, wrist y from all y's
    landmarks[:, 0] = landmarks[:, 0] - base_x
    landmarks[:, 1] = landmarks[:, 1] - base_y

    # 3. Normalization (Scale Invariant)
    # We find the maximum absolute value to scale everything between -1 and 1
    max_value = np.max(np.abs(landmarks))
    if max_value == 0: 
        max_value = 1 # Prevent division by zero
        
    landmarks = landmarks / max_value

    # 4. Flatten
    # Convert 21x2 matrix into a 1x42 vector
    return landmarks.flatten().tolist()