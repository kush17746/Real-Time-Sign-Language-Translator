import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading data...")
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: data.csv not found. Run collect_data.py first!")
    exit()

# 1. Separate Features (X) and Labels (y)
X = data.iloc[:, 1:].values  # All columns except the first
y = data.iloc[:, 0].values   # The first column (labels)

print(f"Dataset Stats: {len(data)} samples, {len(data['label'].unique())} classes.")

# 2. Train the Model (Support Vector Machine)
# We use a Linear SVM because it's fast and works well with high-dimensional data
print("Training model...")
model = SVC(kernel='linear', probability=True)
model.fit(X, y)

# 3. Evaluate (Optional but good to know)
print("Model trained successfully!")

# 4. Save the Model
save_path = 'models/model.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {save_path}")