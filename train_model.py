import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Create dummy data for training
data = {
    'cgpa': [8.5, 7.0, 9.2, 6.5, 8.0, 5.5],
    'internships': [2, 0, 3, 1, 2, 0],
    'projects': [3, 1, 5, 2, 4, 1],
    'placed': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. Train Model
X = df.drop('placed', axis=1)
y = df['placed']
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the model
joblib.dump(model, 'placement_model.joblib')
print("Model saved successfully!")
