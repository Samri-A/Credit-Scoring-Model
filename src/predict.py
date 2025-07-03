import joblib
import numpy as np

# Load the trained model
model_path = "models/random_forest_model/model.pkl"
model = joblib.load(model_path)

print(""" Example input — make sure it matches the order: 
 [Total_Amount, Average_Amount, Transaction_Count, Std_Amount]""")

#input_data = np.array(input("Enter here"))
input_data = np.array([[150000.0, 5000.0, 5, 250.5]])

# Predict
prediction = model.predict(input_data)
proba = model.predict_proba(input_data)

# Output result
print(f"Prediction (0 = Low Risk, 1 = High Risk): {int(prediction[0])}")
print(f"Probability of being High Risk: {proba[0][1]:.4f}")
