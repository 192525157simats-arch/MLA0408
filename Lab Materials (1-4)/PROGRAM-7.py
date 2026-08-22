# Experiment 7
# Aim: To demonstrate the performance of Logistic Regression
# using a chosen dataset with Python

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Breast Cancer dataset
data = load_breast_cancer()

# Input and output
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Values for sigmoid graph
z = np.arange(-5, 5, 0.1)

# Plot sigmoid function
plt.plot(z, sigmoid(z))

plt.title('Visualization of the Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.grid()

plt.show()
