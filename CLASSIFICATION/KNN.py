import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set()

# 1. Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()

# 2. Create a DataFrame with feature data
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# 3. Select specific features for analysis
X = X[['mean area', 'mean compactness']]

# 4. Define the target variable as a 1D array (0 = malignant, 1 = benign)
y = breast_cancer.target  # 0 and 1

# 5. Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Initialize and Train the KNN Classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 8. Make Predictions
y_pred = knn.predict(X_test_scaled)

# 9. Evaluation
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=breast_cancer.target_names,
            yticklabels=breast_cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=breast_cancer.target_names))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 10. Visualization of Decision Boundary
from matplotlib.colors import ListedColormap

# Define the mesh grid
h = .02  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict classifications for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Plot the contour and training points
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Scatter plot of test data
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=cmap_bold,
            edgecolor='k', s=50, label='Actual')

# Highlight misclassified points
misclassified = y_test != y_pred
plt.scatter(X_test_scaled[misclassified, 0], X_test_scaled[misclassified, 1],
            facecolors='none', edgecolors='k', s=100, label='Misclassified')

plt.xlabel('Mean Area (scaled)')
plt.ylabel('Mean Compactness (scaled)')
plt.title(f'KNN Decision Boundary with K=5')
plt.legend()
plt.show()
