import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D:\iris.data.csv")

# Display first few rows of the dataset
print(df.head())

# Extract features and labels
features = df[["sepal length", "sepal width", "petal length", "petal width"]]
labels = df["specie"]

# Initialize and train the model
model = KNeighborsClassifier()
model.fit(features, labels)

# Make a sample prediction
sample_prediction = model.predict([[1.9, 8.9, 3, 6]])
print("Sample Prediction:", sample_prediction)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=2)

# Train KNN with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Evaluate accuracy for different values of k
neighbors = range(1, 21)
accuracies = []

for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

# Plot accuracy vs. number of neighbors
plt.plot(neighbors, accuracies)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Classifier Accuracy for Different k Values")
plt.show()
