import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D:\iris.data.csv")

# Streamlit UI
st.set_page_config(page_title="Iris Classification", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f5f5f5; padding: 20px;}
        h1 {color: #2e3b4e; text-align: center;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🌸 Iris Flower Classification 🌸")
st.markdown("### An interactive and visually appealing KNN classifier for the Iris dataset.")

# Display dataset preview if selected
if st.checkbox("📊 Show dataset preview"):
    st.dataframe(df.style.set_properties(**{'background-color': '#ffffff', 'border': '1px solid #ddd'}))

# Extract features and labels
features = df[["sepal length", "sepal width", "petal length", "petal width"]]
labels = df["specie"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=2)

# Sidebar styling and inputs
st.sidebar.header("🔧 Model Configuration")
k = st.sidebar.slider("Select number of neighbors (k)", 1, 20, 10)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
st.sidebar.success(f"📈 Model Accuracy: {accuracy:.2f}")

# User Input for Prediction
st.subheader("🔍 Make a Prediction")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("🌿 Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
    petal_length = st.number_input("🌿 Petal Length", min_value=0.0, max_value=10.0, value=1.5)
with col2:
    sepal_width = st.number_input("🌿 Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    petal_width = st.number_input("🌿 Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("🌸 Predict Species"):
    prediction = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"🎯 Predicted Species: {prediction[0]}")

# Evaluate accuracy for different k values
neighbors = range(1, 21)
accuracies = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

# Plot accuracy graph
st.subheader("📊 KNN Accuracy for Different k Values")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(neighbors, accuracies, marker='o', linestyle='-', color='#FF5733')
ax.set_xlabel("Number of Neighbors (k)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("KNN Classifier Accuracy", fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)
