# K-Nearest Neighbors (KNN) Classification on Iris Dataset

## Overview
This project implements the **K-Nearest Neighbors (KNN)** algorithm to classify iris flowers based on sepal and petal dimensions. The dataset used is the **Iris dataset**, which contains labeled samples of different iris species.

## Features
- Loads the **Iris dataset** from a CSV file.
- Extracts **features** (sepal length, sepal width, petal length, petal width) and **labels** (species).
- Splits the dataset into **training (60%) and testing (40%)** sets.
- Trains a **KNN model** with a default `n_neighbors=5`.
- Makes a prediction on a new sample.
- Evaluates the model accuracy using a **test dataset**.
- Finds the optimal `n_neighbors` by testing values from **1 to 20**.
- Plots a **graph of accuracy vs. number of neighbors**.

## Installation & Requirements
Ensure you have the following installed:
```bash
pip install pandas scikit-learn matplotlib
```

## Usage
1. Place the `iris.data.csv` file in the appropriate directory.
2. Run the script to train the KNN model and evaluate its performance.
3. Adjust the `n_neighbors` value to observe its effect on accuracy.
4. View the accuracy plot to determine the best K value.

## Code Explanation
1. **Load Dataset**: Read `iris.data.csv` using Pandas.
2. **Extract Features & Labels**: Select the required columns.
3. **Train Initial Model**: Fit a KNN model and make a sample prediction.
4. **Split Data**: Train-test split (60%-40%).
5. **Train & Test KNN Model**: Train a new model with `n_neighbors=10` and compute accuracy.
6. **Optimize K Value**: Iterate over values from 1 to 20, tracking accuracy.
7. **Plot Results**: Visualize accuracy trends.

## Output
- **Predictions** for test samples.
- **Accuracy score** of the trained model.
- **Graph** showing the impact of `n_neighbors` on accuracy.


