# Rainfall Prediction Using Neural Networks

This project implements a neural network model to predict rainfall based on historical meteorological data. It is developed using Python in a Jupyter Notebook environment with libraries such as TensorFlow, Keras, Pandas, and Scikit-learn. The workflow includes data preprocessing, model training, evaluation, and prediction visualization.

---

## Project Overview

The goal of this project is to forecast rainfall amounts using key weather attributes such as temperature, humidity, and atmospheric pressure. The model is trained on a historical dataset and evaluated for its predictive performance. The project outlines an end-to-end pipeline—from data preparation to visualization of results.

---

## Technologies Used

* Python
* Jupyter Notebook
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras

---

## Repository Contents

* `rainfall_prediction_final.ipynb`: Main notebook containing the complete implementation.
* `rainfall.csv`: Dataset used for model training and evaluation.
* `model.h5`: Optional saved model file.
* `README.md`: Project documentation.

---

## Workflow Breakdown

### 1. Data Loading and Preprocessing

* Load the dataset (`rainfall.csv`).
* Handle missing or irrelevant values.
* Drop unnecessary columns and clean data for modeling.

### 2. Feature Engineering

* Select key features such as humidity, temperature, pressure, and wind speed.
* Normalize feature values using `MinMaxScaler`.

### 3. Model Building

* Construct a Sequential model using Keras.
* Employ Dense layers with ReLU activation functions.
* Compile the model with the Adam optimizer and Mean Squared Error (MSE) loss.

### 4. Model Training and Evaluation

* Train the neural network on historical data.
* Validate performance using metrics such as RMSE and R² score.
* Visualize training and validation loss trends.

### 5. Prediction and Visualization

* Predict rainfall on the test set.
* Compare actual vs. predicted values using line plots and scatter plots.

---

## Model Configuration

* **Input Features**: Humidity, Temperature, Pressure, Wind Speed, etc.
* **Architecture**: Fully-connected Dense Neural Network
* **Activation**: ReLU
* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam
* **Evaluation Metrics**: Root Mean Squared Error (RMSE), R² Score

---

## Performance Snapshot

* **Root Mean Squared Error (RMSE)**: e.g., 3.25 mm
* **R² Score**: e.g., 0.85
* **Visuals Included**:

  * Training vs. Validation Loss Curve
  * Actual vs. Predicted Rainfall Plot

---

## How to Run the Project

1. Clone the repository or download the files.
2. Navigate to the project directory.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:

   ```bash
   jupyter notebook rainfall_prediction_final.ipynb
   ```
