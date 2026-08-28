# 🏠 Bengaluru House Price Prediction

### Machine Learning Based Real Estate Price Estimation

A machine learning web application that predicts residential property prices in **Bengaluru, Karnataka** using **XGBoost Regression**.

The project combines data preprocessing, feature engineering, machine learning model training, evaluation, and an interactive **Streamlit dashboard** to provide estimated house prices based on property characteristics.

---

## 📌 Project Overview

Buying or selling a property requires a reasonable understanding of its market value. House prices can vary significantly depending on factors such as location, property size, BHK, bathrooms, balcony availability, and area type.

This project uses historical Bengaluru housing data to train an XGBoost regression model and provide an estimated property price through an easy-to-use web interface.

### 🎯 Objective

Build a complete machine learning pipeline that:

* Cleans and prepares housing data
* Performs feature engineering
* Encodes categorical variables
* Trains an XGBoost regression model
* Evaluates model performance
* Saves the trained model
* Provides predictions through a Streamlit dashboard

---

## ✨ Key Features

* 📍 Location-based house price prediction
* 📐 Property area input
* 🏘️ Area type selection
* 🛏️ BHK selection
* 🚿 Bathroom selection
* 🌅 Balcony selection
* 📊 Area-per-BHK feature
* 🤖 XGBoost regression model
* 💰 Price estimation in Indian Lakhs
* 💵 Price per square foot calculation
* 📊 Model performance metrics
* 🖥️ Interactive Streamlit dashboard
* 🏠 Professional real-estate style interface

---

## 🧠 Machine Learning Workflow

```text
Bengaluru Housing Dataset
          ↓
     Data Cleaning
          ↓
  Feature Engineering
          ↓
   Categorical Encoding
          ↓
    Train/Test Split
          ↓
   XGBoost Regressor
          ↓
     Model Evaluation
          ↓
     Saved Model
          ↓
   Streamlit Dashboard
          ↓
    Price Prediction
```

---

## 🔧 Features Used by the Model

The final prediction dataset contains **150 features** after preprocessing and one-hot encoding.

Important input features include:

| Feature      | Description                    |
| ------------ | ------------------------------ |
| Area         | Property area in square feet   |
| BHK          | Number of bedrooms             |
| Bath         | Number of bathrooms            |
| Balcony      | Number of balconies            |
| Area Type    | Type of property area          |
| Location     | Property location              |
| Area per BHK | Engineered area-to-BHK feature |

Categorical variables such as **location** and **area type** are converted into numerical features using one-hot encoding.

---

## 🤖 Machine Learning Model

### XGBoost Regressor

The project uses **XGBoost Regression** because it is well suited for structured/tabular datasets and can model non-linear relationships between property characteristics and prices.

**Algorithm:** XGBoost Regressor
**Prediction Type:** Regression
**Input Features:** 150

---

## 📊 Model Performance

The final model was evaluated using a separate testing dataset.

| Metric      |           Result |
| ----------- | ---------------: |
| Training R² |       **77.54%** |
| Testing R²  |       **72.03%** |
| MAE         | **₹12.77 Lakhs** |
| RMSE        | **₹18.07 Lakhs** |

### What the metrics mean

**R² Score**
Measures how well the model explains the variation in house prices.

**MAE — Mean Absolute Error**
Represents the average absolute difference between the actual and predicted prices.

**RMSE — Root Mean Squared Error**
Measures prediction error while giving greater weight to larger errors.

---

## 📈 Evaluation

Model evaluation graphs are available in the `evaluation/` directory.

### R² Score

![R² Score](evaluation/r2_training_vs_testing.png)

### Error Metrics

![Error Metrics](evaluation/error_metrics.png)

Additional evaluation visualizations can be added as the project develops, including:

* Actual vs Predicted Prices
* Residual Analysis
* Feature Importance

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard where users can enter property information and receive an estimated price.

### Dashboard Inputs

```text
Location
Area Type
Area
BHK
Bathrooms
Balcony
```

### Dashboard Output

```text
Estimated House Price
Price per Sq. Ft.
Area per BHK
Property Summary
Model Performance
```

---

## 🏠 Example Prediction

Example property:

| Property Detail | Value              |
| --------------- | ------------------ |
| Location        | 7th Phase JP Nagar |
| Area            | 1,200 Sq. Ft.      |
| BHK             | 5                  |
| Bathrooms       | 3                  |
| Balcony         | 2                  |
| Area Type       | Built-up Area      |

The application uses these values to generate an estimated Bengaluru property price using the trained XGBoost model.

---

## 📂 Project Structure

```text
Bengaluru-House-Price-Prediction/
│
├── app.py
├── train_model.py
│
├── models/
│   ├── house_price_model.json
│   └── model_columns.pkl
│
├── assets/
│   └── house.jpg
│
├── evaluation/
│   ├── r2_training_vs_testing.png
│   ├── error_metrics.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Visualization

* Matplotlib

### Web Application

* Streamlit

### Model Storage

* XGBoost JSON Model
* Pickle

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Bengaluru-House-Price-Prediction.git
```

### 2. Open the project

```bash
cd Bengaluru-House-Price-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

## 🔄 Model Training

The training pipeline is available in:

```text
train_model.py
```

The training process includes:

1. Loading the housing dataset
2. Data cleaning
3. Handling missing values
4. Feature engineering
5. Outlier handling
6. Categorical encoding
7. Train/test splitting
8. XGBoost model training
9. Model evaluation
10. Saving the trained model

---

## 📦 Model Files

The trained model is stored separately from the application:

```text
models/
├── house_price_model.json
└── model_columns.pkl
```

`house_price_model.json` contains the trained XGBoost model.

`model_columns.pkl` stores the feature names and order required during prediction.

Keeping the feature order consistent between training and prediction is important for obtaining correct results.

---

## 📊 Dataset

The project uses a Bengaluru residential property dataset containing information about property characteristics and prices.

The dataset is not included in this repository to keep the repository lightweight.

---

## ⚠️ Disclaimer

This application provides an **estimated price based on historical housing data and the trained machine learning model**.

It should not be considered an official property valuation or financial recommendation. Actual market prices may differ depending on property condition, exact location, amenities, market conditions, and other factors not represented in the dataset.

---

## 🚀 Future Improvements

* [ ] Add more recent Bengaluru housing data
* [ ] Improve location-level accuracy
* [ ] Add actual vs predicted visualization
* [ ] Add residual analysis
* [ ] Add feature importance visualization
* [ ] Compare XGBoost with Random Forest and other regression models
* [ ] Deploy the Streamlit application online
* [ ] Add more detailed property analytics

---

## 👨‍💻 Author

### Samarth Kokate

B.Tech Computer Engineering / Data Science

Interested in:

**Machine Learning • Data Science • Python • Data Analytics**

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Built with Python, XGBoost and Streamlit.**

🏠 **Bengaluru House Price Prediction**
