# Factory Reallocation & Shipping Optimization Recommendation System  
## Nassau Candy Distributor

---

## 📌 Project Overview

This project develops a Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor. The system leverages predictive modeling and decision intelligence to optimize product-to-factory assignments, reduce shipping lead times, and improve operational profitability.

Nassau Candy currently assigns products to factories using static rules and legacy processes. These outdated methods result in:

- Suboptimal shipping distances  
- Increased delivery lead times  
- Higher operational costs  
- Margin erosion due to logistics inefficiencies  

This project transforms descriptive analytics into an intelligent, simulation-based decision system capable of recommending optimal factory reallocations at scale.

---

## 🎯 Problem Statement

Nassau Candy lacks:

- A predictive model for estimating shipping lead time  
- A scenario simulation engine for factory reassignment  
- A quantitative evaluation system for operational impact  
- A scalable recommendation mechanism  

As a result, management cannot accurately determine:

- Which products should be reassigned to alternative factories  
- The expected lead time reduction  
- The financial impact of reassignment  
- The safest and most efficient configuration  

This system addresses these gaps using predictive modeling and optimization logic.

---

## 📊 Dataset Description

The dataset contains operational and financial attributes such as:

- Order ID  
- Order Date  
- Ship Date  
- Ship Mode  
- Region  
- Division  
- Product Name  
- Factory  
- Sales  
- Units  
- Cost  
- Gross Profit  

Shipping Lead Time is calculated as:

Lead Time = Ship Date − Order Date

Dataset location:

data/nassau_dataset.csv

---

## 🧠 Methodology

### 1️⃣ Data Preparation

- Convert date columns to datetime format  
- Compute shipping lead time  
- Handle missing values  
- Normalize numerical features  
- Encode categorical variables (Region, Ship Mode, Division, Factory)  
- Remove extreme outliers  
- Create training-ready feature matrix  

---

### 2️⃣ Predictive Modeling

Objective: Predict shipping lead time based on:

- Product  
- Origin factory  
- Destination region  
- Ship mode  

Models implemented:

- Linear Regression (Baseline)  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Model evaluation metrics:

- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

The best-performing model is selected based on accuracy and interpretability.

---

### 3️⃣ Scenario Simulation Engine

For each product:

- Simulate reassignment to alternate factories  
- Predict new lead time  
- Estimate operational improvement  
- Measure profit sensitivity  
- Rank factory options  

---

### 4️⃣ Optimization Logic

Factory reassignment recommendations are ranked using:

- Lead Time Reduction (%)  
- Risk Reduction  
- Profit Stability  
- Scenario Confidence Score  

The system generates top-N recommended factory configurations.

---

## 📈 Key Performance Indicators (KPIs)

- Lead Time Reduction (%)  
- Profit Impact Stability  
- Scenario Confidence Score  
- Recommendation Coverage  

---

## 💻 Streamlit Web Application

The project includes an interactive Streamlit dashboard featuring:

### Factory Optimization Simulator
- Select product  
- Compare predicted performance across factories  

### What-If Scenario Analysis
- Current vs recommended configuration  
- Lead-time improvement visualization  

### Recommendation Dashboard
- Ranked factory reassignment suggestions  
- Expected efficiency gains  

### Risk & Impact Panel
- Profit impact alerts  
- High-risk reassignment warnings  

Run locally:

streamlit run app.py

---

## 🏗 Project Structure

Factory-Reallocation-Shipping-Optimization-Nassau-Candy  
│  
├── app.py  
├── model_training.py  
├── requirements.txt  
├── README.md  
│  
├── data/  
│   └── nassau_dataset.csv  
│  
├── models/  
│   ├── lead_time_model.pkl  
│   ├── random_forest_model.pkl  
│   ├── gradient_boosting_model.pkl  
│   ├── label_encoders.pkl  
│   └── feature_columns.pkl  

---

## 🚀 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  
- Plotly  

---

## 📌 Business Impact

This system enables Nassau Candy to:

- Reduce shipping inefficiencies  
- Improve delivery performance  
- Maintain profit stability  
- Make data-driven factory reassignment decisions  
- Transition from static logistics planning to intelligent optimization  

This project shifts the organization from descriptive analytics to predictive and prescriptive decision intelligence.

---

## 👤 Author

Divyansh Choudhary