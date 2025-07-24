# 5 Linear Regression Projects

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![NumPy](https://img.shields.io/badge/numpy-v1.19+-yellow.svg)
![Pandas](https://img.shields.io/badge/pandas-v1.1+-lightgrey.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-v3.3+-brightgreen.svg)


This repository contains five independent real-time linear regression projects developed in Python, each demonstrating predictive modeling using distinct datasets. These projects utilize the scikit-learn library for linear regression and feature comprehensive data visualization, preprocessing, and evaluation components.

➡️ **To understand the theory and interpretation behind these projects, check out the accompanying article here:**  
[Using Linear Regression on Real-World Datasets: Theory and Results](https://dev.to/ertugrulmutlu/real-time-regression-projects-in-python-with-full-code-p5g)

## Detailed Projects Overview

### 1. Salary Prediction based on Experience

* **Objective:** Predict salaries by leveraging professional experience and additional categorical attributes.
* **Features:** Years of experience, education level, position, industry, and other categorical variables (one-hot encoded).
* **Dataset:** `Salary_dataset.csv`

### 2. Car Price Prediction

* **Objective:** Estimate the market price of cars based on multiple numerical characteristics.
* **Features:** Age, mileage, fuel type, transmission type, owner type, excluding textual details such as 'make' and 'model'.
* **Dataset:** `cars24-car-price-clean2.csv`

### 3. Wine Price Prediction

* **Objective:** Predict selling prices of wine bottles using their characteristics.
* **Features:** Alcohol content, sugar level, acidity, grape variety, and other numeric attributes, excluding the price and vintage year.
* **Dataset:** `wine.csv`

### 4. Insurance Charges Prediction

* **Objective:** Estimate medical insurance charges based on demographic, lifestyle, and geographic details.
* **Features:** Age, BMI, smoking status, gender (binary encoded), geographic region (one-hot encoded).
* **Dataset:** `insurance.csv`

### 5. Trip Duration Prediction

* **Objective:** Forecast trip durations from trip-specific numeric attributes.
* **Features:** Distance, passenger count, pickup location, drop-off location, time-of-day, and other numeric attributes relevant to trip durations.
* **Dataset:** `train.csv`

## Common Workflow Explained

Each project consistently follows the workflow below:

1. **Data Acquisition:** Reading data from CSV files.
2. **Data Cleaning:** Removing incomplete or irrelevant records.
3. **Feature Processing:** Encoding categorical data using binary or one-hot encoding methods.
4. **Data Splitting:** Partitioning data into training and testing subsets (80% for training, 20% for testing).
5. **Model Training:** Implementing and fitting a linear regression model.
6. **Model Evaluation:** Evaluating performance with R² scores and Mean Squared Errors (MSE).
7. **Data Visualization:** Plotting predicted versus actual values for visual inspection of model accuracy.

## Execution Instructions

To execute a project, run the corresponding Python script via the command line:

```bash
python Project_X_name.py
```

Replace `Project_X_name.py` with the specific project's Python filename.

## Required Dependencies

Before running the scripts, ensure the required libraries are installed:

```bash
pip install -r requirements.txt
```

## Data Availability

The datasets mentioned above are available upon request. Please open an issue or contact directly if additional datasets are needed.

## Contributions

We encourage contributions and improvements! Submit a pull request or open an issue with your suggestions or enhancements.

---

Happy exploring and modeling with linear regression!
