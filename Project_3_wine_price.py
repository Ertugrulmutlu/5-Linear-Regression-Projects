import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class WinePriceModel:
    """
    A simple linear regression model to predict wine selling prices based on numeric features.
    
    Attributes:
        file_path (str): Path to the dataset CSV file.
        model (LinearRegression): scikit-learn linear regression model instance.
        X (ndarray): Feature matrix.
        y (ndarray): Target vector (selling_price).
        X_train, X_test, y_train, y_test: Training and testing splits.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = LinearRegression()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Loads dataset from a CSV file, drops rows with missing values,
        and separates features and target variable.
        Excludes 'Price' and 'Year' columns from features.
        """
        df = pd.read_csv(self.file_path)
        df = df.dropna()
        self.y = df['Price'].values
        self.X = df.drop(['Price', 'Year'], axis=1).values

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train(self):
        """
        Trains the linear regression model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluates the model on the test data and prints performance metrics and sample predictions.
        """
        preds = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)

        print(f"Model Accuracy (R²): {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}\n")
        print("Sample Predictions:")
        for i in range(min(5, len(preds))):
            print(f"Predicted: {preds[i]:.2f}, Actual: {self.y_test[i]:.2f}, Input features: {self.X_test[i]}")

    def get_coefficients(self):
        """
        Prints the coefficients and intercept of the trained regression model.
        """
        print("Coefficients:", self.model.coef_)
        print("Intercept:", self.model.intercept_)

def visualize_predictions(y_true, y_pred, title="Actual vs Predicted Wine Prices"):
    """
    Visualizes actual vs predicted selling prices.

    Args:
        y_true (ndarray): True target values.
        y_pred (ndarray): Predicted target values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             linestyle='--', label='Ideal')
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = WinePriceModel("datas/wine.csv")
    model.load_data()
    model.split_data()
    model.train()
    preds = model.model.predict(model.X_test)
    visualize_predictions(model.y_test, preds)
    model.evaluate()
    model.get_coefficients()
