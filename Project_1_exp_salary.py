import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class ExperienceSalaryModel:
    """
    A linear regression model to predict salary based on experience and other features.

    Attributes:
        file_path (str): Path to the CSV dataset.
        target_column (str): Name of the salary column to predict.
        model (LinearRegression): The regression model.
        X (ndarray): Feature matrix after encoding.
        y (ndarray): Target salary values.
        X_train, X_test, y_train, y_test: Training and testing splits.
        feature_names (list): Column names corresponding to encoded features.
    """
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.model = LinearRegression()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self):
        """
        Reads CSV data, drops missing rows, and encodes categorical features.
        Separates features (X) and target salary (y).
        """
        data = pd.read_csv(self.file_path)
        data = data.dropna()

        X = data.drop(self.target_column, axis=1)
        X_encoded = pd.get_dummies(X, drop_first=True)
        self.X = X_encoded.values
        self.y = data[self.target_column].values
        self.feature_names = X_encoded.columns.tolist()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits features and target into training and test sets.

        Args:
            test_size (float): Proportion for test split.
            random_state (int): Seed for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train(self):
        """
        Fits the LinearRegression model on the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Predicts on the test set and prints R², MSE, and sample predictions.
        """
        preds = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)

        print(f"Model Accuracy (R²): {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}\n")
        print("Sample Predictions:")
        for i in range(min(5, len(preds))):
            print(f"Predicted: {preds[i]:.2f}, Actual: {self.y_test[i]:.2f}, Features: {self.X_test[i]}")

    def get_coefficients(self):
        """
        Prints model coefficients and intercept with feature names.
        """
        print("Coefficients (matching order of encoded features):")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")


def visualize_predictions(y_true, y_pred, title="Actual vs Predicted"):
    """
    Plots actual vs predicted values on a scatter plot with an ideal line.

    Args:
        y_true (ndarray): True salary values.
        y_pred (ndarray): Predicted salary values.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', label='Ideal')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = ExperienceSalaryModel("datas/Salary_dataset.csv", target_column="Salary")
    model.load_data()
    model.split_data()
    model.train()
    preds = model.model.predict(model.X_test)
    visualize_predictions(model.y_test, preds)
    model.evaluate()
    model.get_coefficients()