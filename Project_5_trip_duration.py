import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class TripDurationModel:
    """
    A linear regression model to predict trip duration based on numeric features.

    Attributes:
        file_path (str): Path to the CSV dataset.
        model (LinearRegression): The regression model instance.
        X (pd.DataFrame): Feature DataFrame.
        y (ndarray): Target array of trip durations.
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
        Reads the CSV file, drops missing values,
        and separates features (X) and target trip duration (y).
        """
        df = pd.read_csv(self.file_path)
        df = df.dropna()

        # Separate input features and target variable
        self.X = df.drop('trip_duration', axis=1)
        self.y = df['trip_duration'].values

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and test sets.

        Args:
            test_size (float): Fraction of data to use for testing.
            random_state (int): Seed for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train(self):
        """
        Fits the linear regression model on the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Predicts on the test set and prints performance metrics:
        R² score, Mean Squared Error, plus sample predictions.
        """
        preds = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)

        print(f"Model Accuracy (R²): {r2:.4f}")
        print(f"Mean Squared Error: {mse:.2f}\n")
        print("Sample Predictions:")
        for i in range(min(5, len(preds))):
            features = self.X_test.iloc[i].tolist()
            print(f"Predicted: {preds[i]:.2f}, Actual: {self.y_test[i]:.2f}, Features: {features}")

    def get_coefficients(self):
        """
        Prints the regression coefficients for each feature and the intercept.
        """
        print("Coefficients (in order of DataFrame columns):")
        for col, coef in zip(self.X.columns, self.model.coef_):
            print(f"{col}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")


def visualize_predictions(y_true, y_pred, title="Actual vs Predicted Trip Duration"):
    """
    Plots actual vs predicted trip durations on a scatter plot with an ideal reference line.

    Args:
        y_true (ndarray): True trip duration values.
        y_pred (ndarray): Predicted trip duration values.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', label='Ideal')
    plt.xlabel("Actual Trip Duration")
    plt.ylabel("Predicted Trip Duration")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = TripDurationModel("datas/train.csv")
    model.load_data()
    model.split_data()
    model.train()
    preds = model.model.predict(model.X_test)
    visualize_predictions(model.y_test, preds)
    model.evaluate()
    model.get_coefficients()
