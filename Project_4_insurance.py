import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class InsuranceChargeModel:
    """
    A linear regression model to predict insurance charges based on demographic and lifestyle features.

    Attributes:
        file_path (str): Path to the CSV dataset.
        model (LinearRegression): The regression model instance.
        X (pd.DataFrame): Feature DataFrame after encoding.
        y (ndarray): Target array of insurance charges.
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
        Loads the dataset from a CSV file, drops missing values,
        encodes binary and categorical columns, and separates features and target.

        Maps 'sex' and 'smoker' to numeric, and one-hot encodes 'region'.
        """
        df = pd.read_csv(self.file_path)
        df = df.dropna()

        # Encode binary columns
        df['sex'] = df['sex'].map({'male': 1, 'female': 0})
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

        # One-hot encode region column
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df.drop('region', axis=1), region_dummies], axis=1)

        # Set features and target
        self.X = df.drop('charges', axis=1)
        self.y = df['charges'].values

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits features and target into training and test sets.

        Args:
            test_size (float): Fraction of data reserved for testing.
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
        Predicts on the test set and prints R², MSE,
        and sample predictions with corresponding input features.
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
        Prints model coefficients for each feature column and the intercept.
        """
        print("Coefficients (feature columns in order):")
        for col, coef in zip(self.X.columns, self.model.coef_):
            print(f"{col}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")


def visualize_predictions(y_true, y_pred, title="Actual vs Predicted Insurance Charges"):
    """
    Plots actual against predicted insurance charges on a scatter plot.

    Args:
        y_true (ndarray): True charge values.
        y_pred (ndarray): Predicted charge values.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             linestyle='--', label='Ideal')
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = InsuranceChargeModel("datas/insurance.csv")
    model.load_data()
    model.split_data()
    model.train()
    preds = model.model.predict(model.X_test)
    visualize_predictions(model.y_test, preds)
    model.evaluate()
    model.get_coefficients()
