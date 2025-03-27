import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data from the CSV file
file_path = "Cleaned_csvs/combined_last_lines.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Check the first few rows of the data
print(df.head())

# Define the independent variables (X) and dependent variable (y)
X = df[['High', 'Low', 'Open', 'Volume']]  # Independent variables
y = df['Adj Close']  # Dependent variable (target)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)

# Print the coefficients (weights for each feature)
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Calculate the Mean Absolute Error (MAE) of the predictions
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Optionally, print the R-squared value to evaluate the model's performance
r_squared = model.score(X_test, y_test)
print(f"R-squared: {r_squared}")
