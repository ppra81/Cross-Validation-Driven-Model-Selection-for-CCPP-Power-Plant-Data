# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 3: Data Preparation
# Load the dataset
file_path = r"C:\Users\Praneeth\Downloads\CCPP_data.csv"  
df = pd.read_csv(file_path)

# Explore the data
print(df.head())
# Check for missing values
print(df.isnull().sum())

# Split the data into features (X) and target variable (y)
X = df.drop('PE', axis=1)
y = df['PE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building
# Model 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Model 2: Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Step 5: Model Evaluation
# Validation Set (Cross-Validation for Model Selection)
cv_scores_lr = cross_val_score(model_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(model_rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Choose the model with the best performance on the validation set
best_model = model_lr if cv_scores_lr.mean() > cv_scores_rf.mean() else model_rf

# Step 6: Model Interpretation
# Evaluate on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 8: Model in Python
joblib.dump(best_model, 'model.pkl')
