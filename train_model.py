import pandas as pd
df = pd.read_excel("ELECTION DATA SET COMPILATION.xlsx")
df.head()
df.info()
df.columns
df1 = df.copy()
df1.columns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from prophet import Prophet  # Ensure this is installed as 'prophet'
import numpy as np
import joblib

# Encode categorical features (including Region and Incumbent Party)
categorical_columns = ['Region', 'Incumbent Party']  # List all categorical columns
df1 = pd.get_dummies(df1, columns=categorical_columns, drop_first=True)

# Ensure no remaining non-numeric data
print(df1.dtypes)  # This should show all columns as numeric types

# Prepare Features and Targets
X = df1.drop(columns=['Winner', 'NPP', 'NDC', 'Others','Valid Votes'])
y_class = df1['Winner']  # Classification target
y_reg = df1[['NPP', 'NDC', 'Others',]]  # Regression targets

# Save dummy variable names after training
feature_names = X.columns.tolist()

try:
    joblib.dump(feature_names, 'Instances/feature_names.pkl')
    print("Feature names saved successfully to 'Instances/feature_names.pkl'")
except Exception as e:
    print(f"Failed to save feature names: {e}")

# Train-test split for classification and regression
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train Classification Model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

# Train Regression Model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_reg_train, y_reg_train)
y_reg_pred = reg.predict(X_reg_test)

# Evaluate Models
print("Classification Accuracy:", accuracy_score(y_class_test, y_class_pred))
print("Regression RMSE:", np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))


# Time Series Forecasting (Valid Votes Prediction)
turnout_data = df1[['Year', 'Valid Votes']].rename(columns={'Year': 'ds', 'Valid Votes': 'y'})
turnout_data['ds'] = pd.to_datetime(turnout_data['ds'], format='%Y')  # Convert to datetime
prophet_model = Prophet()
prophet_model.fit(turnout_data)

future = prophet_model.make_future_dataframe(periods=1, freq='Y')  # Predict next year
forecast = prophet_model.predict(future)

# Add predicted turnout to the dataset
forecast['ds'] = forecast['ds'].dt.year  # Convert Prophet's 'ds' column to year as integer
forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'Predicted_Valid_Votes'})
df1 = pd.merge(df1, forecast, left_on='Year', right_on='ds', how='left').drop(columns=['ds'])

# Prepare Features and Targets again (with predicted Valid Votes)
X = df1.drop(columns=['Winner', 'NPP', 'NDC', 'Others', 'Valid Votes'])
y_class = df1['Winner']  # Classification target
y_reg = df1[['NPP', 'NDC', 'Others']]  # Regression targets

# Save dummy variable names after training
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'Instances/feature_names.pkl')


# Train-test split for classification and regression
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train Models again
# Classification Model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

# Regression Model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_reg_train, y_reg_train)
y_reg_pred = reg.predict(X_reg_test)

# Evaluate Models again
# Classification
print("Classification Accuracy:", accuracy_score(y_class_test, y_class_pred))

# Regression
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print("Regression RMSE:", rmse)

# Predicted vs Actual for Validation
print("\n--- Classification: Predicted vs Actual ---")
print(pd.DataFrame({'Actual': y_class_test, 'Predicted': y_class_pred}))

print("\n--- Regression: Predicted vs Actual ---")
for i, party in enumerate(['NPP', 'NDC', 'Others']):
    print(f"\n{party}:")
    print(pd.DataFrame({'Actual': y_reg_test.iloc[:, i], 'Predicted': y_reg_pred[:, i]}))

import joblib

# Save the classification model
joblib.dump(clf, 'Instances\ElecClassification_model.pkl')

# Save the regression model
joblib.dump(reg, 'Instances\ElecRegression_model.pkl')

