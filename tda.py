import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("Train data")
print(train_data.info())
print("Test data")
print(test_data.info())

print("Train data prev")
print(train_data.head())
print("Test data prev")
print(test_data.head())

print("Missing values in train")
print(train_data.isnull().sum())
print("Missing values in test")
print(test_data.isnull().sum())

train_data.fillna(train_data.mean(numeric_only=True), inplace=True)
test_data.fillna(test_data.mean(numeric_only=True), inplace=True)
category_clms = train_data.select_dtypes(include=['object']).columns
for col in category_clms:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    if col in test_data.columns:
        test_data[col].fillna(test_data[col].mode()[0], inplace=True)

for col in category_clms:
    train_data[col] = train_data[col].astype('category').cat.codes
    if col in train_data.columns:
        test_data[col] = test_data[col].astype('category').cat.codes



x = train_data.drop(['latitude', 'longitude','id'], axis=1)
y = train_data[['latitude', 'longitude']]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

y_val_pred = model.predict(x_val)

mae = mean_absolute_error(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation MAE: {mae}, RMSE: {rmse}")

x_test = test_data.drop(['id'], axis=1)
x_test = x_test[x_train.columns]
test_pred = model.predict(x_test)

test_data['latitude'] = test_pred[:, 0]
test_data['longitude'] = test_pred[:, 1]

submission = test_data[['id', 'latitude', 'longitude']]
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")