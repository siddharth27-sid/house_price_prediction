import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv('House Price Prediction Dataset.csv')
df = df.drop("B", axis=1)
df = df.rename(columns={
    "CRIM": "CrimeRate",
    "ZN":"ResidentialLand",
    "INDUS": "IndustrialArea",
    "CHAS": "NearCharlesRiver",
    "NOX": "AirPollution",
    "RM": "Rooms",
    "AGE": "HouseAge",
    "DIS": "DistanceToEmployment",
    "RAD": "HighwayAccessibility",
    "TAX": "PropertyTax",
    "PTRATIO": "PupilTeacherRatio",
    "LSTAT": "LowerIncomePct",
    "MEDV": "HousePrice"
})
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

corr = df.select_dtypes(include="number").corr()
print(corr["HousePrice"].sort_values(ascending=False))


plt.hist(df['HousePrice'],color='blue',bins=30,edgecolor='black', alpha=0.7)
plt.xlabel('HousePrice ($1000s)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()


plt.scatter(df["Rooms"], df["HousePrice"], s=10, alpha=0.4,color='green')
plt.xlabel("Number of Rooms")
plt.ylabel("House Price ($1000s)")
plt.title("Rooms vs House Price")
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

print(df.duplicated().sum())

X=df[["CrimeRate",
      "ResidentialLand",
      "IndustrialArea",
      "NearCharlesRiver",
      "AirPollution",
      "Rooms",
      "HouseAge",
      "DistanceToEmployment",
      "HighwayAccessibility",
      "PropertyTax",
      "PupilTeacherRatio",
      "LowerIncomePct"]]
Y=df["HousePrice"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=57)

#linear regression model
model_1 = LinearRegression()
model_1.fit(X_train, Y_train)
Y_pred = model_1.predict(X_test)

for actual, predicted in zip(Y_test[:10], Y_pred[:10]):
    print(f"Actual: {actual:.2f}  Predicted: {predicted:.2f}")


print(f"MAE of linear regression: {mean_absolute_error(Y_test, Y_pred):.2f}")
print(f"MSE of linear regression: {mean_squared_error(Y_test, Y_pred):.2f}")
print(f"R² of linear regression: {r2_score(Y_test, Y_pred):.2f}")

#decision tree model

dt_model_2 = DecisionTreeRegressor(random_state=57)

dt_model_2.fit(X_train, Y_train)

dt_pred = dt_model_2.predict(X_test)

print("Decision Tree")
print(f"MAE of decision tree: {mean_absolute_error(Y_test, dt_pred):.2f}")
print(f"MSE of decision tree: {mean_squared_error(Y_test, dt_pred):.2f}")
print(f"R² of decision tree: {r2_score(Y_test, dt_pred):.2f}")

#Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=57
)

rf_model.fit(X_train, Y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest")
print(f"MAE : {mean_absolute_error(Y_test, rf_pred):.2f}")
print(f"MSE : {mean_squared_error(Y_test, rf_pred):.2f}")
print(f"R²  : {r2_score(Y_test, rf_pred):.2f}")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("Important features to predict house price") 

importance["Importance"] = importance["Importance"] * 100

print(importance)