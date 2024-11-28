import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Intern\House_train.csv')
print(df.shape)
print(df.sample(5))

print(df.columns)
pd.options.display.max_columns = None
df.info()

Category_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
Numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

df = pd.get_dummies(df, drop_first=True)  
df = df.dropna()  

X = df.drop(columns=['SalePrice'])  
y = df['SalePrice']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ["Linear Regression", LinearRegression()],
    ["Decision Tree Regressor", DecisionTreeRegressor()],
    ["RandomForestRegressor", RandomForestRegressor()],
    ["Gradient Boosting Regressor", GradientBoostingRegressor()]
]

model_performance = {}

for model_name, model in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_performance[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }
    
    print(f"{model_name} - MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

performance_df = pd.DataFrame(model_performance).T
print(performance_df)

performance_df[['MAE', 'MSE', 'R2']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

best_model_name = performance_df['R2'].idxmax()
best_model = [model for name, model in models if name == best_model_name][0]

y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title(f'Residuals of {best_model_name}')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_best,
    'Residuals': residuals
})

predictions_df.to_csv(r'C:\Users\HP\OneDrive\Desktop\Intern\House_predictions.csv', index=False)
