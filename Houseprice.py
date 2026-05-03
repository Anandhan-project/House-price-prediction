import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

dataset = pd.read_excel("fata/HousePricePrediction.xlsx")


object_cols = dataset.select_dtypes(include=['object']).columns
print("Categorical variables:", len(object_cols))

int_ = dataset.select_dtypes(include=['int64']).columns
print("Integer variables:", len(int_))

fl_cols = dataset.select_dtypes(include=['float64']).columns
print("Float variables:", len(fl_cols))

numerical_dataset = dataset.select_dtypes(include=['int64', 'float64'])
dataset = pd.read_excel("fata/HousePricePrediction.xlsx")


plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
print("Heatmap saved as correlation_heatmap.png")


unique_values = []
for col in object_cols:
  unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)

plt.figure(figsize=(18, 18))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

dataset.drop(['Id'],axis=1,)

dataset['SalePrice'] = dataset['SalePrice'].fillna(
  dataset['SalePrice'].mean())

new_dataset = dataset.dropna()

new_dataset.isnull().sum()

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
      len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

#vector
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)
print("SVM:")
print( mean_absolute_percentage_error(Y_valid, Y_pred))

#random forest
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
print("\nRandomo forest:")
print(mean_absolute_percentage_error(Y_valid, Y_pred))

#linearReg
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print("\nLenear regression:\n")
print(mean_absolute_percentage_error(Y_valid, Y_pred))

def calculate_metrics(Y_valid, Y_pred, dataset="Test"):
    #def calculate_metrics(y_true, y_pred, dataset_name="Test"):
    mae = mean_absolute_error(Y_valid, Y_pred)
    mse = mean_squared_error(Y_valid, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_valid, Y_pred)
    mape = mean_absolute_percentage_error(Y_valid, Y_pred)
    
      # Adjusted R-squared
    n = len(Y_valid)
    p = 1  # number of features (adjust as needed)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f"\n{'='*50}")
    print(f"{dataset} Set Metrics")
    print(f"{'='*50}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")
    print(f"MAPE: {mape:.2%}")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Then call it
results = calculate_metrics(Y_valid, Y_pred, "Linear Regression")