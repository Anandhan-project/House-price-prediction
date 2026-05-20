import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


dataset = pd.read_excel("fata/HousePricePrediction.xlsx")


object_cols = dataset.select_dtypes(include=['object']).columns

int_ = dataset.select_dtypes(include=['int64']).columns

fl_cols = dataset.select_dtypes(include=['float64']).columns

numerical_dataset = dataset.select_dtypes(include=['int64', 'float64'])


dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()
new_dataset.isnull().sum()

s = (new_dataset.dtypes == 'object')
object_cols = new_dataset.select_dtypes(include=['object']).columns

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]),index=new_dataset.index,columns=OH_encoder.get_feature_names_out(object_cols))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)


#random forest
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

#linearReg
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)



root = tk.Tk()
root.title("House Price Prediction")
root.geometry("600x600")
entries={}

def create_input_fields():
    row = 0
    for col in X.columns[:5]:
        entry = tk.Entry(root)
        entries[col] = entry
        row += 1

create_input_fields()

tk.Label(root, text="Area").grid(row=0, column=2)
LotArea = tk.Entry(root)
LotArea.grid(row=0, column=3)

tk.Label(root, text="Year Built").grid(row=3, column=2)
YearBuilt = tk.Entry(root)
YearBuilt.grid(row=3, column=3)

tk.Label(root, text="Year Modified").grid(row=5, column=2)
YearRemodified = tk.Entry(root)
YearRemodified.grid(row=5, column=3)

tk.Label(root, text="Dwelling Type").grid(row=7, column=2)
fam_options = ["1 family ","2 family ","3 family ","4 family ","5 family "]  # you can expand this list
fam_var = tk.StringVar(value=fam_options[0])  # default value

fam_type = ttk.Combobox(root, textvariable=fam_var, values=fam_options, state="readonly")
fam_type.grid(row=7, column=3)

tk.Label(root, text="Exterior Covering").grid(row=9, column=2)
ext_options = ["VinylSd ","MetalSd","Wd Sdng","HdBoard","BrkFace","WdShing","CemntBd","Plywood","AsbShng","Stucco","BrkComm","AsphShn","Stone","ImStucc","CBlock"]  
ext_var = tk.StringVar(value=ext_options[0]) 

ext_type = ttk.Combobox(root, textvariable=ext_var, values=ext_options, state="readonly")
ext_type.grid(row=9, column=3)


def predict_price():
    try:
        area = float(LotArea.get())
        yearbuild = int(YearBuilt.get())
        modified = int(YearRemodified.get())
        fam_ftype = fam_type.get()
        ext_ftype = ext_type.get()
        stay_map = {
            "1 family": 1,
            "2 family": 2,
            "3 family": 3,
            "4 family": 4,
            "5 family": 5
        }
        fam_value = stay_map.get(fam_ftype, 0)
        ext_map = {
            "VinylSd": 1,
            "MetalSd": 2,
            "Wd Sdng": 3,
            "HdBoard": 4,
            "BrkFace": 5,
            "WdShing": 6,
            "CemntBd": 7,
            "Plywood": 8,
            "AsbShng": 9,
            "Stucco": 10,
            "BrkComm": 11,
            "AsphShn": 12,
            "Stone": 13,
            "ImStucc": 14,
            "CBlock": 15
        }
        ext_value = ext_map.get(ext_ftype, 0)

        input_df = pd.DataFrame([{
            "LotArea": area,
            "YearBuilt": yearbuild,
            "YearRemodAdd": modified,
            "bldgtype": fam_value,
            "Exterior1st":ext_value
        }])

        final_input = input_df.reindex(columns=X.columns, fill_value=0)

        prediction = model_RFR.predict(final_input)
        price_per_sq_area = prediction[0] / area if area > 0 else 0

        result_label.config(
            text=f"Price: {prediction[0]:,.2f} Dollar\n"
                 f"{price_per_sq_area:,.2f} Dollar per sq Feet in California")

    

    except:
        messagebox.showerror("Error", "Please enter valid numeric values")

# Button
tk.Button(root, text="Predict Price", command=predict_price, bg="green", fg="white").grid(row=10, column=3, columnspan=2, pady=20)

# Result label
result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12))
result_label.grid(row=11, column=3, columnspan=2)

root.mainloop()
