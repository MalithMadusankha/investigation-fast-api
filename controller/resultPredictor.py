import pandas as pd
import numpy as np
import joblib
from typing import List, Dict
from models.dataModel import DataModel
# Load the trained model and vectorizer
model = joblib.load("controller/RandomForestRegressorModel.pickle")
import os

current_path = os.getcwd()
current_path = current_path + "/controller/Datasets"

data = pd.read_csv(current_path+"/fraudDataset.csv")
# Split the dataset into input features (X) and target variable (y)
X = data.drop('fraud', axis=1)

# Convert categorical variables into numerical using one-hot encoding
X = pd.get_dummies(X)

dataColombo = pd.read_csv(current_path+'/fraudDatasetColombo.csv')
dataGampaha = pd.read_csv(current_path+'/fraudDatasetGampaha.csv')

    
def PredictorForList(dataObj: DataModel):
    
    s1 = ""
    
    if dataObj.district == "colombo":
        s1 = dataColombo
        s1["district"] = "colombo"
    
    if dataObj.district == "gampaha":
        s1 = dataGampaha
        s1["district"] = "gampaha"
    
    s1["year"] = dataObj.year  

    # Create an empty list to store the predictions
    predictions = []

    # Iterate over each sample in the input data
    for _, sample in s1.iterrows():
        # Extract feature values from the sample row
        provience = sample["provience"]
        district = sample["district"]
        city = sample["city"]
        month = sample["month"]
        year = sample["year"]


        # Create a DataFrame for the sample
        S1 = pd.DataFrame([[provience, district, city, month, year]],
                        columns=['provience', 'district', 'city', 'month', 'year'])

        # Convert categorical variables into numerical using one-hot encoding
        S1 = pd.get_dummies(S1)

        # Ensure that the input data has the same features as the training data
        missing_cols = set(X.columns) - set(S1.columns)
        for col in missing_cols:
            S1[col] = 0

        # Reorder the columns to match the training data
        S1 = S1[X.columns]

        # Predict on the input data
        S1_pred = model.predict(S1)

        # Append the prediction to the list
        predictions.append(S1_pred)
        

    flat_array = [item for sublist in predictions for item in sublist]

    s1["fruad"] = flat_array


    # Find the row with the maximum value in the 'fraud' column
    max_fraud_row = s1[s1["fruad"] == s1['fruad'].max()]

    # Retrieve the city from the maximum fraud row
    max_fraud_city = max_fraud_row['city'].values[0]

    # Print the maximum fraud and the relevant city
    print("Maximum Fraud:", s1['fruad'].max())
    print("City with Maximum Fraud:", max_fraud_city)

    return max_fraud_city

def PredictorForCity(dataObj: DataModel):

    # Define the input data for prediction
    s1 = [dataObj.provience, dataObj.district, dataObj.city, dataObj.month, dataObj.year]


    # Convert categorical variables into numerical using one-hot encoding
    S1 = pd.DataFrame([s1], columns=['provience', 'district', 'city', 'month', 'year'])
    S1 = pd.get_dummies(S1)

    # Ensure that the input data has the same features as the training data
    missing_cols = set(X.columns) - set(S1.columns)
    for col in missing_cols:
        S1[col] = 0

    # Reorder the columns to match the training data
    S1 = S1[X.columns]

    # Predict on the input data
    S1_pred = model.predict(S1)
       # Convert NumPy array to a list
    S1_pred = S1_pred.tolist()
    print(S1_pred)


    return S1_pred
