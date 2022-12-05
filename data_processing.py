from concurrent.futures import process
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import requests
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import imblearn
import tensorflow as tf
from sklearn.mixture import GaussianMixture

#from predict import create_predict_data

pd.set_option('display.max_rows', 500)

# Only processes 1m data for now


def add_adx(data, interval=14):
    #data.ta.adx(cumulative=True, append=True)
    adx_cols = ta.adx(data["High"], data["Low"],
                      data["Close"], length=interval)

    adx_cols = adx_cols.rename({"DMP_14": "+DM", "DMN_14": "-DM"})

    data = data.join(adx_cols)

    return data


def add_rsi(data, interval=14):
    rsi = ta.rsi(data["Close"], length=interval)

    rsi = rsi.rename("RSI")

    data = data.join(rsi)

    return data


def add_macd(data, interval=14):
    macd = ta.macd(data["Close"])

    macd = macd.drop("MACDs_12_26_9", axis=1)

    macd = macd.rename({"MACD_12_26_9": "MACD", "MACDh_12_26_9": "MACD_H"})

    data = data.join(macd)

    return data


def add_log_return(data, interval=1):
    log_r = ta.log_return(data["Close"], length=interval)

    log_r = log_r.rename("LOG_RETURN")

    data = data.join(log_r)


def add_percent_return(data, interval=1):
    percent_r = ta.percent_return(data["Close"], length=interval)

    percent_r = percent_r.rename("PERCENT_RETURN")

    data = data.join(percent_r)

    return data


###################################################### INDICATORS FOR HIGHER TIMEFRAMES #################################################################


def fear_and_greed():

    URL = "https://api.alternative.me/fng/?limit=0"
    r = requests.get(url=URL)
    fng = r.json()
    fng = pd.DataFrame(fng['data'])
    fng["timestamp"] = fng["timestamp"].astype(int)
    fng["value"] = fng["value"].astype(int)
    for index, row in fng.iterrows():
        fng.loc[index, "timestamp"] = datetime.utcfromtimestamp(
            fng.loc[index, "timestamp"])
        fng.loc[index, "timestamp"] = fng.loc[index,
                                              "timestamp"].strftime("%Y-%m-%d")
    fng.rename(columns={"timestamp": "Date", "value": "FnG"}, inplace=True)

    return fng.drop(["time_until_update", "value_classification"], axis=1)


def confirmation_time(data):
    df = pd.DataFrame({"Date": [], "Confirmation Time": []})
    for i in range(len(data)):
        for j in range(3):
            date = datetime.strptime(data.loc[i, "Timestamp"][:-9], "%Y-%m-%d")
            date = date + timedelta(days=j)
            # print(str(date)[:-9])
            df = df.append({"Date": str(date)[
                           :-9], "Confirmation Time": data.loc[i, "median-confirmation-time"]}, ignore_index=True)
    return df


def network_transactions(data):
    df = pd.DataFrame({"Date": [], "Transactions": []})

    for i in range(len(data)):
        for j in range(3):
            date = datetime.strptime(data.loc[i, "Timestamp"][:-9], "%Y-%m-%d")
            date = date + timedelta(days=j)
            # print(str(date)[:-9])
            df = df.append({"Date": str(date)[
                           :-9], "Transactions": data.loc[i, "n-transactions"]}, ignore_index=True)
    return df


def miners_revenue(data):
    df = pd.DataFrame({"Date": [], "Miners Revenue": []})

    for i in range(len(data)):
        for j in range(3):
            date = datetime.strptime(data.loc[i, "Timestamp"][:-9], "%Y-%m-%d")
            date = date + timedelta(days=j)
            df = df.append({"Date": str(date)[
                           :-9], "Miners Revenue": data.loc[i, "miners-revenue"]}, ignore_index=True)

    return df


def standardize_col(data, col):
    df = data.copy()
    df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

############################################################ DATA MANIPULATION #######################################################################

# start and end are complete dates, assumes that data has a UNIX/timestamp column            
def get_data_between(data, start, end):
    start_unix = datetime.strptime(start, "%d/%m/%Y").timestamp()
    
    end_unix = datetime.strptime(end, "%d/%m/%Y").timestamp()
    
    #print("Start : {} || End : {}".format(start_unix, end_unix))
    
    #print(len(data))
    
    return data.loc[(data["Unix"] >= start_unix) & (data["Unix"] < end_unix)]

# Assumes that data1 is a train dataset and data2 is a separate test dataset 
def yearly_custom_splitter(data1, data2):
    train_data = data1.drop(["Date", "Unix", "Target"], axis=1)
    train_labels = data1["Target"]
    test_data = data2.drop(["Date", "Unix", "Target"], axis=1)
    test_labels = data2["Target"]
    variation_check = False

    try:
        train_data.drop("Target_Variation", axis=1, inplace=True)
        test_data.drop("Target_Variation", axis=1, inplace=True)
        variation_check = True
    except:
        pass

    if variation_check:
        return train_data, train_labels, test_data, test_labels, data2["Target_Variation"] 
    else:
        return train_data, train_labels, test_data, test_labels, None

# Returns a sample from data containing any label with equal distribution
def sample_equal_target(data, method="classic"):
    if method == "classic":
        classes = data["Target"].nunique()
        counts = data["Target"].value_counts().sort_index()
        
        #print(counts)
        
        #print("Before",data["Target"].value_counts())
        
        data["Weights"] = 0
        for i in range(classes):
            data["Weights"].loc[data["Target"] == i] = 1/(classes*counts[i])
            
        #print(data["Weights"].loc[data["Target"] == 0])
            
        #sample = data.sample(int(len(data)/2.5),weights=data["Weights"])
        sample = data.sample(60000,weights=data["Weights"])
        
        sample_labels = sample["Target"]
        sample.drop("Weights", inplace=True, axis=1)
        
        #print("After",sample["Target"].value_counts())
    elif method == "smote":
        sm = imblearn.over_sampling.SMOTE(random_state=42)
        sample, sample_labels = sm.fit_resample(data, data["Target"])
        sample.drop("Target", inplace=True, axis=1)
    
    return sample,sample_labels

def custom_splitter(data1, data2):
    train_data = data1.drop(["Date", "Unix", "Target"], axis=1)
    train_labels = data1["Target"]
    test_data = data2.drop(["Date", "Unix", "Target"], axis=1)
    test_labels = data2["Target"]
    variation_check = False

    try:
        train_data.drop("Target_Variation", axis=1, inplace=True)
        test_data.drop("Target_Variation", axis=1, inplace=True)
        variation_check = True
    except:
        pass

    if variation_check:
        return train_data, train_labels, test_data, test_labels, data2["Target_Variation"] 
    else:
        return train_data, train_labels, test_data, test_labels, None    
    
    
########################################################## Standard Data Processing Functions #################################################################################   
    

def merge_1m():
    data2017 = pd.read_csv("minute_data/BTC-2017min.csv")
    data2018 = pd.read_csv("minute_data/BTC-2018min.csv")
    data2019 = pd.read_csv("minute_data/BTC-2019min.csv")
    data2020 = pd.read_csv("minute_data/BTC-2020min.csv")
    data2021 = pd.read_csv("minute_data/BTC-2021min.csv")

    data_by_minute = pd.concat(
        [data2017, data2018, data2019, data2020, data2021], axis=0)
    # data_by_minute = data_by_minute.sort_values(by="Unix", ascending=True, inplace=False) # ??????????????????!?!¿!?¿!?¿!?¿!?¿!?!¿?!¿?!¿?¿!?¿!?¿!?¿
    data_by_minute.to_csv("minute_data/BTC-USD_1m.csv", index=False)

# 1 minute data processing


def process_minute_data(write=True):

    data = pd.read_csv("minute_data/BTC-USD_1m.csv")
    data.sort_values(by="Unix", ascending=True,
                     inplace=True, ignore_index=True)
    data = add_percent_return(data)
    data = add_log_return(data)
    data = add_rsi(data)
    data = add_macd(data)
    data = add_adx(data)
    #data.drop("Adj Close",axis=1,inplace=True)
    columns_to_standardize = ["Volume USD", "MACD", "MACD_H",
                              "RSI", "PERCENT_RETURN", "LOG_RETURN", "ADX14", "-DM", "+DM"]
    means, std = data[columns_to_standardize].mean(
    ), data[columns_to_standardize].std()

    '''
    confirmation = confirmation_time(pd.read_csv("median-confirmation-time.csv"))
    transactions = network_transactions(pd.read_csv("n-transactions.csv"))
    miners_revenue = miners_revenue(pd.read_csv("miners-revenue.csv"))
    fear_and_greed = fear_and_greed()
    data = pd.merge(data,confirmation,on="Date")
    data = pd.merge(data,transactions,on="Date")
    data = pd.merge(data,miners_revenue,on="Date")
    data = pd.merge(data,fear_and_greed,on="Date")
    data = standardize_col(data,"Confirmation Time")
    data = standardize_col(data,"Transactions")
    data = standardize_col(data,"Miners Revenue")
    data = standardize_col(data,"FnG")
    '''

    if write:
        data.dropna(inplace=True)
        data = standardize_col(data, "Volume USD")
        data = standardize_col(data, "MACD")
        data = standardize_col(data, "MACD_H")
        print(data)
        data = standardize_col(data, "RSI")
        data = standardize_col(data, "Variation")
        data = standardize_col(data, ["ADX14", "-DM", "+DM"])
        data.drop(["Volume BTC"], axis=1, inplace=True)
        data.dropna(inplace=True)
        data.to_csv("minute_data/BTC-USD_1M_SIGNALS.csv", index=False)

    return columns_to_standardize, means, std


if __name__ == "__main__":
    process_minute_data()
