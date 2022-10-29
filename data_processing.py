from concurrent.futures import process
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import requests
import pandas_ta as ta

pd.set_option('display.max_rows', 500)

# Only processes 1m data for now


def add_adx(df: pd.DataFrame(), interval: int = 14):
    df['-DM'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['+DM'] = np.where((df['+DM'] > df['-DM']) &
                         (df['+DM'] > 0), df['+DM'], 0.0)
    df['-DM'] = np.where((df['-DM'] > df['+DM']) &
                         (df['-DM'] > 0), df['-DM'], 0.0)
    df['TR_TMP1'] = df['High'] - df['Low']
    if "Adj Close" in df.columns:
        df['TR_TMP2'] = np.abs(df['High'] - df['Adj Close'].shift(1))
        df['TR_TMP3'] = np.abs(df['Low'] - df['Adj Close'].shift(1))
    else:
        df['TR_TMP2'] = np.abs(df['High'] - df['Close'].shift(1))
        df['TR_TMP3'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
    df['TR'+str(interval)] = df['TR'].rolling(interval).sum()
    df['+DMI'+str(interval)] = df['+DM'].rolling(interval).sum()
    df['-DMI'+str(interval)] = df['-DM'].rolling(interval).sum()
    df['+DI'+str(interval)] = df['+DMI'+str(interval)] / \
        df['TR'+str(interval)]*100
    df['-DI'+str(interval)] = df['-DMI'+str(interval)] / \
        df['TR'+str(interval)]*100
    df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] -
                                     df['-DI'+str(interval)])
    df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
    df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
    df['ADX'+str(interval)] = df['DX'].rolling(interval).mean()
    df['ADX'+str(interval)] = df['ADX'+str(interval)
                                 ].fillna(df['ADX'+str(interval)].mean())
    del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR' +
                                                                  str(interval)]
    del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
    del df['DI'+str(interval)], df['-DMI'+str(interval)]
    del df['+DI'+str(interval)], df['-DI'+str(interval)]
    del df['DX']
    df.rename({'ADX'+str(interval) : 'ADX_'+str(interval)})
    return df


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

"""
def add_rsi(data):
    df = data.copy()

    df["Gains"] = df["Variation"] * (df["Variation"] >= 0)
    df["Losses"] = df["Variation"] * (df["Variation"] < 0)
    df["Avg Gain"] = df["Gains"].rolling(14).mean()
    df["Avg Loss"] = df["Losses"].rolling(14).mean()
    for i in range(14):
        df["Avg Gain"].loc[i] = df["Gains"][0:i].mean()
        df["Avg Loss"].loc[i] = df["Losses"][0:i].mean()

    df["Avg Loss"] = -1 * df["Avg Loss"]

    df['RSI'] = 0

    # df['RSI'] = 100 - (100 / (1 + ))

    df['RSI'] = 100 - (100 / (1 + df['Avg Gain'] / df['Avg Loss']))

    df = df.drop(["Gains", "Losses", "Avg Gain", "Avg Loss"], axis=1)

    #df['RSI'].loc[0:14] = 100 - (100 / (1 + df['Avg Gain'].loc[0:14] / df['Avg Loss'].loc[0:14]))

    return df
    """
    
def add_rsi (data, time_window=14):
    diff = data["Close"].diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    data["RSI"] = rsi
    return data    
    
    
    #data.ta.rsi(cumulative=True, append=True)


def add_macd(data):
    df = data.copy()

    df['MACD'] = df["Close"].ewm(span=12, adjust=False).mean(
    ) - df["Close"].ewm(span=26, adjust=False).mean()

    df['MACD_SIGNAL'] = df["MACD"].ewm(span=9, adjust=False).mean()

    df['MACD_H'] = df["MACD"] - df["MACD_SIGNAL"]

    df.drop(["MACD_SIGNAL"], axis=1, inplace=True)

    return df


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
    data.sort_values(by="Unix", ascending=True, inplace=True, ignore_index=True)
    data["Variation"] = (
        data["Close"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data = add_rsi(data)
    data = add_macd(data)
    data = add_adx(data)
    #data.drop("Adj Close",axis=1,inplace=True)
    columns_to_standardize = ["Volume USD", "MACD", "MACD_H", "RSI", "Variation", "ADX14", "-DM", "+DM"]
    means, std = data[columns_to_standardize].mean(), data[columns_to_standardize].std()
    
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