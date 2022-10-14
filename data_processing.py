import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import requests

def add_adx(df: pd.DataFrame(), interval: int=14):
  df['-DM'] = df['Low'].shift(1) - df['Low']
  df['+DM'] = df['High'] - df['High'].shift(1)
  df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
  df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)
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
  df['+DI'+str(interval)] = df['+DMI'+str(interval)] /   df['TR'+str(interval)]*100
  df['-DI'+str(interval)] = df['-DMI'+str(interval)] / df['TR'+str(interval)]*100
  df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] - df['-DI'+str(interval)])
  df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
  df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
  df['ADX'+str(interval)] = df['DX'].rolling(interval).mean()
  df['ADX'+str(interval)] =   df['ADX'+str(interval)].fillna(df['ADX'+str(interval)].mean())
  del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(interval)]
  del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
  del df['DI'+str(interval)], df['-DMI'+str(interval)]
  del df['+DI'+str(interval)], df['-DI'+str(interval)]
  del df['DX']
  return df

def fear_and_greed():

    URL = "https://api.alternative.me/fng/?limit=0"
    r = requests.get(url = URL)
    fng = r.json()
    fng = pd.DataFrame(fng['data'])
    fng["timestamp"] = fng["timestamp"].astype(int)
    fng["value"] = fng["value"].astype(int)
    for index,row in fng.iterrows():
        fng.loc[index,"timestamp"] = datetime.utcfromtimestamp(fng.loc[index,"timestamp"])
        fng.loc[index,"timestamp"] = fng.loc[index,"timestamp"].strftime("%Y-%m-%d")
    fng.rename(columns={"timestamp":"Date", "value":"FnG"},inplace=True)

    return fng.drop(["time_until_update","value_classification"],axis=1)

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

    #df['RSI'] = 100 - (100 / (1 + ))
    
    df['RSI'] = 100 - (100 / (1 + df['Avg Gain'] / df['Avg Loss']))

    df = df.drop(["Gains", "Losses", "Avg Gain", "Avg Loss"],axis=1)

    #df['RSI'].loc[0:14] = 100 - (100 / (1 + df['Avg Gain'].loc[0:14] / df['Avg Loss'].loc[0:14]))


    return df

def add_macd(data):
    df = data.copy()

    df['MACD'] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    df['MACD_SIGNAL'] = df["MACD"].ewm(span=9, adjust=False).mean()

    df['MACD_H'] = df["MACD"] - df["MACD_SIGNAL"]

    df.drop(["MACD_SIGNAL"], axis=1, inplace=True)

    return df

def confirmation_time(data):
    df = pd.DataFrame({"Date":[],"Confirmation Time":[]})
    for i in range(len(data)):
        for j in range(3):
            date = datetime.strptime(data.loc[i,"Timestamp"][:-9],"%Y-%m-%d")
            date = date + timedelta(days=j)
            #print(str(date)[:-9])
            df = df.append({"Date":str(date)[:-9],"Confirmation Time":data.loc[i,"median-confirmation-time"]},ignore_index=True)
    return df

def network_transactions(data):
    df = pd.DataFrame({"Date":[], "Transactions":[]})

    for i in range(len(data)):
        for j in range(3):
            date = datetime.strptime(data.loc[i,"Timestamp"][:-9],"%Y-%m-%d")
            date = date + timedelta(days=j)
            #print(str(date)[:-9])
            df = df.append({"Date":str(date)[:-9],"Transactions":data.loc[i,"n-transactions"]},ignore_index=True)
    return df

