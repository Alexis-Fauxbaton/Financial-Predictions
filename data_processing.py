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

############################################################ PREDICTION DATA CREATION ################################################################


def standard_labeling(data, max_days, target_range):
    predict_data = data.copy().drop(
        ["Open", "Close", "High", "Low", "Symbol"], axis=1)
    for i in range(1, max_days):  # 2jours
        #self.predict_data[["Variation-{}".format(i),"Vol-{}".format(i),"RSI-{}".format(i),"MACD-{}".format(i),"MACD_H-{}".format(i),"CONF-{}".format(i),"TRANS-{}".format(i),"REV-{}".format(i),"FnG-{}".format(i)]] = data[["Variation","Volume","RSI","MACD","MACD_H","Confirmation Time","Transactions","Miners Revenue","FnG"]].shift(i)
        #self.predict_data[["Variation-{}".format(i),"Vol-{}".format(i),"RSI-{}".format(i),"MACD-{}".format(i),"MACD_H-{}".format(i),"CONF-{}".format(i),"TRANS-{}".format(i),"REV-{}".format(i),"FnG-{}".format(i), "ADX-{}".format(i), "+DM-{}".format(i), "-DM-{}".format(i)]] = data[["Variation","Volume","RSI","MACD","MACD_H","Confirmation Time","Transactions","Miners Revenue","FnG","ADX14","+DM","-DM"]].shift(i)
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i), "Regime-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H", "Regime"]].shift(i)
    #self.predict_data["Target"] = (data["Variation"].shift(-1) >= 0)
    predict_data["Target"] = (
        data["Close"].shift(-target_range) - data["Close"] >= 0)
    predict_data["Target"] = np.where(predict_data["Target"] == True, 1, 0)
    predict_data["Target_Variation"] = (
        data["Close"].shift(-target_range) - data["Close"])/data["Close"]
    predict_data.dropna(inplace=True)
    predict_data.reset_index(inplace=True, drop=True)
    predict_data = predict_data[0:len(predict_data)-target_range]

    return predict_data


def meta_labeling_process_par(data, index, target_range):
    rows = data.shape[0]
    if index > rows - (target_range+1):
        return np.nan
    slice_data = data[index:index+target_range+1]
    #max_range = data[["Unix","High"]].loc[data["High"] == data["High"][index:index+max_days-1].max()].reset_index(drop=True)
    #min_range = data[["Unix","Low"]].loc[data["Low"] == data["Low"][index:index+max_days-1].min()].reset_index(drop=True)
    max_unix = None
    min_unix = None
    first = None

    max_range = slice_data[["Unix", "High"]].loc[slice_data["High"]
                                                 == slice_data["High"].max()].reset_index(drop=True)
    min_range = slice_data[["Unix", "Low"]].loc[slice_data["Low"]
                                                == slice_data["Low"].min()].reset_index(drop=True)

    if (max_range["High"].loc[0]/slice_data["Close"].loc[index]) > 1.005:
        max_unix = max_range["Unix"].loc[0]
    if (min_range["Low"].loc[0]/slice_data["Close"].loc[index]) < 0.995:
        min_unix = min_range["Unix"].loc[0]

    # Encoding max/min state to easily code below which one was encountered first
    state_max = 0
    state_min = 0
    if max_unix != None:
        state_max = 1
    if min_unix != None:
        state_min = 1

    state = 2*state_max + state_min

    if state == 0:
        if slice_data.loc[index+target_range, "Close"] > slice_data.loc[index, "Close"]:
            first = 1
        else:
            first = -1
    elif state == 1:
        first = -2
    elif state == 2:
        first = 2
    else:
        if min_unix < max_unix:
            first = -2
        else:
            first = 2

    if index % 1000 == 0:
        print("Itération {}/{}".format(index, data.shape[0]), end='\r')

    return first


def meta_labeling_2_par(data, max_days, target_range):
    predict_data = data.copy().drop(
        ["Open", "Close", "High", "Low", "Symbol"], axis=1)
    size = data.shape[0]
    first = None
    for i in range(1, max_days):  # 2jours
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i), "Regime-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H", "Regime"]].shift(i)

    target = np.zeros(size)
    # Try to parallelize later
    #indexes = [i for i in range(size)]

    args = [(data, i, target_range) for i in data.index]

    pool = Pool(os.cpu_count())

    print("Parallelizing Meta Labeling, using {} threads...".format(os.cpu_count()))

    target = pool.starmap(meta_labeling_process_par, args)

    predict_data["Target"] = target
    predict_data["Target1"] = (
        data["Close"].shift(-target_range) - data["Close"] >= 0)
    predict_data["Target1"] = np.where(predict_data["Target1"] == True, 1, 0)
    predict_data["Target_Variation"] = (
        data["Close"].shift(-target_range) - data["Close"])/data["Close"]
    predict_data.dropna(inplace=True)
    predict_data.reset_index(inplace=True, drop=True)
    predict_data = predict_data[0:len(predict_data)-target_range]

    return predict_data

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

############################################################ BACKTEST FUNCTIONS ######################################################################

def simple_strategy_backtest(test_set):
    init_assets = 100
    assets = init_assets
    test_set = test_set.copy()
    test_set.reset_index(inplace=True, drop=True)
    val = [1]
    
    if test_set["Target"].nunique() in [2,3]:
        val = [1]
    elif test_set["Target"].nunique() == 4:
        val = [1,2]
        
    backtest_set = test_set.loc[test_set["Prediction"].isin(val)]
    backtest_set["Assets"] = 100
    backtest_set.reset_index(inplace=True, drop=True)
    backtest_set_assets = [0 for i in range(len(backtest_set))]
    for index,row in backtest_set.iterrows():
        assets = assets * (1+row["Target_Variation"])
        backtest_set.loc[index,"Assets"] = assets
        backtest_set_assets[index] = assets
    print(10*"#")
    print("Simple Strategy Backtest")
    print("Backtest Time Interval : {} - {}".format(test_set.loc[0,"Date"], test_set.loc[len(test_set)-1,"Date"]))
    print("Value of assets at the beginning of simple strategy : {}".format(init_assets))
    print("Value of assets at the end of simple strategy : {}".format(assets))
    N = np.arange(0,len(backtest_set_assets))
    plt.figure(figsize=[20,10])
    plt.plot(N,backtest_set_assets,label="Évolution du portefeuille")
    plt.show()

############################################################ DATA HANDLER CLASS ######################################################################


class DataHandler:
    def __init__(self, csv_path=None):
        self.data_scaler = None
        self.max_days = None
        self.target_range = None
        self.columns_to_scale = None
        self.predict_data = None
        if csv_path == None:
            self.data = None
        else:
            try:
                self.data = pd.read_csv(csv_path)
                self.data.sort_values(
                    by="Unix", ascending=True, inplace=True, ignore_index=True)
                # TODO ADD CODE TO DETERMINE TIMEFRAME FROM DIFFERENCE IN TIME BETWEEN TWO FIRST ELEMENTS OF DATA
                self.timeframe = ''
            except:
                print("Failed to import data from path {}".format(csv_path))

    def head(self):
        if self.data != None:
            print(self.data.head())

    def load(self, csv_path):
        del self.data

        print("Successfully deleted old data")

        try:
            self.data = pd.read_csv(csv_path)
        except:
            print("Failed to import data at path {}".format(csv_path))

    def get_data(self):
        return self.data

    def get_predict_data(self):
        return self.predict_data

    def plot(self, x, y):
        self.data.plot(x, y, figsize=[20,10])
        return

    def add_indicators(self):
        self.data = add_percent_return(self.data)
        self.data = add_log_return(self.data)
        self.data = add_rsi(self.data)
        self.data = add_macd(self.data)
        self.data = add_adx(self.data)

    #Assess different market regimes
    def add_gaussian_mixture(self):
        data = self.data.copy()[[i%10 == 0 for i in range(len(self.data))]]
        if self.max_days == None:
            raise ("Error, lookback prediction interval has not been initialized")
        else:
            self.gm = GaussianMixture(2, random_state=10, init_params="kmeans")
            mean_std = self.data["Close"].rolling(self.max_days).std()
            self.data["STD30"] = mean_std
            self.data.dropna(inplace=True, axis=0)
            labels = self.gm.fit_predict(np.array(self.data["STD30"]).reshape(-1,1))
            self.data["Regime"] = labels
            self.data.drop("STD30", inplace=True, axis=1)
            self.data.reset_index(drop=True, inplace=True)

    def standardize_indicators(self, method="standard"):
        self.columns_to_scale = ["Volume USD", "MACD", "MACD_H",
                                 "RSI", "PERCENT_RETURN", "LOG_RETURN", "ADX14", "-DM", "+DM"]
        if method == "standard":
            if self.data_scaler == None:
                self.data_scaler = StandardScaler()
            self.data[self.columns_to_scale] = self.data_scaler.fit_transform(
                self.data[self.columns_to_scale])

    def create_predict_data(self, max_days=30, target_range=10, standard=True):
        print("Creating Predict DataFrame")
        if self.max_days == None:
            self.max_days = max_days
        if self.target_range == None:
            self.target_range = target_range
        read = True
        self.predict_data = None
        self.add_gaussian_mixture()
        if standard:
            read = False
            self.predict_data = standard_labeling(
                self.data, max_days, target_range)
        else:
            try:

                self.predict_data = pd.read_csv(
                    "processed_data/processed_1m_{}_{}.csv".format(self.max_days, self.target_range))                    
                #self.predict_data.drop("Target1", axis=1, inplace=True)

                #TODO Rewrite this part
                if -2 in self.predict_data["Target"].values:
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        2, 3)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        1, 2)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -1, 1)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -2, 0)
                elif -1 in self.predict_data["Target"].values:
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        1, 2)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        0, 1)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -1, 0)
            except Exception as e:
                print("Failed to load processed data, Reason : {}".format(e))
                read = False
                self.predict_data = meta_labeling_2_par(
                    self.data, max_days, target_range)
                #self.predict_data.drop("Target1", axis=1, inplace=True)
                if -2 in self.predict_data["Target"].values:
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        2, 3)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        1, 2)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -1, 1)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -2, 0)
                elif -1 in self.predict_data["Target"].values:
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        1, 2)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        0, 1)
                    self.predict_data["Target"] = self.predict_data["Target"].replace(
                        -1, 0)
            try:
                self.predict_data.drop("Target1", axis=1, inplace=True)
            except:
                pass

        if not read:
            self.predict_data = self.predict_data[[
                i % int(max_days/3) == 0 for i in range(len(self.predict_data))]]

        if (not standard) and (read == False):
            print("Writing Processed Data to memory...")
            self.predict_data.to_csv(
                "processed_data/processed_1m_{}_{}.csv".format(max_days, target_range), index=False)
        
    def fit_predict(self, train_start="1/1/2019", train_end="1/1/2021", test_start="1/1/2021", test_end="1/1/2023", max_days=30, target_range=10, labeling=True, equal_sampling=False, sampling_method="classic", epochs=10):
        if self.predict_data == None:
            self.create_predict_data(max_days, target_range, labeling)
        
        print("Predict Data : \n{}".format(self.predict_data))
        
        train_data = get_data_between(self.predict_data, train_start, train_end)
        
        test_data = get_data_between(self.predict_data, test_start, test_end)
        
        test_date = test_data["Date"]
        
        train_data, train_labels, test_data, test_labels, test_variation = yearly_custom_splitter(
        train_data, test_data)
        
        if sampling_method != 'none':
            if equal_sampling:
                train_data["Target"] = train_labels
                train_data, train_labels = sample_equal_target(train_data, method=sampling_method)
            #TODO Remove ?
            else:
                train_data, train_labels = train_data.drop("Target", axis=1), train_data["Target"]
            
        #test_data, test_labels = test_data.drop("Target", axis=1), test_data["Target"]
        
        outputs = train_labels.nunique()
        
        shape = (len(test_data.columns),)
    
        print("Shape of inputs for ML algorithm : {}".format(shape))
        
        if outputs == 2:
        
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=shape),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(2000, activation='relu'),
                tf.keras.layers.Dense(5000, activation='relu'),
                tf.keras.layers.Dense(2000, activation='relu'),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
            
            epochs = 10
        
        else:
            
            print("Using {} classes".format(outputs))
            
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=shape),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(2000, activation='relu'),
                tf.keras.layers.Dense(5000, activation='relu'),
                tf.keras.layers.Dense(2000, activation='relu'),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(outputs, activation='softmax')
            ])

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
            
            epochs = 15
            
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        #model training
    #    model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data,test_labels),callbacks=[tensorboard_callback], batch_size=64)
        model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2, callbacks=[tensorboard_callback], batch_size=64)
        
        model.evaluate(test_data, test_labels)
        
        probs = model.predict(test_data)
        
        if outputs > 2:
            preds = probs.argmax(axis=-1)
        else:
            preds = [1 if i>0.5 else 0 for i in probs]
        
        #print(test_labels, preds)
        print(tf.math.confusion_matrix(test_labels, preds))
        
        test_set = test_data.copy()

        test_set["Prediction"] = preds
        test_set["Target"] = test_labels
        test_set["Target_Variation"] = test_variation
        #Our test set is data2021 (Set to potentially change)
        test_set["Date"] = test_date
        
        if test_labels.nunique() == 3:
            test_set["Prediction"] = test_set["Prediction"].replace(0,-1)
            test_set["Prediction"] = test_set["Prediction"].replace(1,0)
            test_set["Prediction"] = test_set["Prediction"].replace(2,1)
        
        elif test_labels.nunique() == 4:
            test_set["Prediction"] = test_set["Prediction"].replace(0,-2)
            test_set["Prediction"] = test_set["Prediction"].replace(1,-1)
            test_set["Prediction"] = test_set["Prediction"].replace(2,1)
            test_set["Prediction"] = test_set["Prediction"].replace(3,2)
            
        #Backtest of simplest strategy
        simple_strategy_backtest(test_set)

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
