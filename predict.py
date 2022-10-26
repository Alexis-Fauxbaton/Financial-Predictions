from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import data_processing
from multiprocessing import Pool
import os
import imblearn

#TODO USE PANDAS-TA LIB FOR ADDING TECHNICAL INDICATORS

max_days = 30
target_range = 10

def standard_labeling(data, max_days, target_range):
    predict_data = data.copy().drop(["Open", "Close", "High", "Low", "Symbol"], axis=1)
    for i in range(1, max_days):  # 2jours
        #predict_data[["Variation-{}".format(i),"Vol-{}".format(i),"RSI-{}".format(i),"MACD-{}".format(i),"MACD_H-{}".format(i),"CONF-{}".format(i),"TRANS-{}".format(i),"REV-{}".format(i),"FnG-{}".format(i)]] = data[["Variation","Volume","RSI","MACD","MACD_H","Confirmation Time","Transactions","Miners Revenue","FnG"]].shift(i)
        #predict_data[["Variation-{}".format(i),"Vol-{}".format(i),"RSI-{}".format(i),"MACD-{}".format(i),"MACD_H-{}".format(i),"CONF-{}".format(i),"TRANS-{}".format(i),"REV-{}".format(i),"FnG-{}".format(i), "ADX-{}".format(i), "+DM-{}".format(i), "-DM-{}".format(i)]] = data[["Variation","Volume","RSI","MACD","MACD_H","Confirmation Time","Transactions","Miners Revenue","FnG","ADX14","+DM","-DM"]].shift(i)
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H"]].shift(i)
    #predict_data["Target"] = (data["Variation"].shift(-1) >= 0)
    predict_data["Target"] = (
        data["Close"].shift(-target_range) - data["Close"] >= 0)
    predict_data["Target"] = np.where(predict_data["Target"] == True, 1, 0)
    predict_data["Target_Variation"] = (
        data["Close"].shift(-target_range) - data["Close"])/data["Close"]
    predict_data.dropna(inplace=True)
    predict_data.reset_index(inplace=True, drop=True)
    predict_data = predict_data[0:len(predict_data)-target_range]

    return predict_data


def meta_labeling(data, max_days, target_range):
    predict_data = data.copy().drop(["Open", "Close", "High", "Low", "Symbol"], axis=1)
    size = data.shape[0]
    first = None
    for i in range(1, max_days):  # 2jours
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H"]].shift(i)

    target = np.zeros(data.shape[0])
    # Try to parallelize later
    for index,row in data.iterrows():
        #index = row[0]
        slice_data = data[index:index+target_range+1]
        #max_range = data[["Unix","High"]].loc[data["High"] == data["High"][index:index+max_days-1].max()].reset_index(drop=True)
        #min_range = data[["Unix","Low"]].loc[data["Low"] == data["Low"][index:index+max_days-1].min()].reset_index(drop=True)
        max_unix = None
        min_unix = None

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
            first = 0
        elif state == 1:
            first = -1
        elif state == 2:
            first = 1
        else:
            if min_unix < max_unix:
                first = -1
            else:
                first = 1            

        #predict_data.at[index, "Target"] = first
        target[index] = first

        if index % 1000 == 0:
            print("Itération {}/{}".format(index, size), end='\r')

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

# To remove if parallel version works
def meta_labeling_2(data, max_days, target_range):
    predict_data = data.copy().drop(["Open", "Close", "High", "Low", "Symbol"], axis=1)
    size = data.shape[0]
    first = None
    for i in range(1, max_days):  # 2jours
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H"]].shift(i)

    target = np.zeros(size)
    # Try to parallelize later
    for index,row in data.iterrows():
        #index = row[0]
        slice_data = data[index:index+target_range+1]
        #max_range = data[["Unix","High"]].loc[data["High"] == data["High"][index:index+max_days-1].max()].reset_index(drop=True)
        #min_range = data[["Unix","Low"]].loc[data["Low"] == data["Low"][index:index+max_days-1].min()].reset_index(drop=True)
        max_unix = None
        min_unix = None

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
            if slice_data.loc[index+target_range,"Close"] > slice_data.loc[index,"Close"]:
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

        #predict_data.at[index, "Target"] = first
        target[index] = first
        
        if index % 1000 == 0:
            print("Itération {}/{}".format(index, size), end='\r')

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

def meta_labeling_process_par(data,index,target_range):
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
        if slice_data.loc[index+target_range,"Close"] > slice_data.loc[index,"Close"]:
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
    predict_data = data.copy().drop(["Open", "Close", "High", "Low", "Symbol"], axis=1)
    size = data.shape[0]
    first = None
    for i in range(1, max_days):  # 2jours
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H"]].shift(i)

    target = np.zeros(size)
    # Try to parallelize later
    #indexes = [i for i in range(size)]
        
    args = [(data,i,target_range) for i in data.index]
    
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


def create_predict_data(data, max_days=30, target_range=10, standard=True):
    read = True
    if standard:
        read = False
        predict_data = standard_labeling(data, max_days, target_range)
    else:
        try:
            
            predict_data = pd.read_csv("processed_data/processed_1m_{}_{}.csv".format(max_days,target_range))
            #predict_data.drop("Target1", axis=1, inplace=True)
            
            if -2 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(2,3)    
                predict_data["Target"] = predict_data["Target"].replace(1,2)
                predict_data["Target"] = predict_data["Target"].replace(-1,1)
                predict_data["Target"] = predict_data["Target"].replace(-2,0)
            elif -1 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(1, 2)
                predict_data["Target"] = predict_data["Target"].replace(0, 1)
                predict_data["Target"] = predict_data["Target"].replace(-1, 0)
        except:
            read = False
            predict_data = meta_labeling_2_par(data, max_days, target_range)
            #predict_data.drop("Target1", axis=1, inplace=True)
            if -2 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(2,3)    
                predict_data["Target"] = predict_data["Target"].replace(1,2)
                predict_data["Target"] = predict_data["Target"].replace(-1,1)
                predict_data["Target"] = predict_data["Target"].replace(-2,0)
            elif -1 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(1, 2)
                predict_data["Target"] = predict_data["Target"].replace(0, 1)
                predict_data["Target"] = predict_data["Target"].replace(-1, 0)
        try:
            predict_data.drop("Target1", axis=1, inplace=True)
        except:
            pass

    if not read:
        '''
        predict_data = predict_data[[
            i % int(max_days) == 0 for i in range(len(predict_data))]]
        '''
        predict_data = predict_data[[
            i % int(max_days/3) == 0 for i in range(len(predict_data))]]
            

    if (not standard) and (read==False):
        print("Writing Processed Data to memory...")
        predict_data.to_csv("processed_data/processed_1m_{}_{}.csv".format(max_days,target_range), index=False)

    return predict_data


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


def sample_equal_target(data):
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
    
    return sample,sample_labels


def train_and_return_model_for_2021_2022():
    data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")

    standard_labels = True

    predict_data = create_predict_data(data, max_days, target_range, standard_labels)

    data_19_20 = predict_data.loc[(predict_data["Unix"] >= 1546300800) & (
        predict_data["Unix"] < 1609459200)]

    data_2021 = predict_data.loc[predict_data["Unix"] >= 1609459200]

    train_data, train_labels, test_data, test_labels, test_variation = yearly_custom_splitter(
        data_19_20, data_2021)

    shape = (len(test_data.columns),)
    
    train_set = train_data.copy()
    train_set["Target"] = train_labels
        
    #Sampling same number of elements for each class to ensure no one is more likely to be found
    if (not standard_labels) and (train_labels.nunique() == 3):
        print("\nData points in training set before sampling :", len(train_set))
        
        train_data,train_labels = sample_equal_target(train_set)
        print("\nData points in training set after sampling :", len(train_labels))
        train_data.drop("Target",axis=1,inplace=True)
    else:
        print("\nData points in training set :", len(train_set))
    print("\nData points in validation set :", len(test_labels))
    print()
    #raise Exception("Stop")
    
    print(train_labels.value_counts())
    
    outputs = train_labels.nunique()
    
    epochs = 10
    
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
        
        epochs = 20
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model training
    model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data,test_labels),callbacks=[tensorboard_callback], batch_size=64)
    
    probs = model.predict(test_data)
    
    if outputs > 2:
        preds = probs.argmax(axis=-1)
    else:
        preds = [1 if i>0.5 else 0 for i in probs]
    
    #print(test_labels, preds)
    print(tf.math.confusion_matrix(test_labels, preds))
    
    return model

#To finish
def standardize_new_col(data, cols, means, stds):
    data[cols] = data[cols]

#We only buy when model tells us to and sell at the end of the period
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

if __name__ == "__main__":

    # In case we need this to get mean and std from columns to normalize new data
    #raw_data = pd.read_csv("minute_data/BTC-USD_1m.csv")

    data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")

    standard_labels = False

    predict_data = create_predict_data(data, 30, 10, standard_labels)

    data_19_20 = predict_data.loc[(predict_data["Unix"] >= 1546300800) & (
        predict_data["Unix"] < 1609459200)]

    data_2021 = predict_data.loc[predict_data["Unix"] >= 1609459200]

    train_data, train_labels, test_data, test_labels, test_variation = yearly_custom_splitter(
        data_19_20, data_2021)

    shape = (len(test_data.columns),)
    
    print("Shape of inputs for ML algorithm : {}".format(shape))
    
    train_set = train_data.copy()
    train_set["Target"] = train_labels
    
    #Sampling same number of elements for each class to ensure no one is more likely to be found
    if (not standard_labels):
        print("\nData points in training set before sampling :", len(train_set))
        try:
            #train_data,train_labels = sample_equal_target(train_set)
            sm = imblearn.over_sampling.SMOTE(random_state=42)
            train_data, train_labels = sm.fit_resample(train_set, train_labels)
            print("\nData points in training set after sampling :", len(train_labels))
            train_data.drop("Target",axis=1,inplace=True)
        except Exception as e:
            print("Sampling Failed, Reason : {}".format(e))
    else:
        print("\nData points in training set :", len(train_set))
    print("\nData points in validation set :", len(test_labels))
    print()
    #raise Exception("Stop")
    
    print(train_labels.value_counts())
    
    outputs = train_labels.nunique()
    
    epochs = 10
    
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
    test_set["Date"] = data_2021["Date"]
    
    if test_labels.nunique() == 3:
        test_set["Prediction"] = test_set["Prediction"].replace(0,-1)
        test_set["Prediction"] = test_set["Prediction"].replace(1,0)
        test_set["Prediction"] = test_set["Prediction"].replace(2,1)
    
    elif test_labels.nunique() == 4:
        test_set["Prediction"] = test_set["Prediction"].replace(0,-2)
        test_set["Prediction"] = test_set["Prediction"].replace(1,-1)
        test_set["Prediction"] = test_set["Prediction"].replace(2,1)
        test_set["Prediction"] = test_set["Prediction"].replace(3,2)
    
    """
    mean_gain = np.mean(np.abs(test_set["Target_Variation"].loc[test_set["Prediction"] == test_set["Target"]]))
    
    mean_loss = np.mean(np.abs(test_set["Target_Variation"].loc[test_set["Prediction"] != test_set["Target"]]))
    
    print("Mean Gain : {} ||| Mean Loss {}".format(mean_gain, mean_loss))
    """
    
    #Backtest of simplest strategy
    simple_strategy_backtest(test_set)
    
    
    #test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_data,  test_labels, verbose=2)
    
    #time = str(datetime.date.today())
    #model.save('./models/30_to_10m_200_2000_200_y2019_2020_'+time)
    
    """
    #Loading means and stds from non processed dataframe to standardize new data
    cols, prev_means, prev_stds = data_processing.process_minute_data(False)
    
    sept_2022 = pd.read_csv("./minute_data/BTCUSDT-1m-2022-09.csv", names=["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Volume USD", "Trades", "Drop1", "Drop2", "Drop3"])
    
    sept_2022.drop(["Close Time", "Volume", "Trades", "Drop1", "Drop2", "Drop3"], axis=1, inplace=True)
    sept_2022.rename({"Open Time": "Unix"}, axis=1, inplace=True)
    
    sept_2022["Variation"] = (
        sept_2022["Close"] - sept_2022["Close"].shift(1)) / sept_2022["Close"].shift(1)
    
    sept_2022 = data_processing.add_rsi(sept_2022)
    sept_2022 = data_processing.add_macd(sept_2022)
    sept_2022 = data_processing.add_adx(sept_2022)
    
    sept_2022[cols] = (sept_2022[cols] - prev_means)/prev_stds
    
    print(sept_2022)
    
    validation_sept_2022 = standard_labeling(sept_2022, max_days, target_range)
    
    probs = model.predict(validation_sept_2022)
    
    if outputs > 2:
        preds = probs.argmax(axis=-1)
    else:
        preds = [1 if i>0.5 else 0 for i in probs]
    
    #print(test_labels, preds)
    print("Predictions for september of 2022")
    print(tf.math.confusion_matrix(test_labels, preds))
    
    
    """
    
    