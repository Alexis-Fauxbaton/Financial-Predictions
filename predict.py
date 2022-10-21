import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

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

        predict_data.at[index, "Target"] = first

        if index % 1000 == 0:
            print("Itération {}/{}".format(index, size), end='\r')

    predict_data["Target1"] = (
        data["Close"].shift(-target_range) - data["Close"] >= 0)
    predict_data["Target1"] = np.where(predict_data["Target1"] == True, 1, 0)
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
            predict_data.drop("Target1", axis=1, inplace=True)
                        
            if -1 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(1, 2)
                predict_data["Target"] = predict_data["Target"].replace(0, 1)
                predict_data["Target"] = predict_data["Target"].replace(-1, 0)
        except:
            read = False
            predict_data = meta_labeling(data, max_days, target_range)
            if -1 in predict_data["Target"].values:
                predict_data["Target"] = predict_data["Target"].replace(1, 2)
                predict_data["Target"] = predict_data["Target"].replace(0, 1)
                predict_data["Target"] = predict_data["Target"].replace(-1, 0)

    if not read:
        '''
        predict_data = predict_data[[
            i % int(max_days) == 0 for i in range(len(predict_data))]]
        '''
        predict_data = predict_data[[
            i % int(max_days/3) == 0 for i in range(len(predict_data))]]
            

    if (not standard) and (read==False):
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
    
    train_set = train_data.copy()
    train_set["Target"] = train_labels
    
    #Sampling same number of elements for each class to ensure no one is more likely to be found
    if not standard_labels:
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
        
    else:
        
        print("Using {} classes".format(outputs))
        
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=shape),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(2000, activation='relu'),
            tf.keras.layers.Dense(2000, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(outputs, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model training
    model.fit(train_data, train_labels, epochs=10, validation_data=(test_data,test_labels),callbacks=[tensorboard_callback], batch_size=64)
    
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
    
    if test_labels.nunique() > 2:
        test_set["Prediction"] = test_set["Prediction"].replace(0,-1)
        test_set["Prediction"] = test_set["Prediction"].replace(1,0)
        test_set["Prediction"] = test_set["Prediction"].replace(2,1)
        
    
    mean_gain = np.mean(np.abs(test_set["Target_Variation"].loc[test_set["Prediction"] == test_set["Target"]]))
    
    mean_loss = np.mean(np.abs(test_set["Target_Variation"].loc[test_set["Prediction"] != test_set["Target"]]))
    
    print("Mean Gain : {} ||| Mean Loss {}".format(mean_gain, mean_loss))
    
    #test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_data,  test_labels, verbose=2)
    
    #time = str(datetime.date.today())
    #model.save('./models/30_to_10m_200_2000_200_y2019_2020_'+time)