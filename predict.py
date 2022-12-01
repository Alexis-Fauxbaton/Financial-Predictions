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

def main():
    handler = data_processing.DataHandler("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    handler.fit_predict(labeling = True, equal_sampling=True, sampling_method="none")
    pass

if __name__ == "__main__":

    main()
    
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
    
    