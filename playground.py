from datahandler import *
from gui import *
import time
import pandas_ta as ta
from gui import plot_candlesticks

pd.set_option("display.max_columns", None)


class NewDataHandler(DataHandler):

    def __init__(self, csv_path=None, skip=None, index_col=None, dataset=None):
        super().__init__(csv_path, skip, index_col)

        if self.data is None and dataset is not None:
            self.data = dataset

    def create_predict_data(self):
        self.predict_data = self.data[['Unix'] + self.var_indicators + self.var_attributes]


def triple_barrier_labelling(data: pd.DataFrame, upper_barrier=1.02, lower_barrier=0.98, time_limit=30):
    barriers = [None for _ in range(data.shape[0])]
    counters = [None for _ in range(data.shape[0])]
    
    index_list = data.index.to_list()
    list_index = 0
    for idx, row in data.iterrows():
        if idx == index_list[-1]:
            break
        
        curr_close = row['Close']
        
        upper_threshold = curr_close * upper_barrier
        lower_threshold = curr_close * lower_barrier
        
        counter = idx + 1
        while data.loc[counter, "High"] < upper_threshold and data.loc[counter, "Low"] > lower_threshold and (counter - idx) < time_limit and idx + time_limit <= index_list[-1]:
            counter += 1
        
        upper_check = data.loc[counter, "High"] >= upper_threshold
        lower_check = data.loc[counter, "Low"] <= lower_threshold
    
        if upper_check and lower_check: barriers[list_index] = 0 # TODO Undecisive = 0, might want to improve this point for better efficiency
        elif upper_check: barriers[list_index] = 1
        elif lower_check: barriers[list_index] = -1
        elif (counter - idx) == time_limit: barriers[list_index] = 0
        
        counters[list_index] = counter
        
        list_index += 1
        
    data['Label'] = barriers
    data['Index'] = counters
    
    return data

import pandas as pd

def ma_crossover_labelling(data: pd.DataFrame, ma1=15, ma2=50, horizon=15, mode='exponential'):
    names = {'normal': 'MA', 'exponential': 'EMA'}
    name = names[mode] if mode in names.keys() else 'EMA'
    # Calculate the moving averages
    if name == 'MA':
        data[f'{name}{ma1}'] = data['Close'].rolling(window=ma1).mean()
        data[f'{name}{ma2}'] = data['Close'].rolling(window=ma2).mean()
    elif name == 'EMA':
        data[f'{name}{ma1}'] = pd.Series.ewm(data['Close'], span=ma1).mean()
        data[f'{name}{ma2}'] = pd.Series.ewm(data['Close'], span=ma2).mean()
        
    data[f'{name}{ma1} Var'] = ta.percent_return(data[f'{name}{ma1}'], length=1)
    data[f'{name}{ma2} Var'] = ta.percent_return(data[f'{name}{ma2}'], length=1)
    
    data[f'Close_{name}{ma1}_PERC_DIFF'] = ((data[f'{name}{ma1}'] - data["Close"]) / data["Close"])
    data[f'Close_{name}{ma2}_PERC_DIFF'] = ((data[f'{name}{ma2}'] - data["Close"]) / data["Close"])
    data[f'{name}{ma1}_{name}{ma2}_PERC_DIFF'] = ((data[f'{name}{ma2}'] - data[f'{name}{ma1}']) / data[f'{name}{ma1}'])

    # Create a boolean mask for crossover occurrences
    ma1_over_ma2 = data[f'{name}{ma1}'] > data[f'{name}{ma2}']
    ma2_over_ma1_future = data[f'{name}{ma1}'].shift(-horizon) < data[f'{name}{ma2}'].shift(-horizon)
    bearish_crossover_mask = ma1_over_ma2 & ma2_over_ma1_future
    bullish_crossover_mask = ~ma1_over_ma2 & ~ma2_over_ma1_future

    # Initialize labels
    labels = pd.Series(0, index=data.index)

    # Assign labels based on the crossover mask
    labels[bearish_crossover_mask] = -1
    labels[bullish_crossover_mask] = 1

    # Assign labels to the dataframe
    data['Label'] = labels

    return data

def ma_crossover_lagging_labelling(data: pd.DataFrame, ma1=15, ma2=50, horizon=1, mode='exponential'):
    names = {'normal': 'MA', 'exponential': 'EMA'}
    name = names[mode] if mode in [names.keys] else 'EMA'
    
    # Calculate the moving averages
    if name == 'MA':
        data[f'{name}{ma1}'] = data['Close'].rolling(window=ma1).mean()
        data[f'{name}{ma2}'] = data['Close'].rolling(window=ma2).mean()
    elif name == 'EMA':
        data[f'{name}{ma1}'] = pd.Series.ewm(data['Close'], span=ma1).mean()
        data[f'{name}{ma2}'] = pd.Series.ewm(data['Close'], span=ma2).mean()

    # Create a boolean mask for crossover occurrences
    ma1_over_ma2 = data[f'{name}{ma1}'] > data[f'{name}{ma2}']
    ma2_over_ma1_past = data[f'{name}{ma1}'].shift(horizon) < data[f'{name}{ma2}'].shift(horizon)
    bullish_crossover_mask = ma1_over_ma2 & ma2_over_ma1_past
    bearish_crossover_mask = ~ma1_over_ma2 & ~ma2_over_ma1_past

    # Initialize labels
    labels = pd.Series(0, index=data.index)

    # Assign labels based on the crossover mask
    labels[bearish_crossover_mask] = -1
    labels[bullish_crossover_mask] = 1

    # Assign labels to the dataframe
    data[f'Crossover_Lag_{horizon}'] = labels
    
    return data

def main():
    # handler = NewDataHandler("BTCUSDT_15m.csv", index_col=0)
    data = pd.read_csv("BTCUSDT_15m.csv")

    data = get_dollar_bars(data)

    handler = NewDataHandler(dataset=data)

    handler.add_indicators([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV, Indicators.TICK_DENSITY])

    print(handler.head())

    handler.create_var_indicator([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV, Indicators.PERC_RET,
                                  Indicators.TICK_DENSITY])

    handler.data.replace([np.inf, -np.inf], 0, inplace=True)

    plot_candlesticks(handler.data)

    # handler.standardize_data()

    handler.data.dropna(axis=0, inplace=True)

    handler.create_predict_data()

    predict_data = handler.predict_data

    print(predict_data.columns)

    print(predict_data.head())
    
    data = triple_barrier_labelling(data)
    
    print(data.tail(50))
    
    print(data['Label'].value_counts(), data.shape[0])


if __name__ == "__main__":
    main()
