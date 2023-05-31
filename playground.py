from datahandler import *
from gui import *
import time

pd.set_option("display.max_columns", None)


class NewDataHandler(DataHandler):

    def __init__(self, csv_path=None, skip=None, index_col=None, dataset=None):
        super().__init__(csv_path, skip, index_col)

        if self.data is None and dataset is not None:
            self.data = dataset

    def create_predict_data(self):
        self.predict_data = self.data[['Unix'] + self.var_indicators + self.var_attributes]


def triple_barrier_labelling(data: pd.DataFrame, upper_barrier=1.02, lower_barrier=0.99, time_limit=30):
    barriers = [None for _ in range(data.shape[0])]
    counters = [None for _ in range(data.shape[0])]
    
    index_list = data.index.to_list()
    for idx, row in data.iterrows():
        if idx == index_list[-1]:
            break
        
        curr_close = row['Close']
        
        upper_threshold = curr_close * upper_barrier
        lower_threshold = curr_close * lower_barrier
        
        list_index = 0
        counter = idx + 1
        while data.loc[counter, "High"] < upper_threshold and data.loc[counter, "Low"] > lower_threshold and (counter - idx) < time_limit and idx + time_limit <= index_list[-1]:
            counter += 1
        
        upper_check = data.loc[counter, "High"] >= upper_threshold
        lower_check = data.loc[counter, "Low"] <= lower_threshold
    
        # print(idx, counter-idx, upper_check, lower_check)
        
        if counter == time_limit: barriers[list_index] = 0
        elif upper_check: barriers[list_index] = 1
        elif lower_check: barriers[list_index] = -1
        if upper_check and lower_check: barriers[list_index] = np.nan
        
        counters[list_index] = counter
        
        list_index += 1
        
    data['Barrier'] = barriers
    data['Index'] = counters
    
    return data
        

def main():
    # handler = NewDataHandler("BTCUSDT_15m.csv", index_col=0)
    data = pd.read_csv("BTCUSDT_1m.csv")

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
    
    print(data['Barrier'].value_counts(), data.shape[0])


if __name__ == "__main__":
    main()
