from datahandler import *
from gui import *

pd.set_option("display.max_columns", None)


class NewDataHandler(DataHandler):

    def __init__(self, csv_path=None, skip=None, index_col=None, dataset=None):
        super().__init__(csv_path, skip, index_col)

        if self.data is None and dataset is not None:
            self.data = dataset

    def create_predict_data(self):
        self.predict_data = self.data[self.var_indicators + self.var_attributes]


def main():
    # handler = NewDataHandler("BTCUSDT.csv", index_col=0)
    data = pd.read_csv("BTCUSDT.csv")

    data = get_dollar_bars(data)

    handler = NewDataHandler(dataset=data)

    handler.add_indicators([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV])

    handler.create_var_indicator([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV, Indicators.PERC_RET])

    handler.data.replace([np.inf, -np.inf], 0, inplace=True)

    plot_candlesticks(handler.data)

    handler.standardize_data()

    handler.data.dropna(axis=0, inplace=True)

    handler.create_predict_data()

    predict_data = handler.predict_data

    print(predict_data.columns)

    print(predict_data.head())


if __name__ == "__main__":
    main()
