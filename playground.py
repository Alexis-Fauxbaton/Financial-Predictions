from datahandler import *

pd.set_option("display.max_columns", None)


class NewDataHandler(DataHandler):

    def __init__(self, csv_path=None, skip=None, index_col=None):
        super().__init__(csv_path, skip, index_col)

    def create_predict_data(self):
        self.predict_data = self.data[self.var_indicators + self.var_attributes]


def main():
    handler = NewDataHandler("BTCUSDT.csv", index_col=0)

    handler.add_indicators([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV])

    handler.create_var_indicator([Indicators.RSI, Indicators.MACD, Indicators.ADX, Indicators.OBV, Indicators.PERC_RET])

    handler.data.replace([np.inf, -np.inf], 0, inplace=True)

    handler.standardize_data()

    handler.data.dropna(axis=0, inplace=True)

    handler.create_predict_data()

    data = handler.predict_data


if __name__ == "__main__":
    main()
