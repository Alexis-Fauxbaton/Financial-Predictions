from data_processing import *
import tensorflow as tf


class DataHandler:
    def __init__(self, csv_path=None, skip=None, index_col=None):
        self.data_scaler = None
        self.max_days = None
        self.target_range = None
        self.columns_to_scale = None
        self.predict_data = None
        self.model = None
        self.critic = None
        self.indicators = []
        self.var_indicators = []
        self.var_attributes = []
        if csv_path is None:
            self.data = None
        else:
            try:
                self.data = pd.read_csv(csv_path, index_col=index_col)
                self.data.sort_values(
                    by="Unix", ascending=True, inplace=True, ignore_index=True)
                # TODO ADD CODE TO DETERMINE TIMEFRAME FROM DIFFERENCE IN TIME BETWEEN TWO FIRST ELEMENTS OF DATA
                if skip is not None:
                    self.data = self.data[[i % skip == 0 for i in range(self.data.shape[0])]]
                self.timeframe = ''
            except:
                del self.data
                self.data = None
                print("Failed to import data from path {}".format(csv_path))

    def head(self, display=5):
        if isinstance(self.data, pd.DataFrame):
            print(self.data.head(display))

    def tail(self, display=5):
        if isinstance(self.data, pd.DataFrame):
            print(self.data.tail(display))

    def load(self, csv_path, skip, index_col):
        del self.data
        del self.predict_data
        del self.data_scaler

        print("Successfully deleted old data")

        try:
            self.data = pd.read_csv(csv_path, skip, index_col)
        except:
            print("Failed to import data located at path {}".format(csv_path))

    def get_data(self):
        return self.data

    def get_predict_data(self):
        return self.predict_data

    def plot(self, x, y):
        self.data.plot(x, y, figsize=[20, 10])
        return

    def add_indicator(self, indicator: Indicators):
        if indicator == Indicators.ADX:
            self.data = add_adx(self.data)
        elif indicator == Indicators.RSI:
            self.data = add_rsi(self.data)
        elif indicator == Indicators.MACD:
            self.data = add_macd(self.data)
        elif indicator == Indicators.LOG_RET:
            self.data = add_log_return(self.data)
        elif indicator == Indicators.PERC_RET:
            self.data = add_percent_return(self.data)
        elif indicator == Indicators.OBV:
            self.data = add_obv(self.data)
        elif indicator == Indicators.TICK_DENSITY:
            self.data = add_tick_density(self.data)

    def add_indicators(self, indicators: list(Indicators)):
        for indicator in indicators:
            try:
                self.add_indicator(indicator)
                if isinstance(indicator.value, list):
                    for ind in indicator.value:
                        self.indicators.append(ind)
                else:
                    self.indicators.append(indicator.value)
            except Exception as e:
                if isinstance(indicator, list):
                    print(f"Could not add indicator {' '.join(indicator.value)}")
                else:
                    print(f"Could not add indicator {indicator.value}")
                print("Error message", e)

    def standardize_data(self, method="standard"):
        self.columns_to_scale = self.indicators + self.var_attributes
        if method == "standard":
            if self.data_scaler is None:
                self.data_scaler = StandardScaler()
                self.data[self.columns_to_scale] = self.data_scaler.fit_transform(
                    self.data[self.columns_to_scale])
            else:
                self.data[self.columns_to_scale] = self.data_scaler.transform(self.data[self.columns_to_scale])

    def create_var_indicator(self, indicator_list: list(Indicators), interval=1, remove=False):
        for indicator in indicator_list:

            if (indicator == Indicators.PERC_RET) or (indicator == Indicators.LOG_RET):
                self.add_indicator(indicator)
                self.var_attributes.append(indicator.value)

            elif indicator == Indicators.TICK_DENSITY and indicator.value in self.indicators:
                print(indicator)
                self.var_attributes.append(indicator.value)

            elif (isinstance(indicator.value, list) is False) and (indicator.value not in self.indicators):
                print(f"Ignoring indicator {indicator.value}. Reason: Not found in the list of indicators")
                continue

            elif isinstance(indicator.value, list):

                cont = False

                for ind in indicator.value:

                    if ind not in self.indicators:
                        print(f"Ignoring indicators {', '.join(indicator.value)}. Reason: Not found in the list of "

                              f"indicators")

                        cont = True

                        break

                    ind_var = ta.percent_return(self.data[ind], length=interval)

                    ind_var = ind_var.rename(ind + " Var")

                    self.var_indicators.append(ind + " Var")

                    if remove:
                        self.data.drop(ind, axis=1, inplace=True)

                    self.data = self.data.join(ind_var)

                if cont:
                    continue

            else:
                ind_var = ta.percent_return(self.data[indicator.value], length=interval)

                ind_var = ind_var.rename(indicator.value + " Var")

                self.var_indicators.append(indicator.value + " Var")

                if remove:
                    self.data.drop(indicator.value, axis=1, inplace=True)

                self.data = self.data.join(ind_var)

    def create_predict_data(self):
        pass

    def fit_predict(self):
        pass
