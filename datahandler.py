from data_processing import *

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

    def create_predict_data(self):
        pass

    def fit_predict(self):
        pass