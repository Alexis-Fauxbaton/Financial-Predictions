from data_processing import *
from datahandler import *

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

######################################################################## BACKTEST FUNCTIONS ########################################################################

def simple_strategy_backtest(test_set):
    #TODO Adapt strategy to when target range != granularity of data
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
    _, axis = plt.subplots(1, 1, figsize=[20,10])
    axis.plot(N,backtest_set_assets,label="Évolution du portefeuille")
    #net_worth = axis.twinx()
    #net_worth.plot(N, )
    plt.show()

class TrendForecastDataHandler(DataHandler):

    def __init__(self, csv_path=None):
        super().__init__(csv_path)



    def create_predict_data(self, max_days=30, target_range=10, standard=True):
        print("Creating Predict DataFrame")
        if self.max_days == None:
            self.max_days = max_days
        if self.target_range == None:
            self.target_range = target_range
        read = True
        self.predict_data = None
        self.add_gaussian_mixture()
        #print("Labeling mode : ", standard)
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

        ################################ USING EMA INSTEAD OF RAW NOISY DATA ######################################
        cols = list(self.predict_data.columns)
        cols.remove("Unix")
        cols.remove("Date")
        cols.remove("Target")
        cols.remove("Target_Variation")
        for col in cols:
            self.predict_data[col] = ta.ema(self.predict_data[col], 10)
        #self.predict_data[cols] = ta.ema(self.predict_data[cols], 10) SEEMS NOT TO WORK
        self.predict_data.dropna(axis=0, inplace=True)
        ############################################################################################################
        
        
        if not read:
            #skip = int(max_days/3)
            skip = int(max_days/10)
            self.predict_data = self.predict_data[[
                i % skip == 0 for i in range(len(self.predict_data))]]

            if not standard:
                print("Writing Processed Data to memory...")
                self.predict_data.to_csv(
                    "processed_data/processed_1m_{}_{}.csv".format(max_days, target_range), index=False)
                print("Done")
        
    def fit_predict(self, train_start="1/1/2017", train_end="1/1/2021", test_start="1/1/2021", test_end="1/1/2023", max_days=30, target_range=10, labeling=True, equal_sampling=False, sampling_method="classic", epochs=10):
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
                tf.keras.layers.Dense(2*shape[0], activation='relu'),
                tf.keras.layers.Dense(4*shape[0], activation='relu'),
                tf.keras.layers.Dense(2*shape[0], activation='relu'),
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
                tf.keras.layers.Dense(2*shape[0], activation='relu'),
                tf.keras.layers.Dense(4*shape[0], activation='relu'),
                tf.keras.layers.Dense(2*shape[0], activation='relu'),
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
        
        print("Test Set : \n", test_set.head())

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
        
        
        
        
        
        
max_days = 30
target_range = 10

def main():
    handler = TrendForecastDataHandler("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    handler.fit_predict(labeling = True, equal_sampling=True, sampling_method="none")
    pass

if __name__ == "__main__":

    main()