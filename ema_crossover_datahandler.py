from data_processing import *
from datahandler import *
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import sys
import os

EMA_NORMALIZE_FACTOR = 70000  # TODO Find better option
FIRST_EMA = 10#10
SECOND_EMA = 50#50

def standard_labeling(data, max_days, target_range, skip_factor=3, preserve_index=False):

    predict_data = data.copy()
    predict_data["EMA_{}".format(FIRST_EMA)] = ta.ema(predict_data["Close"], FIRST_EMA)
    predict_data["EMA_{}".format(SECOND_EMA)] = ta.ema(predict_data["Close"], SECOND_EMA)
    #predict_data["EMA_200"] = ta.ema(predict_data["Close"], 200)

    for i in range(1, max_days):
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i), "Regime-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H", "Regime"]].shift(i)
    predict_data["Target"] = (
        predict_data["Close"].shift(-target_range) - data["Close"] >= 0)

    sign = (predict_data["EMA_{}".format(SECOND_EMA)] - predict_data["EMA_{}".format(FIRST_EMA)]) <= 0

    predict_data["Target"] = ((predict_data["EMA_{}".format(SECOND_EMA)].shift(-target_range) -
                              predict_data["EMA_{}".format(FIRST_EMA)].shift(-target_range)) <= 0) != sign

    predict_data["Target"] = np.where(
        (predict_data["Target"] == False), 0, predict_data["Target"])
    predict_data["Target"] = np.where((predict_data["Target"] == True) & (
        sign == True), -1, predict_data["Target"])
    predict_data["Target"] = np.where((predict_data["Target"] == True) & (
        sign == False), 1, predict_data["Target"])

    predict_data["Target"] = predict_data["Target"] + 1

    predict_data.dropna(inplace=True)
    predict_data.reset_index(inplace=True, drop=(not preserve_index))
    predict_data = predict_data[0:len(predict_data)-target_range]

    skip = int(max_days/skip_factor)
    skip = 1
    predict_data = predict_data[[
        i % skip == 0 for i in range(len(predict_data))]]

    display_data = predict_data[["Unix", "Close", "EMA_{}".format(FIRST_EMA),
                                 "EMA_{}".format(SECOND_EMA), "Target"]][-1000:].copy().reset_index(drop=True)
    
    predict_close = predict_data["Close"]
    
    predict_data.drop(
        ["Open", "Close", "High", "Low", "Symbol"], axis=1, inplace=True)

    predict_data["EMA_{}".format(FIRST_EMA)] = predict_data["EMA_{}".format(FIRST_EMA)] / EMA_NORMALIZE_FACTOR
    predict_data["EMA_{}".format(SECOND_EMA)] = predict_data["EMA_{}".format(SECOND_EMA)] / EMA_NORMALIZE_FACTOR

    print("Display Data : \n", display_data.tail(10))

    return predict_data, display_data, predict_close


######################################################################## BACKTEST FUNCTIONS ########################################################################

def simple_strategy_backtest(data, model, critic, algorithm, outputs, max_days, target_range, critic_test_size):
    test_set, _, test_close = standard_labeling(
                data[-critic_test_size:], max_days, target_range, 3, False)
    #test_set = test_set[2064370:2064370+500]
    temp_data = test_set.copy().drop(
        ["Date", "Target", "Unix"], axis=1).reset_index(drop=True)
    temp_target = test_set.copy()["Target"].reset_index(drop=True)
    
    test_close.reset_index(inplace=True, drop=True)

    print(temp_data)

    print(temp_target)
    
    test_rsi = ta.rsi(test_close, 14).reset_index(drop=True)[-critic_test_size:]

    print(test_rsi)

    try:
        model.evaluate(temp_data, temp_target)
    except:
        print("{} score : ".format(algorithm), model.score(temp_data, temp_target))

    if algorithm == "MLP":

        probs = model.predict(temp_data)
        
        critic_probs = critic.predict(temp_data)

        if outputs >= 3:
            preds = probs.argmax(axis=-1)
        else:
            preds = [1 if i > 0.5 else 0 for i in probs]
        
        critic_preds = [1 if i > 0.5 else 0 for i in critic_probs]
        
        critic_target = np.where((pd.Series(preds) == temp_target.reset_index(drop=True)), 1, 0)
            
    elif algorithm == "RF":
        preds = model.predict(temp_data)

    print(tf.math.confusion_matrix(temp_target, preds))
    
    print(tf.math.confusion_matrix(critic_target, critic_preds))
    
    
    initial_balance = 100
    usd_held = initial_balance
    shares_held = 0
    net_worth = []
    baseline_net_worth = []
    baseline_shares_held = 0
    usd_balance = []
    
    idx = 0
    for price in test_close.values:
        #print("USD : {} || Shares : {} || Price : {}".format(usd_held, shares_held, price))
        
        if preds[idx] == 2 and critic_probs[idx] >= 0.55:# and test_rsi[idx] <= 65:
            #print(2)
            flowing_amount = usd_held * 0.25
            usd_held = usd_held - flowing_amount
            shares_held = shares_held + flowing_amount / price
            
        elif preds[idx] == 0 and critic_probs[idx] >= 0.55:# and test_rsi[idx] >= 35:
            #print(0)
            flowing_amount = shares_held * 0.25
            shares_held = shares_held - flowing_amount
            usd_held = usd_held + flowing_amount * price
                        
        if idx == 0:
            baseline_shares_held = initial_balance / price
        
        usd_balance.append(usd_held)
        
        baseline_net_worth.append(baseline_shares_held * price)
                
        net_worth.append(usd_held + shares_held * price)
        
        idx += 1
    
    print("Begin Price : {} || End Price : {}".format(test_close.values[0], test_close.values[-1]))
    print("Baseline Factor : {}".format((test_close.values[0] - test_close.values[-1]) / test_close.values[0]))
    print("Strategy net worth : {}\nBaseline Strategy net worth : {}".format(net_worth[-1], baseline_net_worth[-1]))
    
    print(baseline_net_worth[:15])
    
    fig, axis = plt.subplots(2, 1, figsize=[10, 5])

    test_set = test_set.join(test_close)
    test_set["EMA_{}_Display".format(FIRST_EMA)] = ta.ema(test_set["Close"], FIRST_EMA)
    test_set["EMA_{}_Display".format(SECOND_EMA)] = ta.ema(test_set["Close"], SECOND_EMA)
    test_set["Prediction"] = preds
    
    test_set.reset_index(inplace=True, drop=True)
    #critic_probs = pd.Series(critic_probs)
    
    axis[0].plot(test_set["Close"].index, test_set["Close"], label="Close")
    axis[0].plot(test_set["EMA_{}_Display".format(FIRST_EMA)].index,
              test_set["EMA_{}_Display".format(FIRST_EMA)], label="EMA_{}".format(FIRST_EMA))
    axis[0].plot(test_set["EMA_{}_Display".format(SECOND_EMA)].index,
              test_set["EMA_{}_Display".format(SECOND_EMA)], label="EMA_{}".format(SECOND_EMA))

    axis[0].scatter(test_set[(test_set["Prediction"] == 0) & (test_set["Prediction"] == test_set["Target"])].index, test_set[(
        test_set["Prediction"] == 0) & (test_set["Prediction"] == test_set["Target"])]["Close"], alpha=critic_probs[(test_set["Prediction"] == 0) & (test_set["Prediction"] == test_set["Target"])], color='r', marker='o', label="Sell")
    axis[0].scatter(test_set[(test_set["Prediction"] == 2) & (test_set["Prediction"] == test_set["Target"])].index, test_set[(
        test_set["Prediction"] == 2) & (test_set["Prediction"] == test_set["Target"])]["Close"], alpha=critic_probs[(test_set["Prediction"] == 2) & (test_set["Prediction"] == test_set["Target"])], color='b', marker='o', label="Buy")
    axis[0].scatter(test_set[(test_set["Prediction"] == 0) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] > test_set["EMA_{}_Display".format(SECOND_EMA)])].index, test_set[(
        test_set["Prediction"] == 0) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] > test_set["EMA_{}_Display".format(SECOND_EMA)])]["Close"], alpha=critic_probs[(test_set["Prediction"] == 0) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] > test_set["EMA_{}_Display".format(SECOND_EMA)])], color='k', marker='o', label="Fake Sell")
    axis[0].scatter(test_set[(test_set["Prediction"] == 2) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] < test_set["EMA_{}_Display".format(SECOND_EMA)])].index, test_set[(
        test_set["Prediction"] == 2) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] < test_set["EMA_{}_Display".format(SECOND_EMA)])]["Close"], alpha=critic_probs[(test_set["Prediction"] == 2) & (test_set["Prediction"] != test_set["Target"]) & (test_set["EMA_{}_Display".format(FIRST_EMA)] < test_set["EMA_{}_Display".format(SECOND_EMA)])], color='m', marker='o', label="Fake Buy")


    net_worth_axis = axis[0].twinx()
    #reward_list = self.reward_list
    net_worth_axis.plot(range(len(net_worth)), net_worth, label="Net Worth", alpha = 0.25)
    net_worth_axis.plot(range(len(net_worth)), baseline_net_worth, label="Baseline Net Worth", alpha = 0.25)

    axis[1].plot(range(len(usd_balance)), usd_balance, label="USD Balance")

    plt.legend()
    plt.show()


class EMACrossoverDataHandler(DataHandler):

    def __init__(self, csv_path=None, skip=None):
        super().__init__(csv_path, skip)

    def create_predict_data(self, max_days=15, target_range=3, standard=True, preserve_index=False):
        print("Creating Predict DataFrame...", end='\t')
        if self.max_days == None:
            self.max_days = max_days
        if self.target_range == None:
            self.target_range = target_range
        self.predict_data = None
        self.add_gaussian_mixture()
        if standard:
            read = False
            self.predict_data, self.display_data, _ = standard_labeling(
                self.data, max_days, target_range, 3, preserve_index)

        ################################ USING EMA INSTEAD OF RAW NOISY DATA ######################################
        '''
        cols = list(self.predict_data.columns)
        cols.remove("Unix")
        cols.remove("Date")
        cols.remove("Target")
        cols.remove("Target_Variation")
        for col in cols:
            self.predict_data[col] = ta.ema(self.predict_data[col], FIRST_EMA)
        #self.predict_data[cols] = ta.ema(self.predict_data[cols], FIRST_EMA) SEEMS NOT TO WORK
        self.predict_data.dropna(axis=0, inplace=True)
        '''
        ############################################################################################################
        print("Done")

    def fit_predict(self, train_start="1/1/2017", train_end="1/1/2021", test_start="1/1/2021", test_end="1/1/2023", max_days=15, target_range=3, labeling=True, equal_sampling=False, sampling_method="undersample", epochs=10, algorithm="MLP", critic_test_size=4000):
        if self.predict_data == None:
            self.create_predict_data(max_days, target_range, labeling, True)

        #test_index = self.predict_data["index"]
        
        #self.predict_data.drop("index", axis=1, inplace=True)

        print("Predict Data : \n{}".format(self.predict_data))


        """ fig, axis = plt.subplots(1, 1, figsize=[10,5])
                
        axis.plot(self.display_data.index, self.display_data["Close"], label="Close")
        axis.plot(self.display_data.index, self.display_data["EMA_{}".format(FIRST_EMA)], label="EMA_{}".format(FIRST_EMA))
        axis.plot(self.display_data.index, self.display_data["EMA_{}".format(SECOND_EMA)], label="EMA_{}".format(SECOND_EMA))
        axis.scatter(self.display_data[self.display_data["Target"] == 0].index, self.display_data[self.display_data["Target"] == 0]["Close"], alpha = 0.5, color = 'r', marker = 'o', label="Sell")
        axis.scatter(self.display_data[self.display_data["Target"] == 2].index, self.display_data[self.display_data["Target"] == 2]["Close"], alpha = 0.5, color = 'b', marker = 'o', label="Buy")
        
        
        plt.legend()
        plt.show() """

        train_data = get_data_between(
            self.predict_data, train_start, train_end)

        test_data = get_data_between(self.predict_data, test_start, test_end)

        test_date = test_data["Date"]

        train_data, train_labels, test_data, test_labels, test_variation = yearly_custom_splitter(
            train_data, test_data)

        train_data.drop("index", axis=1, inplace=True)

        test_index = test_data["index"]
        print("Test Index\n", test_index)

        test_data.drop("index", axis=1, inplace=True)
        
        
        if sampling_method != 'none':
            if equal_sampling:
                train_data["Target"] = train_labels
                train_data, train_labels = sample_equal_target(
                    train_data, method=sampling_method)

        
        #test_data, test_labels = test_data.drop("Target", axis=1), test_data["Target"]

        outputs = train_labels.nunique()

        shape = (len(test_data.columns),)

        print("Shape of inputs for ML algorithm : {}".format(train_data.shape))

        if algorithm == "MLP":

            if outputs == 3:

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


                critic = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=shape),
                    tf.keras.layers.Dense(2*shape[0], activation='relu'),
                    tf.keras.layers.Dense(4*shape[0], activation='relu'),
                    tf.keras.layers.Dense(2*shape[0], activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])

                critic.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=['accuracy', 'Precision', 'Recall', 'AUC'])


                epochs = 5

            log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
        elif algorithm == "RF":
            model = RandomForestClassifier(max_depth=5)
        else:
            Exception("Entered model is not recognized, Exiting...")
            sys.exit()
        # model training
    #    model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data,test_labels),callbacks=[tensorboard_callback], batch_size=64)

        print("Using {} model".format(algorithm))

        if algorithm == "MLP":
            model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2, callbacks=[
                      tensorboard_callback], batch_size=64)
            model.evaluate(test_data, test_labels)
            
            probs = model.predict(test_data)

            preds = probs.argmax(axis=-1)
                        
            critic_labels = np.where((pd.Series(preds) == test_labels.reset_index(drop=True)), 1, 0)
            
            print("Critic Learning...")
            
            critic.fit(test_data[:-critic_test_size], critic_labels[:-critic_test_size], epochs=10, validation_split=0.2, batch_size=64)
            
            print("Done")
            
            critic.evaluate(test_data[-critic_test_size:], critic_labels[-critic_test_size:])
            
            critic_probs = critic.predict(test_data[-critic_test_size:])
            
            critic_preds = [1 if i > 0.5 else 0 for i in critic_probs]
            
            print("Critic Confusion Matrix")
            print(tf.math.confusion_matrix(critic_labels[-critic_test_size:], critic_preds))
            
            
        else:
            model.fit(train_data, train_labels)
            print("{} score : ".format(algorithm),
                  model.score(test_data, test_labels))

        if algorithm == "MLP":

            probs = model.predict(test_data)

            if outputs >= 3:
                preds = probs.argmax(axis=-1)
            else:
                preds = [1 if i > 0.5 else 0 for i in probs]
        elif algorithm == "RF":
            preds = model.predict(test_data)

        #print(test_labels, preds)
        print(tf.math.confusion_matrix(test_labels, preds))

        test_set = test_data.copy()

        print("Predict Data After : \n", self.predict_data)

        print("Test Set : \n", test_set)

        test_set["Prediction"] = preds
        test_set["Target"] = test_labels
        #test_set["Target_Variation"] = test_variation
        # Our test set is data2021 (Set to potentially change)
        test_set["Date"] = test_date
        test_set["Close"] = self.data["Close"].loc[test_index]
        test_set["EMA_{}_Display".format(FIRST_EMA)] = ta.ema(test_set["Close"], FIRST_EMA)
        test_set["EMA_{}_Display".format(SECOND_EMA)] = ta.ema(test_set["Close"], SECOND_EMA)
                
        #print(pd.merge(test_set["EMA_{}_Display".format(FIRST_EMA)], self.predict_data["EMA_{}".format(FIRST_EMA)] * EMA_NORMALIZE_FACTOR))

        # Backtest of simplest strategy
        simple_strategy_backtest(self.data, model, critic, algorithm, outputs, self.max_days, self.target_range, critic_test_size)


max_days = 30
target_range = 10


def main():
    
    timeframe_skips = {'1m':1, '15m':15, '1h':60, '4h':240}
    
    handler = EMACrossoverDataHandler("minute_data/BTC-USD_1M_SIGNALS.csv", timeframe_skips["1m"])

    handler.fit_predict(labeling=True, equal_sampling=True,
                        sampling_method="undersample", algorithm="MLP", critic_test_size=10000, max_days = 15, target_range=5)


if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    main()
