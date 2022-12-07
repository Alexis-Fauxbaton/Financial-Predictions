from data_processing import *
from datahandler import *
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import sys

EMA_NORMALIZE_FACTOR = 70000  # TODO Find better option


def standard_labeling(data, max_days, target_range, skip_factor=3, preserve_index=False):

    predict_data = data.copy()
    predict_data["EMA_10"] = ta.ema(predict_data["Close"], 10)
    predict_data["EMA_50"] = ta.ema(predict_data["Close"], 50)
    #predict_data["EMA_200"] = ta.ema(predict_data["Close"], 200)

    for i in range(1, max_days):
        predict_data[["Variation-{}".format(i), "RSI-{}".format(i), "MACD-{}".format(
            i), "MACD_H-{}".format(i), "Regime-{}".format(i)]] = data[["Variation", "RSI", "MACD", "MACD_H", "Regime"]].shift(i)
    predict_data["Target"] = (
        predict_data["Close"].shift(-target_range) - data["Close"] >= 0)

    sign = (predict_data["EMA_50"] - predict_data["EMA_10"]) <= 0

    predict_data["Target"] = ((predict_data["EMA_50"].shift(-target_range) -
                              predict_data["EMA_10"].shift(-target_range)) <= 0) != sign

    predict_data["Target"] = np.where(
        (predict_data["Target"] == False), 0, predict_data["Target"])
    predict_data["Target"] = np.where((predict_data["Target"] == True) & (
        sign == True), -1, predict_data["Target"])
    predict_data["Target"] = np.where((predict_data["Target"] == True) & (
        sign == False), 1, predict_data["Target"])

    predict_data["Target"] = predict_data["Target"] + 1

    predict_data.dropna(inplace=True)
    predict_data.reset_index(inplace=True, drop=preserve_index)
    predict_data = predict_data[0:len(predict_data)-target_range]

    skip = int(max_days/skip_factor)
    skip = 1
    predict_data = predict_data[[
        i % skip == 0 for i in range(len(predict_data))]]

    display_data = predict_data[["Unix", "Close", "EMA_10",
                                 "EMA_50", "Target"]][-1000:].copy().reset_index(drop=True)
    predict_data.drop(
        ["Open", "Close", "High", "Low", "Symbol"], axis=1, inplace=True)

    predict_data["EMA_10"] = predict_data["EMA_10"] / EMA_NORMALIZE_FACTOR
    predict_data["EMA_50"] = predict_data["EMA_50"] / EMA_NORMALIZE_FACTOR

    print("Display Data : \n", display_data.tail(10))

    return predict_data, display_data


######################################################################## BACKTEST FUNCTIONS ########################################################################

def simple_strategy_backtest(test_set, model, algorithm, outputs):
    test_set = test_set[-45000:-44000]
    #test_set = test_set[2064370:2064370+500]
    temp_data = test_set.copy().drop(
        ["Prediction", "Target", "Date", "Close", "EMA_10_Display", "EMA_50_Display"], axis=1)
    temp_target = test_set.copy()["Target"]

    try:
        model.evaluate(temp_data, temp_target)
    except:
        print("{} score : ".format(algorithm), model.score(temp_data, temp_target))

    if algorithm == "MLP":

        probs = model.predict(temp_data)

        if outputs >= 3:
            preds = probs.argmax(axis=-1)
        else:
            preds = [1 if i > 0.5 else 0 for i in probs]
    elif algorithm == "RF":
        preds = model.predict(temp_data)

    print(tf.math.confusion_matrix(temp_target, preds))

    fig, axis = plt.subplots(1, 1, figsize=[10, 5])

    axis.plot(test_set["Close"].index, test_set["Close"], label="Close")
    axis.plot(test_set["EMA_10_Display"].index,
              test_set["EMA_10_Display"], label="EMA_10")
    axis.plot(test_set["EMA_50_Display"].index,
              test_set["EMA_50_Display"], label="EMA_50")
    #TODO NOT PLOTTING THE RIGHT THING
    axis.scatter(test_set[(test_set["Target"] == 0)].index, test_set[(
        test_set["Target"] == 0)]["Close"], alpha=0.5, color='r', marker='o', label="Sell")
    axis.scatter(test_set[(test_set["Target"] == 2)].index, test_set[(
        test_set["Target"] == 2)]["Close"], alpha=0.5, color='b', marker='o', label="Buy")
    """ axis.scatter(test_set[(test_set["Target"] == 0) & (test_set["Prediction"] == test_set["Target"])].index, test_set[(
        test_set["Target"] == 0) & (test_set["Prediction"] == test_set["Target"])]["Close"], alpha=0.5, color='r', marker='o', label="Sell")
    axis.scatter(test_set[(test_set["Target"] == 2) & (test_set["Prediction"] == test_set["Target"])].index, test_set[(
        test_set["Target"] == 2) & (test_set["Prediction"] == test_set["Target"])]["Close"], alpha=0.5, color='b', marker='o', label="Buy")
 """
    plt.legend()
    plt.show()


class EMACrossoverDataHandler(DataHandler):

    def __init__(self, csv_path=None):
        super().__init__(csv_path)

    def create_predict_data(self, max_days=15, target_range=3, standard=True, preserve_index=False):
        print("Creating Predict DataFrame...", end='\t')
        if self.max_days == None:
            self.max_days = max_days
        if self.target_range == None:
            self.target_range = target_range
        self.predict_data = None
        self.add_gaussian_mixture()
        #print("Labeling mode : ", standard)
        if standard:
            read = False
            self.predict_data, self.display_data = standard_labeling(
                self.data, max_days, target_range, preserve_index)

        ################################ USING EMA INSTEAD OF RAW NOISY DATA ######################################
        '''
        cols = list(self.predict_data.columns)
        cols.remove("Unix")
        cols.remove("Date")
        cols.remove("Target")
        cols.remove("Target_Variation")
        for col in cols:
            self.predict_data[col] = ta.ema(self.predict_data[col], 10)
        #self.predict_data[cols] = ta.ema(self.predict_data[cols], 10) SEEMS NOT TO WORK
        self.predict_data.dropna(axis=0, inplace=True)
        '''
        ############################################################################################################
        print("Done")

    def fit_predict(self, train_start="1/1/2017", train_end="1/1/2021", test_start="1/1/2021", test_end="1/1/2023", max_days=15, target_range=3, labeling=True, equal_sampling=False, sampling_method="undersample", epochs=10, algorithm="MLP"):
        if self.predict_data == None:
            self.create_predict_data(max_days, target_range, labeling, True)

        #test_index = self.predict_data["index"]
        
        #self.predict_data.drop("index", axis=1, inplace=True)

        print("Predict Data : \n{}".format(self.predict_data))


        """ fig, axis = plt.subplots(1, 1, figsize=[10,5])
                
        axis.plot(self.display_data.index, self.display_data["Close"], label="Close")
        axis.plot(self.display_data.index, self.display_data["EMA_10"], label="EMA_10")
        axis.plot(self.display_data.index, self.display_data["EMA_50"], label="EMA_50")
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
            # TODO Remove ?
            else:
                train_data, train_labels = train_data.drop(
                    "Target", axis=1), train_data["Target"]

        
        #test_data, test_labels = test_data.drop("Target", axis=1), test_data["Target"]

        outputs = train_labels.nunique()

        shape = (len(test_data.columns),)

        print("Shape of inputs for ML algorithm : {}".format(train_data.shape))

        if algorithm == "MLP":

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

                epochs = 7

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

        try:
            model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2, callbacks=[
                      tensorboard_callback], batch_size=64)
            model.evaluate(test_data, test_labels)
        except:
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
        test_set["EMA_10_Display"] = ta.ema(test_set["Close"], 10)
        test_set["EMA_50_Display"] = ta.ema(test_set["Close"], 50)
        
        #print(pd.merge(test_set["EMA_10_Display"], self.predict_data["EMA_10"] * EMA_NORMALIZE_FACTOR))

        # Backtest of simplest strategy
        simple_strategy_backtest(test_set, model, algorithm, outputs)


max_days = 30
target_range = 10


def main():
    handler = EMACrossoverDataHandler("minute_data/BTC-USD_1M_SIGNALS.csv")

    handler.fit_predict(labeling=True, equal_sampling=True,
                        sampling_method="undersample", algorithm="MLP")


if __name__ == "__main__":

    main()
