from datahandler import DataHandler
from ema_crossover_datahandler import EMACrossoverDataHandler
from trend_forecast_datahandler import TrendForecastDataHandler
from datetime import datetime
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import os


class OrderType(Enum):
    SHORT = 0
    NOTHING = 1
    LONG = 2


class OrderAction(Enum):
    NONE = None
    EXIT = 1
    CANCEL_LONGS = 0
    CANCEL_SHORTS = 2


class OrderActivity(Enum):
    CLOSED = 0
    ACTIVE = 1
    CONDITIONAL = 2


class Order:

    order_id = 0

    '''
    2 types : Long or Short
    2 activities for now : Conditional and Market
    Order price is the price at which we set the order
    Entry price is the price at which we enter in trade

    Leverage is used only for returns display
    '''

    def __init__(self, type, activity, order_price, entry_price, quantity, tp=None, sl=None, leverage=7) -> None:
        self.type = type

        self.activity = activity

        self.order_price = order_price

        self.entry_price = entry_price

        self.quantity = quantity

        self.shares = quantity / entry_price

        #TODO Account for leverage in position
        self.leverage = leverage

        self.tp = tp

        self.sl = sl

        self.id = Order.order_id

        Order.order_id += 1

    def get_shares(self):
        return self.shares

    def __str__(self) -> str:
        return "Order ID : {} || Order Type : {} || Order Price : {} || Entry Price : {} || Quantity : {} || TP : {} || SL : {} || Leverage : x{}".format(self.id, self.type, self.order_price, self.entry_price, self.quantity, self.tp, self.sl, self.leverage)

    def __repr__(self) -> str:
        return str(self)

    def update_order(self, current_high, current_close, current_low, action):
        returns = 0
        max_price = np.max([current_high, current_low])
        min_price = np.min([current_high, current_low])
        if self.activity == 'conditional':
            if self.entry_price <= self.order_price and min_price <= self.entry_price:
                self.activity = OrderActivity.ACTIVE.value

            elif self.entry_price >= self.order_price and max_price > self.entry_price:
                self.activity = OrderActivity.ACTIVE.value

        elif action == OrderAction.NONE.value:
            exit_price = 0
            if self.sl != None and self.sl >= min_price:
                self.activity = OrderActivity.CLOSED.value
                if self.type == OrderType.LONG.value:
                    returns = (self.sl / self.entry_price - 1) * \
                        100 * self.leverage
                    exit_price = self.sl
                elif self.type == OrderType.SHORT.value:
                    returns = (1 - self.sl / self.entry_price) * \
                        100 * self.leverage
                    exit_price = self.sl

                #print("Stop Loss hit, Order {} ({}) exited at deficit, Entry Price : {} -> {}, returns : {}".format(
                #    self.id, self.type, self.entry_price, exit_price, returns))
                return self.shares * self.sl

            if self.tp != None and self.tp <= max_price:
                self.activity = OrderActivity.CLOSED.value
                returns = 0
                if self.type == OrderType.LONG.value:
                    returns = (self.tp / self.entry_price - 1) * \
                        100 * self.leverage
                    exit_price = current_low
                elif self.type == OrderType.SHORT.value:
                    returns = (1 - self.tp / self.entry_price) * \
                        100 * self.leverage
                    exit_price = self.tp

                #print("Take Profit hit, Order {} ({}) exited at profit, Entry Price : {} -> {}, returns : {}".format(
                #    self.id, self.type, self.entry_price, exit_price, returns))
                return self.shares * self.tp
        else:
            if action == OrderAction.CANCEL_LONGS.value and self.type == OrderType.LONG.value:
                self.activity = OrderActivity.CLOSED.value
                returns = (current_close / self.entry_price -
                            1) * 100 * self.leverage
            elif action == OrderAction.CANCEL_SHORTS.value and self.type == OrderType.SHORT.value:
                self.activity = OrderActivity.CLOSED.value
                returns = (1 - current_close /
                            self.entry_price) * 100 * self.leverage
            else:
                return self.shares * current_close

            profit = ''
            if returns > 0:
                profit = 'profit'
            else:
                profit = 'deficit'

            #print("Signal flagged, Order {} ({}) exited at {}, Entry Price : {} -> {}, returns : {}".format(
            #    self.id, self.type, profit, self.entry_price, current_close, returns))

        return self.shares * current_close

    def get_activity(self):
        return self.activity
    
    def get_type(self):
        return self.type


class OrderList:

    def __init__(self) -> None:
        self.orders = []

    def __str__(self) -> str:
        str_format = ""
        for order in self.orders:
            str_format = str_format + str(order) + '\n'

        return str_format
    
    def __repr__(self) -> str:
        return str(self)

    def add_order(self, order: Order):
        self.orders.append(order)

    def update_orders(self, current_high, current_low, current_close, action=OrderAction.NONE.value):
        order_balance = 0
        transfer_to_balance = 0
        for order in self.orders:
            order_balance += order.update_order(current_high,
                                                current_low, current_close, action)

            if order.get_activity() == OrderActivity.CLOSED.value:
                transfer_to_balance += order.get_shares() * current_close
                self.orders.remove(order)
                
        return order_balance, transfer_to_balance

    def __len__(self):
        return len(self.orders)


class BackTestEnv:

    '''
    Data is raw data
    '''

    def __init__(self, csv_path=None, prediction="ema", max_days=15, target_range=3, start='1/1/2022', end='1/1/2023', initial_balance=100, leverage=7, model_train_start='1/1/2009', model_train_end='1/1/2020', model_test_start='1/1/2020', model_test_end='1/1/2022') -> None:
        self.datahandler = EMACrossoverDataHandler(csv_path)

        # IN BTC-USD_1M_SIGNALS.csv indicators are already standardized
        self.datahandler.fit_predict(train_start=model_train_start, train_end=model_train_end, test_start=model_test_start, test_end=model_test_end, labeling=True, equal_sampling=True,
                                     sampling_method="undersample", algorithm="MLP", critic_test_size=5000, max_days=max_days, target_range=target_range, critic_batch_size=128)

        # self.datahandler.add_indicators()

        self.model = self.datahandler.model

        self.critic = self.datahandler.critic

        self.max_days = max_days

        self.target_range = target_range

        # self.datahandler.standardize_indicators()

        self.order_list = OrderList()

        self.start_unix = int(datetime.strptime(start, "%d/%m/%Y").timestamp())

        self.end_unix = int(np.min([int(datetime.strptime(end, "%d/%m/%Y").timestamp()), self.datahandler.data["Unix"].max()]))

        self.current_unix = self.start_unix

        self.initial_balance = initial_balance

        self.balance = initial_balance

        self.trading_amount = 0

        self.leverage = leverage

        self.predict_data = self.datahandler.predict_data

        self.prices = []

        self.net_worth = []
        
        self.orders_passed = []
        
        self.probs = self.model.predict(self.predict_data.drop(['Date', 'Unix', 'Target', 'index'], axis=1), batch_size=1000)
        
        self.preds = pd.Series(self.probs.argmax(axis=-1), name='Preds')
        print("Prediction type, ", type(self.preds))
        self.preds = pd.concat([self.preds, self.predict_data["Unix"]], axis=1)
        #self.preds['Unix'] = self.predict_data['Unix']
        
        self.critics = self.critic.predict(self.predict_data.drop(['Date', 'Unix', 'Target', 'index'], axis=1), batch_size=1000)
        print(self.critics)
        self.critics = pd.Series(self.critics.flatten(), name='Critics')
        self.critics = pd.concat([self.critics, self.predict_data["Unix"]], axis=1)
        #self.critics['Unix'] = self.predict_data['Unix']

        print(self.preds)
        
        print(self.critics)

    def step(self):
        #curr_data = self.datahandler.data[self.current_unix -
        #                                  self.max_days+1:self.current_unix+1]

        curr_data = self.datahandler.data.loc[self.datahandler.data["Unix"]
                                              == self.current_unix].reset_index(drop=True)
        '''
        curr_predict_data = self.predict_data.loc[self.predict_data["Unix"]
                                                  == self.current_unix]

        curr_pred = self.preds.loc[self.preds["Unix"]==self.current_unix]
        
        curr_critic = self.critics.loc[self.critics["Unix"]==self.current_unix]
        '''
        
        # To drop for predictions : Date, Unix, Target
        #print(curr_predict_data.columns)

        order_action = OrderAction.NONE.value

        added_order = OrderType.NOTHING.value

        #if len(self.order_list) == 0:
        if True:
            #prob = self.model.predict(curr_predict_data.drop(
            #    ['Date', 'Unix', 'Target', 'index'], axis=1))

            #print(self.preds.loc[self.preds["Unix"]==self.current_unix, "Preds"])
            pred = self.preds.loc[self.preds["Unix"]==self.current_unix, "Preds"].item()

            #pred = prob.argmax(axis=-1)

            #critic = self.critic.predict(curr_predict_data.drop(
            #    ['Date', 'Unix', 'Target', 'index'], axis=1))

            critic = self.critics.loc[self.critics["Unix"]==self.current_unix, "Critics"].item()

            #print(pred, critic)

            if critic > 0.5 and pred != OrderType.NOTHING.value:
                quantity = 0.25 * self.balance
                order_type = pred
                order_action = order_type
                sl = None

                if self.balance >= quantity and quantity > 1e-6:
                    self.balance -= quantity
                    # TODO IMPLEMENT A WAY TO MANIPULATE ALREADY OPEN TRADES
                    if order_type == OrderType.LONG.value:
                        #print("LONG")
                        sl = 0.99*(curr_data.loc[0,'Close'].item())
                        tp = 1.03*(curr_data.loc[0,'Close'].item())
                    elif order_type == OrderType.SHORT.value:
                        #print("SHORT")
                        sl = 1.01*(curr_data.loc[0,'Close'].item())
                        tp = 0.97*(curr_data.loc[0,'Close'].item())
                    '''
                    print("Order Type : ", order_type)
                    print("LONG Order : {} || SHORT Order : {}".format(OrderType.LONG.value, OrderType.SHORT.value))
                    print("{} || {}".format((order_type == OrderType.LONG.value), (order_type == OrderType.SHORT.value)))
                    print("Current Close : ", curr_data.loc[0,'Close'].item(), type(curr_data.loc[0,'Close'].item()))
                    print("Stop Loss : ", sl)
                    '''
                    order = Order(type=order_type, activity=OrderActivity.ACTIVE.value,
                                  order_price=curr_data.loc[0,'Close'].item(), entry_price=curr_data.loc[0,'Close'].item(), quantity=quantity, leverage=self.leverage, sl=sl, tp=tp)
                    self.order_list.add_order(order)
                    added_order = order_type
                else:
                    added_order = order_type


        self.trading_amount, transfer_to_balance = self.order_list.update_orders(
            curr_data['High'].item(), curr_data['Low'].item(), curr_data['Close'].item(), order_action)

        self.prices.append(curr_data['Close'])

        self.net_worth.append(self.balance + self.trading_amount)
        
        self.orders_passed.append(added_order)

        self.balance += transfer_to_balance

        # self.datahandler.data_scaler.inverse_transform(self.curr_data[self.datahandler.columns_to_scale])

        self.current_unix += 60


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    backtest_env = BackTestEnv("./minute_data/BTC-USD_1M_SIGNALS.csv")

    max_backtest_unix = backtest_env.end_unix

    count = 0

    n_steps = (backtest_env.end_unix - backtest_env.start_unix) / 60

    while backtest_env.current_unix < backtest_env.end_unix:
        #if not (backtest_env.datahandler.data["Unix"].isin([backtest_env.current_unix]).item()):
        if backtest_env.current_unix > backtest_env.datahandler.predict_data["Unix"].max().item():
        #if not (backtest_env.current_unix in backtest_env.datahandler.data["Unix"].values):
            break
        backtest_env.step()

        if count % 1000 == 0:
            '''
            print(backtest_env.order_list)
            print("Net Worth :", backtest_env.net_worth[-1])
            print('\n')
            '''
            #TODO DO PREDICTIONS ALL AT ONCE TO SAVE COMPUTE TIME
            print("Step [{}/{}] ({} %)".format(count+1, n_steps, (count+1)/n_steps * 100), end='\n')
            
        count += 1
        
    N = pd.Series(range(0,len(backtest_env.prices)))
    
    prices = pd.Series(backtest_env.prices)
    
    orders_passed = pd.Series(backtest_env.orders_passed)
    
    net_worth = pd.Series(backtest_env.net_worth)
    
    fig, axis = plt.subplots()
    
    axis.plot(N, prices, label='Price')
    
    #TODO FIX N size and prices[...] size that must be the same
    axis.scatter(N[orders_passed == OrderType.LONG.value], prices[orders_passed == OrderType.LONG.value], 'o', color='b', label='Long')
    
    axis.scatter(N[orders_passed == OrderType.SHORT.value], prices[orders_passed == OrderType.SHORT.value], 'o', color='r', label='Short')
    
    net_worth_axis = axis.twinx()
    
    net_worth_axis.plot(N, net_worth, label='Net Worth in USD')
    
    plt.legend()
    
    plt.plot()
    
    
    