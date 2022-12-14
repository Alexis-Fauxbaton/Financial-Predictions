from datahandler import DataHandler
from ema_crossover_datahandler import EMACrossoverDataHandler
from trend_forecast_datahandler import TrendForecastDataHandler
from datetime import datetime
import numpy as np
from enum import Enum

order_id = 0

'''
BUY AND SELL ARE FOR BUYBACKS AND SELLBACKS AFTER SHORTS AND LONGS RESPECTIVELY
'''
class OrderType(Enum):
    SHORT = 0
    NOTHING = 1
    LONG = 2
    
    BUY = 3
    SELL = 4

class OrderAction(Enum):
    NONE = None
    EXIT = 1

class OrderActivity(Enum):
    CLOSED = 0
    ACTIVE = 1
    CONDITIONAL = 2
    
class Order:

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

        self.leverage = leverage

        self.tp = tp

        self.sl = sl

        self.id = order_id

        order_id += 1

    def update_order(self, current_high, current_close, current_low, action):
        if self.activity == 'conditional':
            max_price = np.max([current_high, current_low])
            min_price = np.min([current_high, current_low])
            if self.entry_price <= self.order_price and min_price <= self.entry_price:
                self.activity = OrderActivity.ACTIVE

            elif self.entry_price >= self.order_price and max_price > self.entry_price:
                self.activity = OrderActivity.ACTIVE

        if action != OrderAction.EXIT:
            if self.sl != None and self.sl >= min_price:
                self.activity = OrderActivity.CLOSED
                returns = 0
                if self.type == OrderType.LONG:
                    returns = (self.sl / self.entry_price - 1) * \
                        100 * self.leverage
                elif self.type == OrderType.SHORT:
                    returns = (1 - self.sl / self.entry_price) * \
                        100 * self.leverage

                print("Order {} exited at deficit, returns : {}".format(
                    self.id, returns))
                return self.shares * self.sl

            if self.tp != None and self.tp <= max_price:
                self.activity = OrderActivity.CLOSED
                returns = 0
                if self.type == OrderType.LONG:
                    returns = (self.tp / self.entry_price - 1) * \
                        100 * self.leverage
                elif self.type == OrderType.SHORT:
                    returns = (1 - self.tp / self.entry_price) * \
                        100 * self.leverage

                print("Order {} exited at profit, returns : {}".format(
                    self.id, returns))
                return self.shares * self.tp
        else:
            self.activity = OrderActivity.CLOSED
            returns = 0
            if self.type == OrderType.LONG:
                returns = (self.current_close / self.entry_price -
                           1) * 100 * self.leverage
            elif self.type == OrderType.SHORT:
                returns = (1 - self.current_close /
                           self.entry_price) * 100 * self.leverage

            profit = ''
            if returns > 0:
                profit = 'profit'
            else:
                profit = 'deficit'

            print("Order {} exited at {profit}, returns : {}".format(
                self.id, returns))
            return self.shares * self.tp

        return self.shares * current_close

    def get_activity(self):

        return self.activity


class OrderList:

    def __init__(self) -> None:
        self.orders = []

    def add_order(self, order: Order):
        self.orders.append(order)

    def update_orders(self, current_high, current_low, current_close, action=OrderAction.NONE):
        order_balance = 0
        for order in self.orders:
            order_balance += order.update_order(current_high,
                                                current_low, current_close, action)

            if order.get_activity() == OrderActivity.CLOSED:
                self.orders.remove(order)
        return order_balance
    
    def __len__(self):
        return len(self.orders)


class BackTestEnv:

    '''
    Data is raw data
    '''

    def __init__(self, csv_path='None', prediction="ema", max_days=15, target_range=3, start='1/1/2022', end='1/1/2023', initial_balance=100, leverage=7, model_train_start='1/1/2009', model_train_end='1/1/2020', model_test_start='1/1/2020', model_test_end='1/1/2022') -> None:
        self.datahandler = EMACrossoverDataHandler(csv_path)

        # IN BTC-USD_1M_SIGNALS.csv indicators are already standardized
        self.datahandler.fit_predict(train_start=model_train_start, train_end=model_train_end, test_start=model_test_start, test_end=model_test_end, labeling=True, equal_sampling=True,
                                     sampling_method="undersample", algorithm="MLP", critic_test_size=5000, max_days=max_days, target_range=target_range)

        # self.datahandler.add_indicators()

        self.model = self.datahandler.model

        self.critic = self.datahandler.critic

        self.max_days = max_days

        self.target_range = target_range

        # self.datahandler.standardize_indicators()

        self.order_list = OrderList()

        self.start_unix = int(datetime.strptime(start, "%d/%m/%Y").timestamp())

        self.end_unix = int(datetime.strptime(end, "%d/%m/%Y").timestamp())

        self.current_unix = self.start_unix

        self.initial_balance = initial_balance

        self.balance = initial_balance
        
        self.trading_amount = 0

        self.leverage = leverage

        self.predict_data = self.datahandler.predict_data
        
        self.prices = []
        
        self.net_worth = []

    def step(self):
        curr_data = self.datahandler.data[self.current_unix -
                                          self.max_days+1:self.current_unix+1]

        curr_data = self.datahandler.data.loc[self.datahandler.data["Unix"] == self.current_unix]

        curr_predict_data = self.predict_data.loc[self.predict_data["Unix"]
                                                  == self.current_unix]

        #To drop for predictions : Date, Unix, Target
        print(curr_predict_data.columns)

        if len(self.order_list) == 0:

            prob = self.model.predict(curr_predict_data.drop(['Date', 'Unix', 'Target'], axis=1))

            pred = prob.argmax(axis=-1)

            critic = self.critic.predict(curr_predict_data.drop(['Date', 'Unix', 'Target'], axis=1))
            
            if critic > 0.5 and pred != OrderType.NOTHING:
                quantity = self.balance
                order_type = pred
                order_action = OrderAction.NONE
                
                self.balance -= quantity
                if len(self.order_list) != 0:
                    order_action = OrderAction.EXIT
                
                order = Order(type=order_type, activity=OrderActivity.ACTIVE, order_price=curr_data['Close'], entry_price=curr_data['Close'], quantity=quantity, leverage=self.leverage)
                self.order_list.add_order(order)
                
                self.trading_amount = self.order_list.update_orders(curr_data['High'], curr_data['Low'], curr_data['Close'], order_action)
                
        # self.datahandler.data_scaler.inverse_transform(self.curr_data[self.datahandler.columns_to_scale])

        self.current_unix += 60


if __name__ == "__main__":
    backtest_env = BackTestEnv("./minute_data/BTC-USD_1M_SIGNALS.csv")

    backtest_env.step()