import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

INITIAL_BALANCE = 100

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, lookback=30, max_steps=100):
        super(TradingEnv).__init__()
        
        self.data = data
        self.lookback = lookback
        self.max_steps = max_steps
        #Define the number of possible actions here
        
        #Actions : BUY || SELL || NOTHING
        #How Much : 10 || 50 || 100 % of balance
        #7 Combinations (Only one case with NOTHING)
        self.action_space = spaces.Discrete(7)
        
        #We assume that the only columns that are not fed to the agent are unix date + last 4 OHLC (1 + 4)
        #Each row of data contains the OHLC, Unix, perct_change since last candle and indicators needed for the agent FOR 1 TIMESTAMP
        self.observation_space = spaces.Tuple(spaces.Box(low=[-np.inf for i in range(self.data.shape[1]-(1 + 4))], high=[np.inf for i in range(self.data.shape[1]-(1 + 4))], shape=(self.lookback, self.data.shape[1] - (1 + 4))), spaces.Box(low=[-np.inf, -np.inf], high=[np.inf, np.inf], shape=(1,2)))
        
    def reset(self):
        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.net_worth = 0
        self.current_step = np.random.randint(0,self.data.shape[0])
        self.balance_list = [self.balance]
        self.shares_held_list = [self.shares_held]
        self.net_worth_list = [self.net_worth]
        self.price_list = []
        
        
        
        return self._next_observation()
    
    def _next_observation(self):
        cpy_data = self.data.copy().drop(["Open", "Low", "Close", "High", "Unix", "Symbol"], axis=1)
        frame = np.array(self.data.loc[self.current_step:self.current_step-self.lookback,cpy_data.columns])
        
        obs = np.append(frame, [self.balance, self.shares_held], axis=0)
        
        
        return obs
    
    def step(self, action):
        
        self._take_action(action)
        
        self.current_step += 1
        
        if self.current_step >= self.data.shape[0]:
            done = True
        else:
            done = self.net_worth <= 0
            
        discount = self.current_step / self.max_steps
        
        reward = discount * self.net_worth
        
        obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def take_action(self, action):
        
        current_price = np.random.uniform(self.data.loc[self.current_step,"Low"], self.data.loc[self.current_step, "High"])
        
        if self.current_step == 0:
            self.price_list.append(current_price)
        
        #Convert 10% of balance into asset
        if action == 0:
            shares_bought = (0.1 * self.balance) / current_price
            self.shares_held += shares_bought
            self.balance = self.balance * 0.9
        
        #Convert 50% of balance into asset
        elif action == 1:
            shares_bought = (0.5 * self.balance) / current_price
            self.shares_held += shares_bought
            self.balance = self.balance * 0.5
            
        #Convert 100% of balance into asset            
        elif action == 2:
            shares_bought = (1 * self.balance) / current_price
            self.shares_held += shares_bought
            self.balance = 0
        
        #Convert 10% of shares into balance    
        elif action == 3:
            shares_sold = 0.1 * self.shares_held
            self.shares_held -= shares_sold
            self.balance += shares_sold * current_price
            
        #Convert 50% of shares into balance    
        elif action == 3:
            shares_sold = 0.5 * self.shares_held
            self.shares_held -= shares_sold
            self.balance += shares_sold * current_price
            
        #Convert 100% of shares into balance    
        elif action == 3:
            shares_sold = self.shares_held
            self.shares_held -= shares_sold
            self.balance += shares_sold * current_price
            
        self.net_worth = self.balance + self.shares_held * current_price
        
        self.balance_list.append(self.balance)
        self.shares_held_list.append(self.shares_held)
        self.net_worth_list.append(self.net_worth)
        self.price_list.append(current_price)
        
    def render(self, mode='human', close=False):
        print("Step ", self.current_step)
        print("Balance :", self.balance)
        print("Shares :", self.shares_held)
        print("Net Worth :", self.net_worth)
        steps = np.arange(len(self.balance_list))
        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(2, 2)
        
        axis[0,0].plot(steps, self.net_worth_list, label="Net Worth")
        axis[0,0].set_title("Evolution of Net Worth in USD")
        
        axis[1,0].plot(steps, self.balance_list, label="Balance")
        axis[1,0].set_title("Evolution of Balance in USD")
        
        axis[0,1].plot(steps, self.shares_held_list, label="Shares Held")
        axis[0,1].set_title("Evolution of Held Shares")
        
        plt.show()