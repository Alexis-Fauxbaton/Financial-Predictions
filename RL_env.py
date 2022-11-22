import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

INITIAL_BALANCE = 100
MAX_BALANCE = 1000
OUTPUT_SIZE = 7

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, lookback=30, max_steps=300):
        super(TradingEnv).__init__()
        
        self.data = data
        cpy_data = self.data.copy().drop(["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"], axis=1)
        self.features = cpy_data.columns
        self.lookback = lookback
        self.max_steps = max_steps
        #Define the number of possible actions here
        
        #Actions : BUY || SELL || NOTHING
        #How Much : 10 || 50 || 100 % of balance
        #7 Combinations (Only one case with NOTHING)
        self.action_space = spaces.Discrete(OUTPUT_SIZE)
        
        #We assume that the only columns that are not fed to the agent are unix date + last 4 OHLC (1 + 4)
        #Each row of data contains the OHLC, Unix, perct_change since last candle and indicators needed for the agent FOR 1 TIMESTAMP
        #self.observation_space = spaces.Tuple(spaces.Box(low=[-np.inf for i in range(self.data.shape[1]-(1 + 4))], high=[np.inf for i in range(self.data.shape[1]-(1 + 4))], shape=(self.lookback, self.data.shape[1] - (1 + 4))), spaces.Box(low=[-np.inf, -np.inf], high=[np.inf, np.inf], shape=(1,2)))
        self.observation_space = spaces.Tuple([spaces.Box(low=np.array([-np.inf for i in range(self.data.shape[1]-(1 + 4))]), high=np.array([np.inf for i in range(self.data.shape[1]-(1 + 4))]), dtype=np.float64), 
                                               spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.float64)])
        
    def reset(self):
        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.net_worth = 0
        self.current_step = np.random.randint(0 + self.lookback,self.data.shape[0] - self.max_steps)
        self.current_step_idx = 0
        self.balance_list = [self.balance]
        self.shares_held_list = [self.shares_held]
        self.net_worth_list = [self.net_worth]
        self.price_list = []
        self.holding = 0
        self.balance_input = 0
        self.net_worth_input = 0
        
        
        
        return self._next_observation()
    
    def _next_observation(self):
        #cpy_data = self.data.copy().drop(["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"], axis=1)
        #frame = self.data.loc[self.current_step-self.lookback+1:self.current_step,cpy_data.columns]
        frame = self.data.loc[self.current_step-self.lookback+1:self.current_step,self.features]
        #print(frame)
        
        #print([self.balance, self.shares_held])
                
        obs = torch.cat([torch.tensor(frame.values).flatten(), torch.tensor([self.balance_input, self.shares_held, self.net_worth_input])])
        
        
        return obs
    
    def step(self, action):
        
        action_bonus, current_price = self._take_action(action)
        
        done = 0
        
        if self.current_step >= self.data.shape[0] or ((self.current_step_idx + 1) % self.max_steps == 0):
            done = 1
        elif (self.net_worth <= 0):
            done =  1
            
        discount = (((self.current_step_idx % self.max_steps) + 1) / self.max_steps) * (0.9999)**self.current_step_idx
        discount = 1
        
        weighted_net_worth = 0.35 * self.balance + 0.65 * self.shares_held * current_price
        
        #reward = discount * self.net_worth - self.current_step_idx * self.holding + action_bonus
        reward = discount * weighted_net_worth - np.sqrt(self.current_step_idx) * self.holding + action_bonus
        
        #print(discount * self.net_worth, - (self.current_step_idx) * self.holding, action_bonus)
        
        self.current_step += 1
        self.current_step_idx += 1
        
        #print("Reward : ", reward)
        
        if self.current_step_idx+1 % 50 == 0:
            print("Step ", self.current_step)
            print("Balance :", self.balance)
            print("Shares :", self.shares_held)
            print("Net Worth :", self.net_worth)
        
        obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def _take_action(self, action):
        
        current_price = np.random.uniform(self.data.loc[self.current_step,"Low"], self.data.loc[self.current_step, "High"])
        
        if self.current_step_idx == 0:
            self.price_list.append(current_price)
        
        reward = 0
        
        if (self.balance <= 5 and action in [0, 1, 2]) or (self.shares_held * current_price <= 2 and action in [3, 4, 5]):
            reward = - self.net_worth * 0.5
        else:        
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
            elif action == 4:
                shares_sold = 0.5 * self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
                
            #Convert 100% of shares into balance    
            elif action == 5:
                shares_sold = self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
        
        if action == 6:
            self.holding += 1
        else:
            self.holding = 0
            
        if self.balance >= MAX_BALANCE:
            self.balance = MAX_BALANCE
                
        if self.shares_held * current_price >= MAX_BALANCE:
                self.shares_held = MAX_BALANCE / current_price
                
        self.net_worth = self.balance + self.shares_held * current_price
        
        self.balance_input = self.balance / MAX_BALANCE
        
        self.net_worth_input = self.net_worth / MAX_BALANCE
        
        self.balance_list.append(self.balance)
        self.shares_held_list.append(self.shares_held)
        self.net_worth_list.append(self.net_worth)
        self.price_list.append(current_price)
        
        return reward, current_price
        
    def render(self, mode='human', close=False):
        print("Step ", self.current_step)
        print("Balance :", self.balance)
        print("Shares :", self.shares_held)
        print("Net Worth :", self.net_worth)
        steps = np.arange(len(self.balance_list))
        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(2, 2, figsize=(20, 10))
        
        axis[0,0].plot(steps, self.net_worth_list, label="Net Worth")
        axis[0,0].set_title("Evolution of Net Worth in USD")
        
        axis[1,0].plot(steps, self.balance_list, label="Balance")
        axis[1,0].set_title("Evolution of Balance in USD")
        
        axis[0,1].plot(steps, self.shares_held_list, label="Shares Held")
        axis[0,1].set_title("Evolution of Held Shares")
        
        axis[1,1].plot(steps, self.price_list, label="Asset Price")
        axis[1,1].set_title("Evolution of Asset Price valued against USD")
        
        plt.show()
        
        
        
if __name__ == "__main__":
    print("Testing Trading Environment")
    
    print("Loading Data...\t", end='')
    
    data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    print("Done")
    
    env = TradingEnv(data)
    
    env.reset()
    
    for epoch in range(1000):
        #action = np.random.randint(0,7)
        
        action = -1
        
        while action not in [i for i in range(0,7)]:
            print("Enter action")
            action = int(input())
        print("Valid")
        
        obs, _, _, _ = env.step(action)
        
        print(obs)
        
        if epoch % 100 == 0:
            pass
        env.render()
            
    env.render()