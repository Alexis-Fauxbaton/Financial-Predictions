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
    
    def __init__(self, data, lookback=60, max_steps=300, same_window=False):
        super(TradingEnv).__init__()
        
        self.data = data
        cpy_data = self.data.copy().drop(["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"], axis=1)
        self.features = cpy_data.columns
        self.lookback = lookback
        self.max_steps = max_steps
        self.actions = []
        self.discount_reward = 1
        self.window_start = np.random.randint(0 + self.lookback,self.data.shape[0] - self.max_steps)
        self.same_window = same_window
        self.ep_count = 0
        #Define the number of possible actions here
        
        #Actions : BUY || SELL || NOTHING
        #How Much : 10 || 50 || 100 % of balance
        #7 Combinations (Only one case with NOTHING)
        self.action_space = spaces.Discrete(OUTPUT_SIZE)
        
        #We assume that the only columns that are not fed to the agent are unix date + last 4 OHLC (1 + 4)
        #Each row of data contains the OHLC, Unix, perct_change since last candle and indicators needed for the agent FOR 1 TIMESTAMP
        #self.observation_space = spaces.Tuple(spaces.Box(low=[-np.inf for i in range(self.data.shape[1]-(1 + 4))], high=[np.inf for i in range(self.data.shape[1]-(1 + 4))], shape=(self.lookback, self.data.shape[1] - (1 + 4))), spaces.Box(low=[-np.inf, -np.inf], high=[np.inf, np.inf], shape=(1,2)))
        #self.observation_space = spaces.Tuple([spaces.Box(low=np.array([-np.inf for i in range(self.data.shape[1]-(1 + 4))]), high=np.array([np.inf for i in range(self.data.shape[1]-(1 + 4))]), dtype=np.float64), 
        #                                       spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.float64)])
        
        print("Box size : ", (len(self.features)) * self.lookback + 3)
        self.observation_space = spaces.Box(low=np.array([-np.inf for i in range((len(self.features)) * self.lookback + 3)]), high=np.array([np.inf for i in range((len(self.features)) * self.lookback + 3)]), dtype=np.float64)
    def reset(self):
        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.net_worth = 0
        if not self.same_window:
            self.current_step = np.random.randint(0 + self.lookback,self.data.shape[0] - self.max_steps)
        else:
            self.current_step = self.window_start
        self.current_step_idx = 0
        #self.balance_list = [self.balance]
        #self.shares_held_list = [self.shares_held]
        #self.net_worth_list = [self.net_worth]
        
        self.balance_list = []
        self.shares_held_list = []
        self.net_worth_list = []
        
        self.price_list = []
        self.actions = []
        self.weighted_net_worth_list = []
        self.reward_list = []
        self.sharpe_ratio_list = []
        self.discount_reward = 1
        self.holding = 0
        self.balance_input = 0
        self.net_worth_input = 0
        self.ep_count += 1
        print("Episode : ", self.ep_count)
        
        
        
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
        self.weighted_net_worth_list.append(weighted_net_worth)


        wnw_change = 0
        if len(self.weighted_net_worth_list) == 1:
            wnw_change = 0
        else:
            wnw_change = 100 * (self.weighted_net_worth_list[-1] - self.weighted_net_worth_list[-2]) / self.weighted_net_worth_list[-2]
        
        #TODO Prendre en compte le nombre de trades gagnants dans la reward + moyenne exponentielle des rewards de tout l'épisode (?)
        #TODO Tenter d'utiliser le mouvement du prix futur pour influencer la reward
        action_reward_weight = [2, 5, 10]
        reward_horizon = 40
    
        curr_closing_price = self.data.loc[self.current_step, "Close"]
        horizon_closing_price = self.data.loc[self.current_step + reward_horizon, "Close"]
        perc_change = (horizon_closing_price - curr_closing_price) / curr_closing_price
        
        bonus = 0
        
        if action in range(0,6):
            bonus = action_reward_weight[action % len(action_reward_weight)]
        else:
            bonus = 5
        
        holding_bonus = 0
        if perc_change >= 0:
            if action in range(0,3):
                base_reward = 5
            elif action == 6:
                base_reward = 0
            else:
                base_reward = -5
            holding_bonus = 5
        elif perc_change < 0:
            if action in range(3,6):
                base_reward = 5
            elif action == 6:
                base_reward = 0
            else:
                base_reward = -5
            holding_bonus = -5
        '''  
        holding_malus = 0    
        if action == 6:
            holding_malus = 5
        '''
        
        #Calculate Ep sharpe ratio
        sharpe_window = 20
        strategy = pd.Series(self.net_worth_list[-sharpe_window:-1])
        sharpe_ratio = (strategy.diff().mean() / strategy.std()) * np.sqrt(sharpe_window)
        if self.current_step_idx < sharpe_window:
            sharpe_ratio = 0
        if (not isinstance(sharpe_ratio, float)) or sharpe_ratio == np.nan or strategy.std() == 0:
            sharpe_ratio = 0
        #print(sharpe_ratio)
        self.sharpe_ratio_list.append(sharpe_ratio)
        #sharpe_ratio = (Average(52 week series of weekly returns)/StDev(52 week series of weekly returns))*52^.5
        #reward = discount * pow(weighted_net_worth, 1.1) - self.holding + action_bonus + self.shares_held * (current_price / INITIAL_BALANCE) * 100 To work with
        reward = 100 * sharpe_ratio +  0.15 * weighted_net_worth +  0.4 * base_reward * bonus + 0.25 * action_bonus + 0.2 * holding_bonus * (self.shares_held * 100 - self.balance_input)
        #reward *= self.discount_reward
        
        if reward >= 0:
            self.discount_reward *= 0.999
        else:
            self.discount_reward *= 1.001
            
        self.reward_list.append(reward)
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
        
        if self.current_step_idx == 299 and self.ep_count % 2000 == 0:
            self.render()
        
        return obs, reward, done, {}
    
    def _take_action(self, action):
        
        current_price = np.random.uniform(self.data.loc[self.current_step,"Low"], self.data.loc[self.current_step, "High"])
        
        #if self.current_step_idx == 0:
        #    self.price_list.append(current_price)
        
        self.balance_list.append(self.balance)
        self.shares_held_list.append(self.shares_held)
        self.net_worth_list.append(self.net_worth)
        self.price_list.append(current_price)
        
        reward = 0
        
        if (self.balance <= 2 and action in [0, 1, 2]) or (self.shares_held * current_price <= 0.5 and action in [3, 4, 5]):
            reward = - self.net_worth * 0.5
            self.actions.append(6)
        else:        
            #Convert 10% of balance into asset
            if action == 0:                 
                shares_bought = (0.1 * self.balance) / current_price
                self.shares_held += shares_bought
                self.balance = self.balance * 0.9
                self.actions.append(action)
            
            #Convert 50% of balance into asset
            elif action == 1:
                shares_bought = (0.5 * self.balance) / current_price
                self.shares_held += shares_bought
                self.balance = self.balance * 0.5
                self.actions.append(action)
                
            #Convert 100% of balance into asset            
            elif action == 2:
                shares_bought = (1 * self.balance) / current_price
                self.shares_held += shares_bought
                self.balance = 0
                self.actions.append(action)
            
            #Convert 10% of shares into balance    
            elif action == 3:
                shares_sold = 0.1 * self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
                self.actions.append(action)
                
            #Convert 50% of shares into balance    
            elif action == 4:
                shares_sold = 0.5 * self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
                self.actions.append(action)
                
            #Convert 100% of shares into balance    
            elif action == 5:
                shares_sold = self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
            
                self.actions.append(action)
        
        if action == 6:
            self.holding += 1
            self.actions.append(action)
        else:
            self.holding = 0
            
        if self.balance >= MAX_BALANCE:
            self.balance = MAX_BALANCE
                
        if self.shares_held * current_price >= MAX_BALANCE:
                self.shares_held = MAX_BALANCE / current_price
                
        self.net_worth = self.balance + self.shares_held * current_price
        
        self.balance_input = self.balance / MAX_BALANCE
        
        self.net_worth_input = self.net_worth / MAX_BALANCE
        
        #self.balance_list.append(self.balance)
        #self.shares_held_list.append(self.shares_held)
        #self.net_worth_list.append(self.net_worth)
        #self.price_list.append(current_price)
        
        return reward, current_price
        
    def render(self, mode='human', close=False):
        print("Step ", self.current_step)
        print("Balance :", self.balance)
        print("Shares :", self.shares_held)
        print("Net Worth :", self.net_worth)
        print("Average reward :", np.mean(self.reward_list))
        print("Baseline Strategy Net Worth:", INITIAL_BALANCE * self.price_list[-1] / self.price_list[0])
        alphas = [0.25, 0.5, 1]
        steps = np.arange(len(self.balance_list))
        price = pd.Series(self.price_list)
        actions = pd.Series(self.actions)
        actions = actions.reindex_like(price)
        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(2, 2, figsize=(20, 10))
        
        axis[0,0].plot(steps, self.net_worth_list, label="Net Worth")
        axis[0,0].set_title("Evolution of Net Worth in USD")
        
        axis[1,0].plot(steps, self.balance_list, label="Balance")
        axis[1,0].set_title("Evolution of Balance in USD")
        
        axis[0,1].plot(steps, self.price_list, label="Strategy Sharpe Ratio")
        for i in range(OUTPUT_SIZE):
            if i in range(0,3):
                axis[0,1].scatter(price[actions == i].index, price[actions == i], color = 'b', marker = 'o', alpha = alphas[i], label="Buy")
            elif i in range(3,6):
                axis[0,1].scatter(price[actions == i].index, price[actions == i], color = 'r', marker = 'o', alpha = alphas[i % 3], label="Sell")
        axis[0,1].set_title("Evolution of Asset Price valued against USD + Sharpe Ratio of strategy")
        sr_axis = axis[0,1].twinx()
        #reward_list = self.reward_list
        sr_axis.plot(steps, self.sharpe_ratio_list, label="Sharpe Ratio", color='m', alpha = 0.25)
        sr_axis.set_ylabel('Sharpe Ratio', color='m')
       
                
        axis[1,1].plot(steps, self.price_list, label="Asset Price")
        for i in range(OUTPUT_SIZE):
            if i in range(0,3):
                axis[1,1].scatter(price[actions == i].index, price[actions == i], color = 'b', marker = 'o', alpha = alphas[i], label="Buy")
            elif i in range(3,6):
                axis[1,1].scatter(price[actions == i].index, price[actions == i], color = 'r', marker = 'o', alpha = alphas[i % 3], label="Sell")
                
        axis[1,1].set_title("Evolution of Asset Price valued against USD")
        
        reward_axis = axis[1,1].twinx()
        #reward_list = self.reward_list
        reward_axis.plot(steps, self.reward_list, label="Reward", color='g', alpha = 0.25)
        reward_axis.set_ylabel('Rewards', color='g')
        
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