import gym
from gym import spaces


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data):
        super(TradingEnv).__init__()
        
        self.data = data
        
        #Define the number of possible actions here
        
        #Actions : BUY || SELL || NOTHING
        #How Much : 10 || 50 || 100
        #9 Combinations (How much will not matter in the case of NOTHING)
        self.action_space = spaces.Discrete(9)
        
        self.observation_space = spaces.Box()