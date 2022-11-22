from a2c import *
from RL_env import *
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


if __name__ == "__main__":
    print("Testing PPO + Environment")
    
    print("Loading Data...\t", end='')
    
    data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    print("Done")
    
    removed_cols = ["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"]
    
    
    #env = DummyVecEnv([lambda:TradingEnv(data)])

    env = TradingEnv(data)

    input_size = env.lookback * (len(data.columns) - len(removed_cols)) + 3 #2 for balance, shares held and net worth
    
    print("Input Size : ", input_size)
    
    print("Output Size : ", OUTPUT_SIZE)
    
    ppo = ActorCritic(input_size, OUTPUT_SIZE)
    
    ppo.train(env, 1000, env.max_steps)
    
    '''
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    ppo = ActorCritic(1, 4)
    
    ppo.train(env, 100)
    '''
    
    #ppo = PPO("MlpPolicy", env, verbose=1)
    
    #ppo.learn(total_timesteps=10_000)
    