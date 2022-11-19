from a2c import *
from RL_env import *


if __name__ == "__main__":
    print("Testing PPO + Environment")
    
    print("Loading Data...\t", end='')
    
    data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    print("Done")
    
    removed_cols = ["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"]
    
    
    env = TradingEnv(data)

    input_size = env.lookback * (len(data.columns) - len(removed_cols)) + 2 #2 for balance and shares held
    
    print("Input Size : ", input_size)
    
    print("Output Size : ", OUTPUT_SIZE)
    
    ppo = ActorCritic(input_size, OUTPUT_SIZE)
    
    ppo.train(env, 10, env.max_steps)