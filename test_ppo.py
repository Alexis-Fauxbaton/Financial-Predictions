from a2c import *
from RL_env import *
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from time import sleep


if __name__ == "__main__":
    print("Testing PPO + Environment")
    
    print("Loading Data...\t", end='')
    
    #data = pd.read_csv("minute_data/BTC-USD_1M_SIGNALS.csv")
    #data = pd.DataFrame({})
    print("Done")
    
    removed_cols = ["Open", "Low", "Close", "High", "Unix", "Symbol", "Date"]
    
    
    #env = DummyVecEnv([lambda:TradingEnv(data)])

    '''
    env = TradingEnv(data)



    input_size = env.lookback * (len(data.columns) - len(removed_cols)) + 3 #2 for balance, shares held and net worth
    
    #input_size = 30 * (len(data.columns) - len(removed_cols)) + 3 #2 for balance, shares held and net worth

    print("Input Size : ", input_size)
    
    print("Output Size : ", OUTPUT_SIZE)
    
    #ppo = ActorCritic(input_size, OUTPUT_SIZE)
    
    #ppo.train(env, 50, env.max_steps)
    
    '''
    
    
    
    
    
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    #ppo = ActorCritic(1, 4)
    
    #ppo.train(env, 100)

    # Custom MLP policy of three layers of size 128 each
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[320, 3200, 3200, dict(pi=[3200, 320, 32], vf=[3200, 320, 32])])

    ppo = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    
    ppo.learn(total_timesteps=100000)
    
    obs = env.reset()
    while True:
        action, _states = ppo.predict(obs)
        obs, rewards, dones, info = env.step(int(action))
        env.render()
        if dones == 1:
            obs = env.reset()
        sleep(1)
    
    
    #ppo = PPO("MlpPolicy", env, verbose=1)
    
    #ppo.learn(total_timesteps=10_000)
    #ppo.learn(total_timesteps=1000000)


    
    