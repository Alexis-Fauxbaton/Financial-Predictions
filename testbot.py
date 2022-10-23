import tensorflow as tf
import pandas as pd
import predict
import data_processing
import time

from binance.spot import Spot as Client
from binance.websocket.spot.websocket_client import SpotWebsocketClient as WebsocketClient

def message_handler(message):
    print(message)

def get_balances(client, symbols):
    content = client.account()
    balances = pd.DataFrame(content['balances'])
    outputs = [None for i in symbols]

    i = 0
    for symbol in symbols:
        outputs[i] = float(balances.loc[balances["asset"] == symbol, "free"])
        
        i+=1
    return tuple(outputs)

if __name__ == "__main__":
    max_days = predict.max_days
    target_range = predict.target_range
    
    
    #Loading means and stds from non processed dataframe to standardize new data
    cols, prev_means, prev_stds = data_processing.process_minute_data(False)
    
    #values at the initialization of test bot
    start_btc = 1
    start_usdt = 10000

    with open('TEST_API', 'r') as f:
        raw_text = f.read()
        content = raw_text.split('\n')
        key,secret = content[1],content[-1]

    client = Client(key=key, secret=secret, base_url='https://testnet.binance.vision')
    
    start = time.time()
    print(client.time())
    end = time.time()
    
    print("Time taken to retrieve client time : {}s".format(end-start))
    
    print(client.ui_klines("BTCUSDT", "1m", limit=max_days))
            
    #print(client.account())
    
    #print(client.user_asset())
    
    
    start = time.time()
    usdt_balance, btc_balance = get_balances(client, ["USDT", "BTC"])
    end = time.time()
    
    print("Time taken to retrieve balances : {}s".format(end-start))
    
    print("USDT Balance : {}\nBTC Balance : {}".format(usdt_balance, btc_balance))
    
    print(cols, prev_means, prev_stds)
    
    #model = predict.train_and_return_model_for_2021_2022()
    
    """
    curr_time = client.time()
    while 1:
        curr_time = client.time()
        if curr_time % 60 == 0:
            predict_data = client.klines("BTCUSDT", "1m", limit=max_days)
        
        time.sleep(0.9)
    """
    
    
    