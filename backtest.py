import pandas as pd
from datahandler import *
from data_processing import *
import torch
import matplotlib.pyplot as plt
import mplfinance as mpf

def backtest_crossover(data, predict_data, model, seq_length, device, balance=100):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    
    N = np.arange(4 * seq_length + 1)
    
    assets = balance
    
    asset_value = [balance]
    
    i = 0
    while i < dataset.shape[0]:
        print('\r{}'.format(i * 100 / data.shape[0]), end='\r')

        if i >= data.shape[0] - seq_length:
            break
        
        seq = dataset.loc[i:i + seq_length].values
        
        hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
        model.num_layers, model.hidden_size).to(device))
        output, _ = model(torch.Tensor(seq).to(device), hidden)
                        
        output = output[-1, :]
        prob = torch.max(output, axis=-1).values.cpu()
                
        output = int((torch.argmax(output, axis=-1) - 1).cpu())

        curr_close = data.loc[i + seq_length, 'Close']
        
        if output == 0 or prob < 0.9:
            asset_value.append(assets)
            i += 1
            continue
        
        elif output == 1:
            tp = 1.005 * curr_close
            sl = 0.9975 * curr_close
            
        elif output == -1:
            tp = 0.995 * curr_close
            sl = 1.0025 * curr_close
        
        check_profit_data = data.loc[i + seq_length + 1 :i + seq_length + 7, ['Open', 'High', 'Low', 'Close']]
        out = False
        for _, row in check_profit_data.iterrows():
            if output == 1:
                if row['High'] >= tp:
                    assets *= 1.005
                    break
                elif row['Low'] <= sl:
                    assets *= 0.9975
                    out = True
                    break
            elif output == -1:
                if row['Low'] <= tp:
                    assets *= 1.005
                    out = True
                    break
                elif row['High'] >= sl:
                    assets *= 0.9975
                    out = True
                    break
        if not out:
            assets *= row['Close'] / curr_close
                
        asset_value.append(assets)
        
        i += seq_length + 8
    
    print()
    plt.plot(range(len(asset_value)), asset_value, label='Portfolio Value')
    plt.title('Evolution of asset value')
    plt.legend()
    plt.show()