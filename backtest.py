import pandas as pd
from datahandler import *
from data_processing import *
import torch
import matplotlib.pyplot as plt
import mplfinance as mpf

def baseline_crossover(data, predict_data, seq_length, balance=100, noise_threshold=0.6, target_return=0.005, ratio=2):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    
    N = np.arange(4 * seq_length + 1)
    
    assets = balance
    
    asset_value = [balance]
    
    i = 0
    profits = []
    while i < dataset.shape[0]:
        print('\r{}'.format(i * 100 / data.shape[0]), end='\r')

        if i >= data.shape[0] - seq_length:
            break  
                        
        output = predict_data.loc[i + seq_length, 'Target']
    
        curr_close = data.loc[i + seq_length, 'Close']
        
        rnd = np.random.uniform(0, 1)
        
        if output == 0 and rnd > noise_threshold: # Add noise to see at which precision threshold the strategy works
            if predict_data.loc[i + seq_length, 'MA10 UP']:
                output = 1
            else:
                output = -1 
            
        if output == 0:
            asset_value.append(assets)
            i += 1
            continue
        
        elif output == 1:
            # tp = 1.005 * curr_close # 15 min standard 
            # sl = 0.9975 * curr_close
            tp = (1 + target_return) * curr_close
            sl = (1 - target_return / ratio) * curr_close
            
        elif output == -1:
            # tp = 0.995 * curr_close
            # sl = 1.0025 * curr_close
            tp =  curr_close / (1 + target_return)
            sl =  curr_close / (1 - target_return / ratio)
        
        check_profit_data = data.loc[i + seq_length + 1 :i + seq_length + 7, ['Open', 'High', 'Low', 'Close']]
        out = False
        for _, row in check_profit_data.iterrows():
            if output == 1:
                if row['High'] >= tp:
                    profits.append(2)
                    assets *= 1.005
                    out = True
                    break
                elif row['Low'] <= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
            elif output == -1:
                if row['Low'] <= tp:
                    profits.append(2)
                    assets *= 1.005
                    out = True
                    break
                elif row['High'] >= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
        if not out:
            if row['Close'] > curr_close:
                profits.append(output * 1)
            else:
                profits.append(output * (-1))
            assets *= row['Close'] / curr_close
                
        asset_value.append(assets)
        
        i += seq_length + 8
        
    print()
    print('\r', pd.Series(profits).value_counts(), len(profits))
    plt.plot(range(len(asset_value)), asset_value, label='Portfolio Value')
    plt.title('Evolution of asset value')
    plt.legend()
    plt.show()

#TODO Add a clause that checks for past volatility before entering a trade
def backtest_crossover(data, predict_data, model, seq_length, device, balance=100, target_return=0.005, ratio=2):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    
    N = np.arange(4 * seq_length + 1)
    
    assets = balance
    
    asset_value = [balance]
    
    i = 0
    profits = []
    while i < dataset.shape[0]:
        print('\r{}'.format(i * 100 / data.shape[0]), end='\r')

        if i >= data.shape[0] - seq_length:
            break
        
        seq = dataset.loc[i:i + seq_length].values
        volatility = (seq['High'] - seq['Low']).std()

        if not model.bidirectional:
            hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
                    model.num_layers, model.hidden_size).to(device))  # Hidden state and cell state
        else:
            hidden = (torch.zeros(model.num_layers * 2, model.hidden_size).to(device), torch.zeros(
                    model.num_layers * 2, model.hidden_size).to(device))
            
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
            # tp = 1.005 * curr_close # 15 min standard 
            # sl = 0.9975 * curr_close
            tp = (1 + target_return) * curr_close
            sl = (1 - target_return / ratio) * curr_close
            
        elif output == -1:
            # tp = 0.995 * curr_close
            # sl = 1.0025 * curr_close
            tp =  curr_close / (1 + target_return)
            sl =  curr_close / (1 - target_return / ratio)
            
        
        check_profit_data = data.loc[i + seq_length + 1 :i + seq_length + 7, ['Open', 'High', 'Low', 'Close']]
        out = False
        for _, row in check_profit_data.iterrows():
            if output == 1:
                if row['Low'] <= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
                elif row['High'] >= tp:
                    profits.append(2)
                    assets *= 1.005
                    out=True
                    break
            elif output == -1:
                if row['High'] >= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
                elif row['Low'] <= tp:
                    profits.append(2)
                    assets *= 1.005
                    out = True
                    break
                
        if not out:
            if row['Close'] > curr_close:
                profits.append(output * 1)
            else:
                profits.append(output * (-1))
            assets *= row['Close'] / curr_close
                
        asset_value.append(assets)
        
        i += seq_length + 8
    
    print()
    print('\r', pd.Series(profits).value_counts(), len(profits))
    plt.plot(range(len(asset_value)), asset_value, label='Portfolio Value')
    plt.title('Evolution of asset value')
    plt.legend()
    plt.show()
    

def backtest_crossover_v2(data, predict_data, model, seq_length, device, balance=100, target_return=0.005, ratio=2):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    
    N = np.arange(4 * seq_length + 1)
    
    assets = balance
    
    asset_value = [balance]
    
    i = 0
    output_buffer = [] # Evaluate the last 3 outputs
    profits = []
    while i < dataset.shape[0]:
        print('\r{}'.format(i * 100 / data.shape[0]), end='\r')

        if i >= data.shape[0] - seq_length:
            break
        
        seq = dataset.loc[i:i + seq_length].values
        
        #hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
        #model.num_layers, model.hidden_size).to(device))
        if not model.bidirectional:
            hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
                    model.num_layers, model.hidden_size).to(device))  # Hidden state and cell state
        else:
            hidden = (torch.zeros(model.num_layers * 2, model.hidden_size).to(device), torch.zeros(
                    model.num_layers * 2, model.hidden_size).to(device))
            
        output, _ = model(torch.Tensor(seq).to(device), hidden)
                        
        output = output[-1, :]
        prob = torch.max(output, axis=-1).values.cpu()
                
        output = int((torch.argmax(output, axis=-1) - 1).cpu())

        output_buffer.append(output)
        
        l = len(output_buffer)
        
        if l == 1 or l == 2:
            i += 1
            continue
        
        elif l == 3:
            if not (output_buffer[0] == output_buffer[1] == output_buffer[2]):
                output_buffer = []
                i += 1
                continue
            else:
                output_buffer.pop(0)
        
        curr_close = data.loc[i + seq_length, 'Close']
        
        if output == 0 or prob < 0.9:
            asset_value.append(assets)
            i += 1
            continue
        
        elif output == 1:
            # tp = 1.005 * curr_close # 15 min standard 
            # sl = 0.9975 * curr_close
            tp = (1 + target_return) * curr_close
            sl = (1 - target_return / ratio) * curr_close
            
        elif output == -1:
            # tp = 0.995 * curr_close
            # sl = 1.0025 * curr_close
            tp =  curr_close / (1 + target_return)
            sl =  curr_close / (1 - target_return / ratio)
        
        check_profit_data = data.loc[i + seq_length + 1 :i + seq_length + 7, ['Open', 'High', 'Low', 'Close']]
        out = False
        for _, row in check_profit_data.iterrows():
            if output == 1:
                if row['High'] >= tp:
                    profits.append(2)
                    assets *= 1.005
                    out=True
                    break
                elif row['Low'] <= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
            elif output == -1:
                if row['Low'] <= tp:
                    profits.append(2)
                    assets *= 1.005
                    out = True
                    break
                elif row['High'] >= sl:
                    profits.append(-2)
                    assets *= 0.9975
                    out = True
                    break
        if not out:
            if row['Close'] > curr_close:
                profits.append(output * 1)
            else:
                profits.append(output * (-1))
            assets *= row['Close'] / curr_close
                
        asset_value.append(assets)
        
        i += seq_length + 8
    
    print()
    print('\r', pd.Series(profits).value_counts(), len(profits))
    plt.plot(range(len(asset_value)), asset_value, label='Portfolio Value')
    plt.title('Evolution of asset value')
    plt.legend()
    plt.show()