import PySimpleGUI as sg
import plotly.graph_objects as go
from torchutils import *
from data_processing import *
from playground import *
import torch
import matplotlib.pyplot as plt

def plot_candlesticks(dataset):
    fig = go.Figure(data=[go.Candlestick(x=dataset["Unix"],
                                         open=dataset['Open'], high=dataset['High'],
                                         low=dataset['Low'], close=dataset['Close'])])
    fig.show()

def visualize_model_outputs(data, predict_data, model, seq_length, device):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    val_indices = range(round(0.7 * dataset.shape[0])), round(0.9 * dataset.shape[0])
    test_indices = range(round(0.9 * dataset.shape[0]), dataset.shape[0] - seq_length)
    
    N = np.arange(4 * seq_length + 1)
    
    cmap = {-1: 'red', 0: 'black', 1: 'green'}
    for idx, _ in data.iterrows():
        if idx >= data.shape[0] - seq_length:
            break
                
        if idx % seq_length != 0 or idx - 2 * seq_length < 0:
            continue
    
        seq = dataset.loc[idx:idx + seq_length].values
        
        hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
        model.num_layers, model.hidden_size).to(device))
        output, _ = model(torch.Tensor(seq).to(device), hidden)
                        
        output = output[-1, :]
                
        output = int((torch.argmax(output, axis=-1) - 1).cpu())

        curr_close = data.loc[idx + seq_length - 1, 'Close']
        
        if output == 0:
            continue
        
        elif output == 1:
            tp = 1.005 * curr_close
            sl = 0.9975 * curr_close
            
        elif output == -1:
            tp = 0.995 * curr_close
            sl = 1.0025 * curr_close
        
        closes = data.loc[idx - 2 * seq_length : idx + 2 * seq_length, 'Close']
        ma5 = data.loc[idx - 2 * seq_length : idx + 2 * seq_length, 'MA5']
        ma10 = data.loc[idx - 2 * seq_length : idx + 2 * seq_length, 'MA10']
        
        pred_index = 3 * seq_length - 1
        
        color = cmap[output]
        
        plt.plot(N, closes, label='Close')
        plt.plot(N, ma5, label='MA5')
        plt.plot(N, ma10, label='MA10')        
        plt.axvline(pred_index, color=color, linestyle='--')
        plt.axhline(tp, color='green', linestyle='--')
        plt.axhline(sl, color='red', linestyle='--')
        plt.legend()
        plt.show()
        
        print('Done')
    
def test():
    layout = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("OK")]]

    # Create the window
    window = sg.Window("Demo", layout)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

    window.close()


if __name__ == "__main__":
    test()
    
