import PySimpleGUI as sg
import plotly.graph_objects as go
from torchutils import *
from data_processing import *
from playground import *
import torch
import matplotlib.pyplot as plt
import mplfinance as mpf


def plot_candlesticks(dataset):
    fig = go.Figure(data=[go.Candlestick(x=dataset["Unix"],
                                         open=dataset['Open'], high=dataset['High'],
                                         low=dataset['Low'], close=dataset['Close'])])
    fig.show()


def visualize_model_outputs(data, predict_data, model, seq_length, device, display=True):
    dataset = predict_data.drop(['Target', 'Unix'], axis=1, errors='ignore')
    val_indices = range(
        round(0.7 * dataset.shape[0])), round(0.9 * dataset.shape[0])
    test_indices = range(
        round(0.9 * dataset.shape[0]), dataset.shape[0] - seq_length)

    N = np.arange(4 * seq_length + 1)

    # cmap = {-1: 'red', 0: 'black', 1: 'green'}
    cmap = {-2: 'red', -1: 'orange', 1: 'blue', 2: 'green'}
    profits = []
    idx = dataset.index[0]
    while idx < dataset.shape[0]:
        print('\r{}'.format(idx * 100 / data.shape[0]), end='\r')
        if idx >= data.shape[0] - seq_length:
            break

        # if idx % seq_length != 0 or idx - 2 * seq_length < 0:
        if idx - 2 * seq_length < 0:
            idx += 1
            continue

        seq = dataset.loc[idx:idx + seq_length].values

        hidden = (torch.zeros(model.num_layers, model.hidden_size).to(device), torch.zeros(
            model.num_layers, model.hidden_size).to(device))
        output, _ = model(torch.Tensor(seq).to(device), hidden)

        output = output[-1, :]
        prob = torch.max(output, axis=-1).values.cpu()


        output = int((torch.argmax(output, axis=-1) - 1).cpu())

        curr_close = data.loc[idx + seq_length, 'Close']

        if output == 0 or prob < 0.9:
            idx += 1
            continue

        elif output == 1:
            tp = 1.005 * curr_close
            sl = 0.9975 * curr_close

        elif output == -1:
            tp = 0.995 * curr_close
            sl = 1.0025 * curr_close

        check_profit_data = data.loc[idx + seq_length + 1:idx +
                                     seq_length + 7, ['Open', 'High', 'Low', 'Close']]
        out = False
        c = 1
        for _, row in check_profit_data.iterrows():
            if output == 1:
                if row['High'] >= tp:
                    profits.append(2)
                    out = True
                    break
                elif row['Low'] <= sl:
                    profits.append(-2)
                    out = True
                    break
            elif output == -1:
                if row['Low'] <= tp:
                    profits.append(2)
                    out = True
                    break
                elif row['High'] >= sl:
                    profits.append(-2)
                    out = True
                    break
            c += 1
        if not out:
            if row['Close'] > curr_close:
                profits.append(1)
            else:
                profits.append(-1)

        closes = data.loc[idx - 2 * seq_length: idx + 2 * seq_length, 'Close']
        ma5 = data.loc[idx - 2 * seq_length: idx + 2 * seq_length, 'MA5']
        ma10 = data.loc[idx - 2 * seq_length: idx + 2 * seq_length, 'MA10']

        pred_index = 3 * seq_length

        if display and output == -1:
            # color = cmap[output]
            plot_data = data.loc[idx - 2 * seq_length: idx + 2 * seq_length].reset_index(drop=True)
            plot_data['Old Index'] = plot_data.index
            plot_data.index = pd.to_datetime(plot_data['Unix'], unit='ms')

            color = cmap[profits[-1]]
            ma_plot = mpf.make_addplot(plot_data[['MA5', 'MA10']])
            entry_date = plot_data.loc[plot_data['Old Index']
                                       == pred_index].index[0]
            exit_date = plot_data.loc[plot_data['Old Index']
                                      == pred_index + 7].index[0]
            mpf.plot(plot_data[['Open', 'High', 'Low', 'Close']], type='candle', style='yahoo', title=f'{output}', addplot=[ma_plot], hlines=dict(
                hlines=[tp, sl], colors=['g', 'r'], linestyle='--'), vlines=dict(vlines=[entry_date, exit_date], colors=[color, 'black'], linestyle='--'), savefig=f'./backtest_img/{idx}.png')
            # plt.plot(plot_data['MA5'], label='MA5')
            # plt.plot(plot_data['MA10'], label='MA10')
            # plt.plot(N, closes, label='Close')
            # plt.plot(N, ma5, label='MA5')
            # plt.plot(N, ma10, label='MA10')
            # plt.axvline(plot_data.loc[plot_data['Old Index'] == pred_index].index, color=color, linestyle='--')
            # plt.axvline(plot_data.loc[plot_data['Old Index'] == pred_index + 7].index, color='black', linestyle='--')
            # plt.axhline(tp, color='green', linestyle='--')
            # plt.axhline(sl, color='red', linestyle='--')
            # plt.title(f"{output}")
            # plt.legend()
            # plt.show()
            # plt.savefig(f'./backtest_img/{idx}.png')
        idx += (c + 1)
    print()
    print('\r', pd.Series(profits).value_counts(), len(profits))


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
