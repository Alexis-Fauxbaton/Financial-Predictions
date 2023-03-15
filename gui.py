import PySimpleGUI as sg
import plotly.graph_objects as go


def plot_candlesticks(dataset):
    fig = go.Figure(data=[go.Candlestick(x=dataset["Unix"],
                                         open=dataset['Open'], high=dataset['High'],
                                         low=dataset['Low'], close=dataset['Close'])])
    fig.show()


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
