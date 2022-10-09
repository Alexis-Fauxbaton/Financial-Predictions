import plotly.graph_objects as go
import pandas as pd



if __name__ == "__main__":
    data = pd.read_csv("dollar_data/dollar_data.csv")
    data = data.loc[data["Unix"] >= 1609459200][0:1000]

    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

    fig.update_layout(
        xaxis_rangeslider_visible='slider' in range(100)
    )

    fig.show()