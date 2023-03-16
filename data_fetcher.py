from datetime import datetime
from enum import Enum
from binance.spot import Spot as Client
from binance.websocket.spot.websocket_client import SpotWebsocketClient as WebsocketClient
import pandas as pd


class Pairs(Enum):
    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"


class TimeFrame(Enum):
    M1 = "1m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"


class TimeFrameMinuteConverter(Enum):
    M1 = 1
    M15 = 15
    H1 = 60
    D1 = 1440


class DataFetcher:

    def __init__(self) -> None:
        self.url = None


class BinanceDataFetcher(DataFetcher):

    def __init__(self, pair=Pairs.BTCUSDT.value, timeframe=TimeFrame.M15.value, start_year="2023",
                 start_month="01") -> None:
        super().__init__()
        self.url = "https://api.binance.com/api/v3/uiKlines"

        with open('LIVE_API', 'r') as f:
            raw_text = f.read()
            content = raw_text.split('\n')
            self.key, self.secret = content[1], content[-1]

        self.timeframe = timeframe

        self.client = Client(api_key=self.key, api_secret=self.secret, base_url="https://api.binance.com")

        self.pair = pair

    def getAllCandlesUntil(self, end_date="2023-02-28", timeframe: TimeFrame = TimeFrame.M15):
        start_time = "2017-08-01"

        format = "%Y-%m-%d"

        start_timestamp = int(
            datetime.strptime(start_time, format).timestamp()) * 1000  # *1000 to convert to milliseconds

        end_timestamp = int(datetime.strptime(end_date, format).timestamp()) * 1000  # *1000 to convert to milliseconds

        curr_time = start_timestamp

        candles = []

        limit = 10

        curr_time = self.client.ui_klines("BTCUSDT", timeframe.value, limit=1, startTime=start_timestamp)[0][0]

        while curr_time < end_timestamp:
            dt_object = datetime.fromtimestamp(int(curr_time / 1000))
            print("\r", dt_object, end="")
            r = self.client.ui_klines("BTCUSDT", timeframe.value, limit=limit, startTime=curr_time)
            candles += [r[i][:-3] for i in range(len(r))]

            curr_time += 15 * 60 * limit * 1000  # *1000 to convert to milliseconds

        print(str(dt_object))
        data = pd.DataFrame(candles, columns=["Unix", "Open", "High", "Low", "Close", "Volume", "Close Unix",
                                              "Quote Asset Volume", "NTrades"])
        print()
        print(data.tail())
        print(data.shape)

        data.drop_duplicates(inplace=True)

        data.to_csv(f"fetched_{self.pair}_{timeframe.value}.csv")


if __name__ == "__main__":
    fetcher = BinanceDataFetcher()

    fetcher.getAllCandlesUntil(timeframe=TimeFrame.M1)
