from dataclasses import dataclass
from trend_forecast_datahandler import TrendForecastDataHandler
from multiprocessing import Pool
import os
import imblearn

#TODO USE PANDAS-TA LIB FOR ADDING TECHNICAL INDICATORS

max_days = 30
target_range = 10

def main():
    handler = TrendForecastDataHandler("minute_data/BTC-USD_1M_SIGNALS.csv")
    
    handler.fit_predict(labeling = True, equal_sampling=True, sampling_method="none")
    pass

if __name__ == "__main__":

    main()
    
    