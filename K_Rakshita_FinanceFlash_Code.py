#!/usr/bin/env python
# coding: utf-8

# In[27]:


# !pip install pandas_datareader ta


# In[28]:


# Import necessary libraries
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from ta.trend import cci
import matplotlib.pyplot as plt


# ### Scrape the EUR/INR currency data from Yahoo Finance 

# In[29]:


def get_currency_data(start_date, end_date, currency_pair='EURINR'):
    try:
        # Fetch data using yfinance
        currency_data = yf.download(f'{currency_pair}=X', start=start_date, end=end_date)

        # Reset the index
        currency_data.reset_index(inplace=True)

        return currency_data

    except Exception as e:
        print(f"Error in getting currency data: {e}")
        return None


start_date = '2023-01-01'
end_date = '2024-02-16'
currency_data = get_currency_data(start_date, end_date)
currency_data = currency_data.dropna()

# Display the first few rows of the data
print(currency_data.head())


# ## Moving Average

# ### Calculate and Generate

# In[30]:


def calculate_and_generate_MA_signals(data, short_window=1, long_window=5):
    try:
        # Calculate one-day Moving Average
        sma_short = SMAIndicator(data['Close'], window=short_window)
        data['1day_MA'] = sma_short.sma_indicator()

        # Calculate one-week Moving Average
        sma_long = SMAIndicator(data['Close'], window=long_window)
        data['1week_MA'] = sma_long.sma_indicator()

        # Drop rows with NaN values in the newly calculated columns
        data = data.dropna(subset=['1day_MA', '1week_MA'])

        # Generate signals based on crossover
        data['Signal'] = 0.0
        data.loc[data['1day_MA'] > data['1week_MA'], 'Signal'] = 1.0  # Buy Signal
        data.loc[data['1day_MA'] < data['1week_MA'], 'Signal'] = -1.0  # Sell Signal
        
        # Create a new column 'Signal_Label' based on numerical 'Signal'
        data['Signal_Label'] = data['Signal'].map({1.0: 'BUY', -1.0: 'SELL', 0.0: 'NEUTRAL'})

        return data

    except Exception as e:
        print(f"Error in calculating and generating signals: {e}")
        return None

    
MA_Signals = calculate_and_generate_MA_signals(currency_data.copy(), short_window=1, long_window=5)

# Display the first few rows of the data with signals
if MA_Signals is not None:
    print(MA_Signals[['Date', 'Close', '1day_MA', '1week_MA', 'Signal', 'Signal_Label']].head())
    
# Save the specified columns of the new DataFrame to an Excel file
MA_Signals[['Date', 'Close', '1day_MA', '1week_MA', 'Signal', 'Signal_Label']].to_excel("F:\Project Files\Trading-Technical Analysis\MA_Signals.xlsx", index=False)


# ### Visualize

# In[31]:


def visualize_ma_signals(data, ma_one_day_col, ma_one_week_col):
    try:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data[ma_one_day_col], label='One-Day MA', linewidth=2)
        plt.plot(data['Date'], data[ma_one_week_col], label='One-Week MA', linewidth=2)
        
        # Scatter plot buy signals (green triangle)
        plt.scatter(data['Date'][data['Signal'] == 1], 
                    data[ma_one_day_col][data['Signal'] == 1], 
                    marker='^', color='g', label='Buy Signal')

        # Scatter plot sell signals (red triangle)
        plt.scatter(data['Date'][data['Signal'] == -1], 
                    data[ma_one_day_col][data['Signal'] == -1], 
                    marker='v', color='r', label='Sell Signal')

        plt.title('Buy/Sell Signals based on MA Crossovers')
        plt.xlabel('Date')
        plt.ylabel('MA Values')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error in visualizing signals: {e}")

# Visualize signals
visualize_ma_signals(MA_Signals, '1day_MA', '1week_MA')


# ## CCI (Commodity Channel Index)

# ### Calculate and Generate

# In[32]:


import pandas as pd

def calculate_cci_and_signals(data, cci_window=5, overbought_threshold=100, oversold_threshold=-100):
    try:
        # Calculate typical price
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3

        # Calculate Simple Moving Average (SMA) of Typical Price
        data['SMA'] = data['Typical_Price'].rolling(window=cci_window).mean()

        # Calculate Mean Absolute Deviation
        data['Mean_Deviation'] = abs(data['Typical_Price'] - data['SMA']).rolling(window=cci_window).mean()

        # Calculate Commodity Channel Index (CCI)
        data['CCI'] = (data['Typical_Price'] - data['SMA']) / (0.015 * data['Mean_Deviation'])

        # Drop rows with NaN values in the newly calculated columns
        data = data.dropna(subset=['CCI'])
        
        data['Buy_Signal'] = (data['CCI'] < oversold_threshold) & (data['CCI'].shift(1) >= oversold_threshold)
        data['Sell_Signal'] = (data['CCI'] > overbought_threshold) & (data['CCI'].shift(1) <= overbought_threshold)

        return data

    except Exception as e:
        print(f"Error in calculating CCI and generating signals: {e}")
        return None

    
CCI_Signals = calculate_cci_and_signals(currency_data.copy())

# Display the first few rows of the data with signals
if CCI_Signals is not None:
    print(CCI_Signals.head())

# Save the specified columns of the new DataFrame to an Excel file
CCI_Signals[['Date', 'Typical_Price', 'SMA', 'Mean_Deviation', 'CCI', 'Buy_Signal', 'Sell_Signal']].to_excel("F:\Project Files\Trading-Technical Analysis\CCI_Signals.xlsx", index=False)


# ### Visualize

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt

def plot_cci(data):
    try:
        # Check if 'CCI' column is present in the DataFrame
        if 'CCI' not in data.columns:
            raise ValueError("The 'CCI' column is not present in the DataFrame.")

        # Plot CCI
        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data['CCI'], color='blue', label='CCI')

        # Plot overbought and oversold thresholds
        plt.axhline(y=100, color='red', linestyle='--', label='Overbought Threshold')
        plt.axhline(y=-100, color='green', linestyle='--', label='Oversold Threshold')

        # Plot buy signals
        plt.plot(data[data['Buy_Signal'] == True]['Date'],
                 data['CCI'][data['Buy_Signal'] == True],
                 '^', markersize=10, color='g', lw=0, label='Buy Signal')

        # Plot sell signals
        plt.plot(data[data['Sell_Signal'] == True]['Date'],
                 data['CCI'][data['Sell_Signal'] == True],
                 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        plt.title('Commodity Channel Index (CCI)')
        plt.xlabel('Date')
        plt.ylabel('CCI')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error in plotting CCI: {e}")


plot_cci(CCI_Signals.copy()) 


# ##  Bollinger Band

# ### Calculate and Generate

# In[34]:


def calculate_and_generate_bb_signals(data, window_size=5, num_std=2):
    try:
        # Calculate Bollinger Bands
        data['Rolling_Mean'] = data['Close'].rolling(window=window_size).mean()
        data['Upper_Band_One_Week'] = data['Rolling_Mean'] + (num_std * data['Close'].rolling(window=window_size).std())
        data['Lower_Band_One_Week'] = data['Rolling_Mean'] - (num_std * data['Close'].rolling(window=window_size).std())

        # Drop rows with NaN values in the newly calculated columns
        data = data.dropna(subset=['Rolling_Mean', 'Upper_Band_One_Week', 'Lower_Band_One_Week'])

        # Create a copy of the DataFrame to avoid modifying the original data
        signals = data[['Date', 'Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week']].copy()

        # Create a new column 'Signal' and initialize with zeros
        signals['Signal'] = 0.0    
        signals.loc[signals['Close'] < signals['Lower_Band_One_Week'], 'Signal'] = 1.0
        signals.loc[signals['Close'] > signals['Upper_Band_One_Week'], 'Signal'] = -1.0
        signals['Signal_Label'] = signals['Signal'].map({1.0: 'BUY', -1.0: 'SELL', 0.0: 'NEUTRAL'})

        return signals

    except Exception as e:
        print(f"Error in generating signals: {e}")
        return None

    
BB_Signals = calculate_and_generate_bb_signals(currency_data.copy())

# Display the first few rows of the data with signals
if BB_Signals is not None:
    print(BB_Signals[['Date','Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week', 'Signal', 'Signal_Label']].head())

# Save the new DataFrame to an Excel file
BB_Signals[['Date','Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week', 'Signal', 'Signal_Label']].to_excel("F:\Project Files\Trading-Technical Analysis\BB_Signals.xlsx", index=False)


# ### Visualize

# In[35]:


import matplotlib.pyplot as plt

def visualize_bb_signals(data, upper_band_col, lower_band_col):
    try:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
        plt.plot(data['Date'], data[upper_band_col], label='Upper Bollinger Band', linestyle='--')
        plt.plot(data['Date'], data[lower_band_col], label='Lower Bollinger Band', linestyle='--')

        # Scatter plot buy signals (green upward arrow)
        buy_signals = data[data['Signal'] == 1]
        plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

        # Scatter plot sell signals (red downward arrow)
        sell_signals = data[data['Signal'] == -1]
        plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

        plt.title('Bollinger Bands and Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error in visualizing signals: {e}")

# Visualize signals for Bollinger Bands
visualize_bb_signals(BB_Signals, 'Upper_Band_One_Week', 'Lower_Band_One_Week')


# ## **Dynamic Thresholds:** Instead of fixed upper and lower bands, dynamically adjusting the threshold based on market volatility or historical price movements.

# ### Calculate and Generate

# In[36]:


def calculate_and_generate_dynamic_bb(data, window_size=5, num_std=2, volatility_multiplier=1.0):
    try:
        # Calculate dynamic thresholds based on volatility
        volatility = data['Close'].pct_change().rolling(window=window_size).std()
        upper_threshold = data['Close'] + volatility * num_std * volatility_multiplier
        lower_threshold = data['Close'] - volatility * num_std * volatility_multiplier

        # Create columns for dynamic upper and lower bands
        data['Dynamic_Upper_Band'] = upper_threshold
        data['Dynamic_Lower_Band'] = lower_threshold
        
        # Drop rows with NaN values in the newly calculated columns
        data = data.dropna(subset=['Dynamic_Upper_Band', 'Dynamic_Lower_Band'])

        # Make trading decisions based on dynamic Bollinger Bands
        def make_dynamic_bollinger_bands_decision(row):
            try:
                upper_bound = row['Dynamic_Upper_Band']
                lower_bound = row['Dynamic_Lower_Band']

                if row['Close'] > upper_bound:
                    return 'SELL'  # Consider selling as price is above dynamic upper bound
                elif row['Close'] < lower_bound:
                    return 'BUY'   # Consider buying as price is below dynamic lower bound
                else:
                    return 'NEUTRAL'
            except Exception as e:
                print(f"Error in making Dynamic Bollinger Bands decision: {e}")
                return 'NEUTRAL'

        # Apply the dynamic Bollinger Bands trading decision function
        data['Dynamic_BB_Signals'] = data.apply(make_dynamic_bollinger_bands_decision, axis=1)

        return data

    except Exception as e:
        print(f"Error in calculating and making Dynamic Bollinger Bands decision: {e}")
        return None

# Create a new DataFrame to store the results
dynamic_bb_Signal = calculate_and_generate_dynamic_bb(currency_data.copy(), num_std=2, volatility_multiplier=1.5)

# Display the first few rows of the new data with dynamic Bollinger Bands trading decisions
print(dynamic_bb_Signal[['Date', 'Close', 'Dynamic_Upper_Band', 'Dynamic_Lower_Band', 'Dynamic_BB_Signals']].head(5))

# Save the new DataFrame to an Excel file
dynamic_bb_Signal[['Date', 'Close', 'Dynamic_Upper_Band', 'Dynamic_Lower_Band', 'Dynamic_BB_Signals']].to_excel("F:\Project Files\Trading-Technical Analysis\Dynamic_BB_Signals.xlsx", index=False)


# ### Visualize

# In[37]:


def visualize_dynamic_bb_signals(data, upper_band_col, lower_band_col, signal_col):
    try:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
        plt.plot(data['Date'], data[upper_band_col], label='Dynamic Upper Bollinger Band', linestyle='--')
        plt.plot(data['Date'], data[lower_band_col], label='Dynamic Lower Bollinger Band', linestyle='--')

        # Scatter plot buy signals (green upward arrow)
        buy_signals = data[data[signal_col] == 'BUY']
        plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

        # Scatter plot sell signals (red downward arrow)
        sell_signals = data[data[signal_col] == 'SELL']
        plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

        plt.title('Dynamic Bollinger Bands and Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error in visualizing signals: {e}")

# Visualize signals for Dynamic Bollinger Bands
visualize_dynamic_bb_signals(dynamic_bb_Signal, 'Dynamic_Upper_Band', 'Dynamic_Lower_Band', 'Dynamic_BB_Signals')


# ## Adjust the Number of Standard Deviations (num_std): Experiment with different values for the number of standard deviations used in calculating the Bollinger Bands. Use 1.5 as standard deviation.

# ### Calculate and Generate

# In[38]:


def calculate_and_generate_bb_signals_std(data, window_size=5, num_std=1.5):
    try:
        # Calculate Bollinger Bands
        data['Rolling_Mean'] = data['Close'].rolling(window=window_size).mean()
        data['Upper_Band_One_Week'] = data['Rolling_Mean'] + (num_std * data['Close'].rolling(window=window_size).std())
        data['Lower_Band_One_Week'] = data['Rolling_Mean'] - (num_std * data['Close'].rolling(window=window_size).std())

        # Drop rows with NaN values in the newly calculated columns
        data = data.dropna(subset=['Rolling_Mean', 'Upper_Band_One_Week', 'Lower_Band_One_Week'])

        # Create a copy of the DataFrame to avoid modifying the original data
        signals = data[['Date','Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week']].copy()

        # Create a new column 'Signal' and initialize with zeros
        signals['Signal'] = 0.0    
        signals.loc[signals['Close'] < signals['Lower_Band_One_Week'], 'Signal'] = 1.0
        signals.loc[signals['Close'] > signals['Upper_Band_One_Week'], 'Signal'] = -1.0
        signals['Signal_Label'] = signals['Signal'].map({1.0: 'BUY', -1.0: 'SELL', 0.0: 'NEUTRAL'})

        return signals

    except Exception as e:
        print(f"Error in generating signals: {e}")
        return None


BB_Signals_STD = calculate_and_generate_bb_signals_std(currency_data.copy())

# Display the first few rows of the data with signals
if BB_Signals_STD is not None:
    print(BB_Signals_STD[['Date', 'Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week', 'Signal', 'Signal_Label']].head(40))

# Save the new DataFrame to an Excel file
BB_Signals_STD[['Date', 'Close', 'Upper_Band_One_Week', 'Lower_Band_One_Week', 'Signal', 'Signal_Label']].to_excel("F:\Project Files\Trading-Technical Analysis\BB_Signals_STD.xlsx", index=False)


# ### Visualize

# In[39]:


def visualize_bb_std_signals(data, upper_band_col, lower_band_col):
    try:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Close Price', linewidth=2)
        plt.plot(data['Date'], data[upper_band_col], label='Upper Bollinger Band', linestyle='--')
        plt.plot(data['Date'], data[lower_band_col], label='Lower Bollinger Band', linestyle='--')

        # Scatter plot buy signals (green upward arrow)
        buy_signals = data[data['Signal'] == 1]
        plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

        # Scatter plot sell signals (red downward arrow)
        sell_signals = data[data['Signal'] == -1]
        plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

        plt.title('Bollinger Bands and Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error in visualizing signals: {e}")

# Visualize signals for Bollinger Bands
visualize_bb_std_signals(BB_Signals_STD, 'Upper_Band_One_Week', 'Lower_Band_One_Week')

