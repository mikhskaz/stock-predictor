import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# List of stock tickers
hist_stocks = {'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA'}

# Create the output directory if it doesn't exist
output_dir = 'sarima_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Forecast horizon (number of days to predict)
forecast_days = 90

for stock in hist_stocks:
    try:
        # Read the CSV file
        filename = f'cleaned_{stock}.csv'
        df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')

        # Ensure the data is sorted by date
        df = df.sort_index()

        # Check if 'Close' column exists
        if 'Close' not in df.columns:
            print(f"'Close' column not found in {filename}")
            continue

        # Auto-tune SARIMA parameters
        auto_model = auto_arima(df['Close'], seasonal=True, m=5, d=1, D=1,
                                trace=False, error_action='ignore', suppress_warnings=True)
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"Best SARIMA order for {stock}: {order}, seasonal_order: {seasonal_order}")

        # Fit the SARIMA model
        model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)

        # Forecast future stock prices
        forecast = results.get_forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

        # Use summary_frame to get forecasted mean and confidence intervals
        forecast_df = forecast.summary_frame(alpha=0.05)
        forecast_df.index = forecast_index  # Align the forecast index
        forecast_df.rename(columns={'mean': 'Forecast', 'mean_ci_lower': 'Lower CI', 'mean_ci_upper': 'Upper CI'},
                           inplace=True)

        # Save the forecasted prices to a CSV file
        output_filename = os.path.join(output_dir, f'forecast_{stock}.csv')
        forecast_df.to_csv(output_filename, columns=['Forecast', 'Lower CI', 'Upper CI'])

        # Determine the start date for plotting (e.g., last 2 years)
        plot_start_date = df.index[-1] - pd.DateOffset(years=2)

        # Plot the historical and forecasted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'][df.index >= plot_start_date], label='Historical Close Prices')
        plt.plot(forecast_df['Forecast'], label='Forecasted Prices', color='red')
        plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3)
        plt.title(f'{stock} Stock Price Forecast (SARIMA)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plot_filename = os.path.join(output_dir, f'forecast_plot_{stock}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Forecast saved and plot generated for {stock}")

    except Exception as e:
        print(f"Error processing {stock}: {e}")
