import os
import pandas as pd
import yfinance as yf
import shutil
import contextlib

# Configurations
offset = 4000
limit = 2000
period = '6mo'  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

# Download all NASDAQ traded symbols
url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
data = pd.read_csv(url, sep='|')
data_clean = data[data['Test Issue'] == 'N']
symbols = data_clean['NASDAQ Symbol'].tolist()
print(f"Total number of symbols traded = {len(symbols)}")

# Download Historical Data
limit = limit if limit else len(symbols)
end = min(offset + limit, len(symbols))
is_valid = [False] * len(symbols)

with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        for i in range(offset, end):
            symbol = symbols[i]
            data = yf.download(symbol, period=period)
            if len(data.index) == 0:
                continue

            is_valid[i] = True
            data.to_csv(f"hist/{symbol}.csv")

print(f"Total number of valid symbols downloaded = {sum(is_valid)}")

# Save Metadata for Valid Symbols
valid_data = data_clean[is_valid]
valid_data.to_csv('symbols_valid_meta.csv', index=False)

# Separate ETFs and Stocks
etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()


def move_symbols(symbols, destination):
    for symbol in symbols:
        filename = f"{symbol}.csv"
        shutil.move(os.path.join('hist', filename), os.path.join(destination, filename))


# Move files to respective directories
move_symbols(etfs, "etfs")
move_symbols(stocks, "stocks")

# Clean up
if not os.listdir('hist'):
    os.rmdir('hist')

print("Data successfully organized into 'stocks' and 'etfs' directories.")