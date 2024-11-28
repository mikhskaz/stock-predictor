import os
import csv

# List of important stocks to process
hist_stocks = {'AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA'}

# Directory containing the CSV files
input_dir = "popular_stocks"

# Iterate over each stock in hist_stocks
for stock in hist_stocks:
    file_name = f"{stock}.csv"
    file_path = os.path.join(input_dir, file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        cleaned_rows = []  # List to store the cleaned rows

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Identify and skip the irrelevant lines (Ticker, Date, etc.)
        # Start processing only from the line containing 'Date'
        start_index = 0
        for i, line in enumerate(lines):
            if 'Date' in line:  # We assume 'Date' is in the header row
                start_index = i
                break

        # Process all rows starting from the line containing 'Date'
        for line in lines[start_index + 1:]:  # Skip the 'Date' row itself
            cleaned_rows.append(line.strip().split(','))

        # Write the cleaned data to a new CSV file for the stock
        output_file = f"cleaned_{stock}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write the header row
            writer.writerow(['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])

            # Write all the rows
            writer.writerows(cleaned_rows)

        print(f"Cleaned data saved to {output_file}")

    else:
        print(f"File for {stock} not found.")
