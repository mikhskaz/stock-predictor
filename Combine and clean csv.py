import os
import csv

# Directory containing the CSV files
input_dir = "hist"
output_file = "combined_last_lines.csv"

# Initialize a list to store the last rows
last_rows = []

# Iterate over all CSV files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)

        # Extract ticker from the file name (remove file extension)
        ticker = os.path.splitext(file)[0]

        # Read the file and get the last line
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Ensure there is at least one line (skip empty files)
        if lines:
            # Get the last non-header line
            last_line = lines[-1].strip()

            # Split the last line by commas and store the data along with the ticker
            data = last_line.split(',')

            # Ensure the line has exactly 7 values (Date, Adj Close, Close, High, Low, Open, Volume)
            if len(data) == 7:
                date = data[0]
                adj_close = data[1]
                close = data[2]
                high = data[3]
                low = data[4]
                open_price = data[5]
                volume = data[6]

                # Append the formatted row with the ticker
                last_rows.append([date, ticker, adj_close, close, high, low, open_price, volume])

# If there are any rows to write
if last_rows:
    # Write all collected rows to a new CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header
        writer.writerow(['Date', 'Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])

        # Write the last rows for each ticker
        writer.writerows(last_rows)

    print(f"Combined last lines saved to {output_file}")
else:
    print("No valid data found.")