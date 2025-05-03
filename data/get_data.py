from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import csv
from datetime import datetime

# Define a class to handle the IB API connection and data callbacks
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # List to store historical data
        self.data_received = threading.Event()  # Event to signal when data is received
        self.symbol = None # Store the symbol for filename
        self.durationStr = None # Store the duration string
        self.barSizeSetting = None # Store the bar size

    def historicalData(self, reqId, bar):
        """Callback function to handle incoming historical data"""
        date_str_from_ib = bar.date
        try:
            # Split the date string by space
            parts = date_str_from_ib.split()
            date_time_str = " ".join(parts[:2]) # Take first two parts: date and time
            timezone_str = parts[2] if len(parts) > 2 else None # Get timezone if available

            # Parse the date and time string (without timezone for now)
            date_obj = datetime.strptime(date_time_str, "%Y%m%d %H:%M:%S")

            formatted_date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") # Format to YYYY-MM-DD HH:MM:SS
            self.data.append({
                'date': formatted_date_str, # Use the formatted date string
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        except ValueError as e:
            print(f"Error parsing date string: {date_str_from_ib}. Error: {e}")
            print(f"Parts of the date string: {parts}") # Debugging: print the parts
            date_obj = None
            print(f"Skipping bar due to date parsing error: Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}, Volume: {bar.volume}, WAP: {bar.wap}, BarCount: {bar.barCount}")


    def historicalDataEnd(self, reqId, start, end):
        """Callback when all historical data has been received"""
        print(f"Historical data received for reqId {reqId}")
        # Use the stored durationStr and barSizeSetting for file naming
        duration_part = self.durationStr.replace(' ', '') if self.durationStr else 'unknownDur'
        bar_size_part = self.barSizeSetting.replace(' ', '') if self.barSizeSetting else 'unknownBar'
        if self.symbol:
            file_path = f'{self.symbol}_{bar_size_part}_bar_{duration_part}_historical_data.csv'
        else:
            file_path = 'historical_data.csv' # default if symbol is not set
        self.write_to_csv(file_path)
        self.data_received.set()

    def write_to_csv(self, file_path):
        """Write the historical data to a CSV file"""
        if not self.data:
            print("No data to write to CSV.")
            return
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['date', 'open', 'high', 'low', 'close', 'volume']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        print(f"Data successfully written to {file_path}")

# Function to create a stock contract
def create_stock_contract(symbol):
    """Create a contract for the specified stock symbol"""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"  # Stock
    contract.exchange = "SMART"  # Smart routing
    contract.currency = "USD"  # US Dollar
    return contract

# Function to run the API message loop
def run_loop(app):
    app.run()

# Main function to execute the program
def main():
    # Initialize the IB API client
    app = IBapi()
    app.connect("127.0.0.1", 7497, 123)  # Adjust host, port, clientId as needed

    # Start the API message loop in a separate thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    # Wait briefly for the connection to establish
    time.sleep(1)

    symbols = ['SPY'] # Get data for all symbols

    for ticker_symbol in symbols:
        print(f"Fetching historical data for {ticker_symbol}...")
        # Store the ticker symbol in the IBapi instance
        app.symbol = ticker_symbol

        # Clear previous data
        app.data = []
        app.data_received.clear()

        # Create a contract for the desired stock
        contract = create_stock_contract(ticker_symbol)

        durationStr = "20 Y"
        barSizeSetting = "1 hour"

        app.durationStr = durationStr
        app.barSizeSetting = barSizeSetting

        # Request historical data
        app.reqHistoricalData(
            reqId=1,                    # int: unique identifier for this request
            contract=contract,          # IBApi.Contract: the instrument to fetch (e.g., Stock, Future, Forex)
            endDateTime="",             # str: "YYYYMMDD HH:MM:SS [TZ]" or "" for current server time
            durationStr=durationStr,    # str: "<n> <unit>" where unit ∈ {S, D, W, M, Y}
            barSizeSetting=barSizeSetting,    # str: bar size
            whatToShow="TRADES",        # str: data type
            useRTH=1,                   # int: 1 = only regular trading hours; 0 = all available hours
            formatDate=1,               # int: 1 = date as string (yyyyMMdd[ HH:mm:ss]); 2 = epoch time
            keepUpToDate=False,         # bool: False = one-shot request; True = subscribe to updates (bars ≥5 secs)
            chartOptions=[]             # list[TagValue]: optional customization, usually empty [] or None
        )


        # Wait for the data to be received (timeout after 300 seconds)
        app.data_received.wait(timeout=300)
        if not app.data_received.is_set():
            print(f"Timeout waiting for data for {ticker_symbol}. No CSV file was written.")
        else:
            print(f"Historical data CSV file created for {ticker_symbol}.")
        time.sleep(2) # Small delay between requests


    # Disconnect from the API after all symbols are processed
    app.disconnect()
    print("Data fetching and CSV creation completed for all symbols.")

# Entry point of the program
if __name__ == "__main__":
    main()