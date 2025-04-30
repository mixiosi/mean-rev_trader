import threading
import time
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from datetime import datetime
import joblib
from queue import Queue  # Import the Queue class


# Helper function to round prices to a given tick size (default = 0.01 for US stocks)
def round_price(price, tick=0.01):
    return round(price / tick) * tick


# SignalFilter Class for Machine Learning Model
class SignalFilter:
    def __init__(self, model_path):
        """
        Initialize with the path to a pre-trained model.
        Loads the ML model from a file.
        """
        self.model = joblib.load(model_path)

    def predict(self, features):
        """
        Predict the probability of trade success.
        Uses the loaded ML model to predict the probability of a successful trade (class 1).
        """
        prob = self.model.predict_proba(features)[0, 1]  # Probability of class 1 (success)
        return prob


# Trading Application Class (API Connection and Callback Handling)
class TradingApp(EWrapper, EClient):
    def __init__(self, host='127.0.0.1', port=7497, client_id=618, model_path='trading_signal_model.joblib'):
        """
        Initializes the TradingApp.
        Connects to TWS, initializes data structures, and starts the connection thread.
        """
        EClient.__init__(self, self)
        self.data = {}  # Stores market data:  {reqId: [bars]}
        self.positions = {}  # Tracks open positions: {symbol: {'quantity': x, 'avg_cost': y}}
        self.account_values = {}  # Stores account summary: {tag: value}
        self.next_order_id = None  # Stores the next valid order ID
        self.order_id_queue = Queue()  # Use a thread-safe queue for order IDs
        self.connect(host, port, client_id)  # Connect to TWS
        self.data_engine = DataEngine(self, model_path)  # Pass model_path to DataEngine.  Handles data and signals.
        self.order_manager = OrderManager(self)  # Handles order placement.
        self.performance_analyzer = PerformanceAnalyzer()  # Tracks trade performance.
        self.conn_thread = threading.Thread(target=self.run, daemon=True)  # Start the API connection in a separate thread
        self.conn_thread.start()
        time.sleep(20)  # Wait for connection to establish.  Important.
        print(f"Next Order ID: {self.next_order_id}")


    def nextValidId(self, orderId: int):
        """
        Callback to set the initial order ID.
        Called by TWS when the connection is established.
        """
        self.next_order_id = orderId
        self.order_id_queue.put(orderId)  # Put the initial order ID in the queue

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print(f"Order Status - OrderId: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}")

        # Check if the parent order has been submitted/confirmed
        if orderId == self.order_manager.active_parent_order_id and status in (
                "Submitted", "PreSubmitted", "Filled"):  # Include filled.
            print(f"Parent order {orderId} confirmed or filled.")
            self.order_manager.order_confirmation_queue.put(True)  # Signal order confirmation
            # Don't reset here, we will reset after *all* children are confirmed/filled.
        # Check for child orders:
        elif parentId == self.order_manager.active_parent_order_id:
            # Store the status of child orders.
            self.order_manager.child_order_status[orderId] = status
            # Check if all child orders have been confirmed/filled
            # IMPORTANT CHANGE: Only Filled or Cancelled
            if all(s in ("Filled", "Cancelled") for s in self.order_manager.child_order_status.values()):
                print("All child orders confirmed or filled.")
                self.order_manager.active_parent_order_id = None  # Clear the active parent ID
                self.order_manager.child_order_status.clear()

    def get_next_order_id(self):
        """
        Retrieve and increment the next order ID.
        Retrieves a new order ID from the queue.  Blocks until an ID is available.
        """
        order_id = self.order_id_queue.get()  # Get from the queue.  Blocks if empty.
        self.next_order_id = order_id + 1  # Increment for the next request.
        self.order_id_queue.put(self.next_order_id)  # Put the incremented value back into the queue
        return order_id

    def historicalData(self, reqId: int, bar):
        """
        Callback to handle incoming historical data.
        Stores historical 5-minute bars.
        """
        self.data_engine.historical_data_handler(reqId, bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """
        Callback to process data after historical data is fully received.
        Called after all historical data for a request has been received.
        """
        self.data_engine.process_historical_data(reqId)

    def historicalDataUpdate(self, reqId: int, bar):
        """
        Callback to handle real-time updates to historical data.
        Handles new bars as they are received during a streaming historical data request.
        """
        # print(f"Historical Data Update received - ReqId: {reqId}, Date: {bar.date}")
        self.data_engine.process_new_bar(reqId, bar)

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """
        Callback to store account summary data.
        Handles account summary information (e.g., cash balance).
        """
        if reqId == 9001:  # Using a specific reqId for account summary
            self.account_values[tag] = value

    def accountSummaryEnd(self, reqId: int):
        """
        Callback to signal the end of account summary data.
        """
        pass

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """
        Callback to update position information.
        Called by TWS when the position of a symbol changes.
        """
        symbol = contract.symbol
        self.positions[symbol] = {'quantity': position, 'avg_cost': avgCost}


# Data Processing Engine with ML Signal Filtering
class DataEngine:
    def __init__(self, app, model_path):
        """
        Initializes the DataEngine.
        Stores a reference to the TradingApp, and initializes data structures.
        """
        self.app = app  # Reference to the main TradingApp instance
        self.historical_data = {}  # Stores historical data: {reqId: [bars]}
        self.symbols = {}  # Maps request IDs to symbols: {reqId: symbol}
        self.signal_filter = SignalFilter(model_path)  # Initialize SignalFilter with the model path

    def create_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD'):
        """
        Create a contract object for a given symbol.
        Simplifies the creation of Contract objects for different securities.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    def set_symbol(self, reqId, symbol):
        """
        Map request ID to symbol.
        Stores the symbol associated with a given historical data request ID.
        """
        self.symbols[reqId] = symbol

    def historical_data_handler(self, reqId, bar):
        """
        Store historical 5-minute bars.
        Appends a new historical bar to the list for the given request ID.
        """
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    def process_historical_data(self, reqId):
        """
        Process initial historical data (optional initialization).
        This method is called after all historical data has been received.
        """
        print(f"Historical data loaded for {self.symbols[reqId]}")

    def process_new_bar(self, reqId, bar):
        """
        Process new 5-minute bars and generate signals with ML filtering.
        This is the core logic for signal generation.
        """
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        df = pd.DataFrame(self.historical_data[reqId])  # Convert to DataFrame for easier calculations
        if len(df) < 20:  # Ensure enough data for indicators
            return

        # Calculate indicators
        df['sma20'] = df['close'].rolling(window=20).mean()  # 20-period simple moving average
        df['std20'] = df['close'].rolling(window=20).std()  # 20-period standard deviation
        df['upper_band'] = df['sma20'] + 2 * df['std20']  # Upper Bollinger Band
        df['lower_band'] = df['sma20'] - 2 * df['std20']  # Lower Bollinger Band
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)  # 14-period ATR
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)  # 14-period RSI

        last_bar = df.iloc[-1]  # Get the last row (most recent bar)
        mean_vol = df['volume'].tail(6).head(5).mean()  # Last 5 bars before current bar
        current_vol = last_bar['volume']
        vol_z = (current_vol - mean_vol) / mean_vol if mean_vol != 0 else 0  # Volume z-score

        # Calculate features for ML model
        if last_bar['std20'] == 0:
            deviation = 0  # Handle the case where std20 is zero
        else:
            deviation = (last_bar['close'] - last_bar['sma20']) / last_bar['std20']  # Deviation from SMA, in standard deviations
        atr_norm = last_bar['atr'] / last_bar['close']  # ATR normalized by price
        rsi = last_bar['rsi']
        features = np.array([[deviation, vol_z, atr_norm, rsi]])  # Create feature array

        # Signal Generation with ML Filtering
        if last_bar['close'] < last_bar['lower_band'] and vol_z > 0.15 and len(df) > 20:  # added len(df) > 20
            prob = self.signal_filter.predict(features)  # Predict probability of success
            if prob > 0.6:  # Only proceed if probability exceeds threshold
                self.app.order_manager.evaluate_long_entry(reqId, last_bar, last_bar['atr'])  # BUY
        elif last_bar['close'] > last_bar['upper_band'] and vol_z < -0.1 and len(df) > 20:  # added len(df) > 20
            prob = self.signal_filter.predict(features)
            if prob > 0.6:  # Only proceed if probability exceeds threshold
                self.app.order_manager.evaluate_short_entry(reqId, last_bar, last_bar['atr'])  # SELL


# Order Management System
class OrderManager:
    def __init__(self, app):
        self.app = app
        self.active_parent_order_id = None  # Track the active parent order ID
        self.order_confirmation_queue = Queue()  # Use a queue to signal order confirmation
        self.child_order_status = {} # Keep track of child order statuses.

    def evaluate_long_entry(self, reqId, bar, atr):
        """
        Evaluate and execute a long position entry.
        """
        symbol = self.app.data_engine.symbols[reqId]
        if symbol in self.app.positions and self.app.positions[symbol]['quantity'] != 0:
            return  # Avoid duplicate positions

        # Check if there's an active parent order
        if self.active_parent_order_id is not None:
            print(f"Waiting for confirmation of parent order {self.active_parent_order_id} before placing new order...")
            self.order_confirmation_queue.get()  # Block until confirmed.  This is KEY!

        entry_price = round_price(bar['close'])
        stop_loss = round_price(entry_price - 2.5 * atr)
        take_profit = round_price(entry_price + 5 * atr)
        quantity = self.calculate_position_size(symbol, atr)
        print(f"evaluate_long_entry - Quantity calculated: {quantity}")
        contract = self.app.data_engine.create_contract(symbol)

        parent_order_id = self.app.get_next_order_id()  # Get a *new* order ID here!
        self.active_parent_order_id = parent_order_id  # Store the new parent ID


        # Create ALL order objects here:
        parent = Order()
        parent.orderId = parent_order_id
        parent.action = 'BUY'
        parent.orderType = 'LMT'
        parent.lmtPrice = entry_price
        parent.totalQuantity = quantity
        parent.transmit = False  # Transmit the PARENT order
        parent.eTradeOnly = False
        parent.firmQuoteOnly = False
        
        profit = Order()
        profit.orderId = self.app.get_next_order_id()  # Get next ID for child
        profit.action = 'SELL'
        profit.orderType = 'LMT'
        profit.lmtPrice = take_profit
        profit.totalQuantity = quantity
        profit.parentId = parent_order_id  # Correct parent ID
        profit.transmit = False # Transmit the take-profit order.
        profit.eTradeOnly = False  # Make sure to set these for ALL orders.
        profit.firmQuoteOnly = False

        stop = Order()
        stop.orderId = self.app.get_next_order_id()  # Get next ID for child
        stop.action = 'SELL'
        stop.orderType = 'STP'
        stop.auxPrice = stop_loss
        stop.totalQuantity = quantity
        stop.parentId = parent_order_id  # Correct parent ID
        stop.transmit = True # Transmit the stop-loss order
        stop.eTradeOnly = False  # Make sure to set these for ALL orders.
        stop.firmQuoteOnly = False

        orders = [parent, profit, stop]  # create a list

        for order in orders:
            self.app.placeOrder(order.orderId, contract, order)
        print(f"Long entry placed for {symbol} at {entry_price}")

    def evaluate_short_entry(self, reqId, bar, atr):
        """
        Evaluate and execute a short position entry.
        """
        symbol = self.app.data_engine.symbols[reqId]
        if symbol in self.app.positions and self.app.positions[symbol]['quantity'] != 0:
            return  # Avoid duplicate positions

        # Check if there's an active parent order
        if self.active_parent_order_id is not None:
            print(
                f"Waiting for confirmation of parent order {self.active_parent_order_id} before placing new order...")
            self.order_confirmation_queue.get()  # Block until confirmed


        entry_price = round_price(bar['close'])
        stop_loss = round_price(entry_price + 2.5 * atr)
        take_profit = round_price(entry_price - 5 * atr)
        quantity = self.calculate_position_size(symbol, atr)
        print(f"evaluate_short_entry - Quantity calculated: {quantity}")
        contract = self.app.data_engine.create_contract(symbol)


        parent_order_id = self.app.get_next_order_id()  # Get a *new* order ID here!
        self.active_parent_order_id = parent_order_id  # Store the new parent ID

        # Create ALL order objects here:
        parent = Order()
        parent.orderId = parent_order_id
        parent.action = 'SELL'  # Correct action for short entry
        parent.orderType = 'LMT'
        parent.lmtPrice = entry_price
        parent.totalQuantity = quantity
        parent.transmit = False
        parent.eTradeOnly = False
        parent.firmQuoteOnly = False

        profit = Order()
        profit.orderId = self.app.get_next_order_id() # Get next ID for child
        profit.action = 'BUY'  # Opposite action for take profit on short
        profit.orderType = 'LMT'
        profit.lmtPrice = take_profit
        profit.totalQuantity = quantity
        profit.parentId = parent_order_id  # Correct parent ID
        profit.transmit = False
        profit.eTradeOnly = False
        profit.firmQuoteOnly = False

        stop = Order()
        stop.orderId = self.app.get_next_order_id()  # Get next ID for child
        stop.action = 'BUY'  # Opposite action for stop loss on short
        stop.orderType = 'STP'
        stop.auxPrice = stop_loss
        stop.totalQuantity = quantity
        stop.parentId = parent_order_id  # Correct parent ID
        stop.transmit = True
        stop.eTradeOnly = False
        stop.firmQuoteOnly = False

        orders = [parent, profit, stop]

        for order in orders:
            self.app.placeOrder(order.orderId, contract, order)
        print(f"Short entry placed for {symbol} at {entry_price}")

    def calculate_position_size(self, symbol, atr):
        """
        Calculate position size based on 1% account risk.
        This is a risk management calculation.
        """
        self.app.reqAccountSummary(9001, "All", "$LEDGER")  # Request account summary.
        time.sleep(1)  # Wait for account update.  Important.
        equity = float(self.app.account_values.get('TotalCashBalance', 100000))  # Default $100K
        risk_per_share = 2.5 * atr
        max_risk = equity * 0.01  # 1% risk per trade
        quantity = int(max_risk / risk_per_share)
        print(f"Debug position size calculation:")
        print(f"  Equity: {equity}")
        print(f"  ATR: {atr}")
        print(f"  Risk per share: {risk_per_share}")
        print(f"  Max Risk: {max_risk}")
        print(f"  Calculated Quantity (before min/max): {quantity}")
        quantity = max(1, min(quantity, 1000))  # Limit between 1 and 1000 shares
        print(f"  Final Quantity: {quantity}")
        return quantity


    def bracket_order(self, parent_order_id, action, quantity, stop_loss, take_profit, entry_price):
        """
        Create a bracket order. DEPRECATED - USE evaluate_long_entry/evaluate_short_entry instead
        """
        print("WARNING: bracket_order is deprecated. Use evaluate_long_entry or evaluate_short_entry.")
        return []


# Performance Tracking
class PerformanceAnalyzer:
    def __init__(self):
        """
        Initializes the PerformanceAnalyzer.
        Sets up the DataFrame to store trade logs.
        """
        self.trade_log = pd.DataFrame(columns=[  # Store trades
            'entry_time', 'exit_time', 'symbol', 'direction',
            'quantity', 'entry_price', 'exit_price', 'pnl'
        ])

    def log_trade(self, entry_time, exit_time, symbol, direction, quantity, entry_price, exit_price):
        """
        Log a completed trade.
        Adds a new trade record to the trade log.
        """
        pnl = (exit_price - entry_price) * quantity if direction == 'BUY' else (entry_price - exit_price) * quantity
        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl
        }])
        self.trade_log = pd.concat([self.trade_log, new_trade], ignore_index=True)

    def calculate_sharpe(self, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio of returns.
        A measure of risk-adjusted return.
        """
        returns = self.trade_log['pnl'].pct_change().dropna()
        if len(returns) == 0:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)

    def generate_report(self):
        """
        Generate a performance report.
        Currently, it plots the cumulative P&L.
        """
        if self.trade_log.empty:
            print("No trades to report.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(self.trade_log['exit_time']), self.trade_log['pnl'].cumsum())
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.show()


# Main Execution Loop
def main():
    """
    Main function to run the trading application.
    Creates the TradingApp, requests data, and keeps the program running.
    """
    app = TradingApp(model_path='trading_signal_model.joblib')  # Specify the path to your pre-trained model
    symbols = ['SPY', 'NVDA', 'RGTI', 'GOLD', 'QBTS', 'QUBT', 'TSLA', 'IONQ']  # Symbols to trade
    # Wait until next_order_id is received (connection established)
    timeout_start = time.time()
    timeout_seconds = 30  # Wait for up to 30 seconds for connection
    while app.next_order_id is None:
        time.sleep(1)
        if time.time() - timeout_start > timeout_seconds:
            print("Timeout waiting for connection to TWS. Aborting.")
            app.disconnect()
            return
        print("Waiting for connection...")
    print(f"Connection established. Next Order ID: {app.next_order_id}")

    # Request historical data and real-time updates
    for idx, symbol in enumerate(symbols):
        contract = app.data_engine.create_contract(symbol)
        app.data_engine.set_symbol(idx, symbol)
        app.reqHistoricalData(  # Request historical data.  This also starts the stream of real-time updates.
            reqId=idx,
            contract=contract,
            endDateTime='',  # Empty string for current time
            durationStr='20 D',  # 20 days of data
            barSizeSetting='5 mins',  # 5-minute bars
            whatToShow='TRADES',  # Get trade data
            useRTH=0,  # 0 for all data, 1 for Regular Trading Hours
            formatDate=1,  # Return dates as YYYYMMDD HH:MM:SS
            keepUpToDate=True,  # Request real-time updates
            chartOptions=[]
        )

    time.sleep(10)  # Additional delay to allow data to start coming in

    # Request position updates.  This will trigger the position() callback.
    app.reqPositions()

    # Keep the program running to receive events from TWS
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # Handle Ctrl+C
        app.performance_analyzer.generate_report()  # Generate report on exit
        app.disconnect()  # Disconnect from TWS
        print("Program terminated.")


if __name__ == '__main__':
    main()