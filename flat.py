from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextOrderId = None

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextOrderId = orderId

    def position(self, account, contract: Contract, position: float, avgCost: float):
        """Called for each open position—send opposite MKT order to flatten."""
        if position == 0:
            return  # nothing to do

        # Determine side to close
        action = "SELL" if position > 0 else "BUY"
        qty = abs(position)

        # Build market order
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = qty
        order.tif = "DAY"  # or GTC, as desired

        # Place the closing order
        self.placeOrder(self.nextOrderId, contract, order)
        print(f"Placing {action} {qty} of {contract.symbol} to flatten.") 

    def positionEnd(self):
        """All positions have been received."""
        print("All positions processed, disconnecting.")
        self.disconnect()


if __name__ == "__main__":
    app = IBApp()
    app.connect("127.0.0.1", 7497, clientId=123)
    # Step 1: cancel any open orders (optional)
    app.reqGlobalCancel()
    # Step 2: pull positions and flatten them in the callbacks
    app.reqPositions()
    app.run()
