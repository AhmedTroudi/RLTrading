import sys
from broker import Broker
from functions import getStockDataVector
from functions import getState
from functions import formatPrice

# parameters: <stock_name> <n> <epochs> (python 3.6)

stock_name, state_size, nb_epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
broker = Broker(state_size)  # n = how many days represent a state
data = getStockDataVector(stock_name)
data_size = len(data) - 1
batch_size = 20  # after this number of examples update weights


for e in range(nb_epochs + 1):
    print("Epoch Number:" + str(e) + "/" + str(nb_epochs))
    state = getState(data, 0, state_size + 1)
    print(state.shape)
    reward = 0
    gain = 0
    broker.trades_list = []
    broker.portfolio = 100000
    for t in range(data_size):
        action = broker.act(state)

        if action == 1 and broker.portfolio >= data[t]:    # buy
            broker.trades_list.append(data[t])
            broker.portfolio = broker.portfolio - data[t]
            print("Buy: " + formatPrice(data[t]) + "| Portfolio Value: " + formatPrice(
                broker.portfolio) + "| Inventory Size:" + str(len(broker.trades_list)))
            reward = 0

        elif action == 2 and len(broker.trades_list) > 0:  # sell
            buying_price = broker.trades_list.pop(0)
            broker.portfolio = broker.portfolio + data[t]
            reward = max(data[t] - buying_price, 0)
            gain += data[t] - buying_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(
                data[t] - buying_price) + "| Portfolio Value: " + formatPrice(broker.portfolio) + "| Inventory Size:" + str(
                len(broker.trades_list)))
        elif action == 0:                                            # hold
            print(
                "Hold | Portfolio Value: " + formatPrice(broker.portfolio) + "| Inventory Size:" + str(len(broker.trades_list)))
            reward = 0

        done = True if t == data_size - 1 else False
        next_state = getState(data, t + 1, state_size + 1)
        broker.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("_________________________________")
            print("Epoch {0} Profit \n:".format(str(e)) + formatPrice(gain))

        if len(broker.memory) > batch_size:
            broker.expReplay(batch_size)
            print("exp replay executed")

    if e % 10 == 0:
        broker.model.save("models/" + stock_name + "_2ep_" + str(e))