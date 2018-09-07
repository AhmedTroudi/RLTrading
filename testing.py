from broker import Broker
from keras.models import load_model
from functions import getStockDataVector,getState,formatPrice,plotdata
import sys

# parameters: python testing.py {stock} {model_name} (python 3.6)

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
state_size = model.layers[0].input.shape.as_list()[1]
broker = Broker(state_size, True, model_name)
data = getStockDataVector(stock_name)
data_size = len(data) - 1
batch_size = 20
gain = 0
broker.trades_list = []
reward = 0

state = getState(data, 0, state_size + 1)

for t in range(data_size):

    action = broker.act(state)

    if action == 1 and broker.portfolio >= data[t]:  # buy
        broker.trades_list.append(data[t])
        broker.portfolio = broker.portfolio - data[t]
        print("Bought: " + formatPrice(data[t]) + "| Portfolio Value: " + formatPrice(broker.portfolio))
        
        reward = data[t]-data[t+1]
    
    elif action == 2 and len(broker.trades_list) > 0:  # sell
        buying_price = broker.trades_list.pop(0)
        
        reward = data[t] - buying_price
        
        gain += data[t] - buying_price
        broker.portfolio = broker.portfolio + data[t]
        print("Sold: " + formatPrice(data[t]) + " | Financial Gain: " + formatPrice(
            data[t] - buying_price) + "| Portfolio Value: " + formatPrice(broker.portfolio))

    else:
        print("Hold| Portfolio Value: " + formatPrice(broker.portfolio))
        reward = 0
    
    done = True if t == data_size - 1 else False
    next_state = getState(data, t + 1, state_size + 1)
    broker.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("_________________________________")
        print("Selling what's left...")  # if some trades are left in inventory
        #plotdata(stock_name)

        while len(broker.trades_list) > 0:
            buying_price = broker.trades_list.pop(0)
            broker.portfolio = broker.portfolio + data[t]
            gain += data[t] - buying_price
            print("Sell: " + formatPrice(data[t]) + " | Financial Gain: " + formatPrice(
                data[t] - buying_price) + "| Portfolio Value: " + formatPrice(broker.portfolio) + "| Inventory Size:" + str(
                len(broker.trades_list)))
        print("_______________________________________________")
        print(stock_name + " Overall Gain: \n" + formatPrice(gain))
