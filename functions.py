import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getStockDataVector(key):
    data=pd.read_csv("data/" + key + ".csv")
    #print(data.head())
    #date=data["Date"].values.reshape(data["Date"].shape[0],1)
    op=data["Open"].values.reshape(data["Open"].shape[0],1)
    return op
# prints formatted price
def formatPrice(n):
    return ("-€" if n < 0 else "€") + str(abs(n))


# returns the sigmoid (used to create an array representing states)
def sigmoid(signal):
    # Prevent overflow.(classical sigmoid raises Math range overflow error as training progresses)
    signal = np.clip(signal, -500, 500)

    # Calculate activation signal
    signal = 1.0 / (1 + np.exp(-signal))

    return signal


def getState(data, t, n):
    state=[]
    #close = data.values
    #diff = np.diff(close)
    for i in range(t,t+n-1):
    
        state.append(sigmoid(data[:,0][i]-data[:,0][i-1])) #just normalizing values

    state=np.array(state)
    state=state.reshape(1,n-1)
    return state

