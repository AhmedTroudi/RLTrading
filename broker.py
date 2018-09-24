from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense,Dropout, Activation
from keras.layers.recurrent import LSTM

from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD,RMSprop

import numpy as np
import random
from collections import deque
from functions import sigmoid




class Broker:  # corresponds to the agent in RL framework
    def __init__(self, state_size, is_evaluated=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # hold, buy or sell
        self.memory = deque(maxlen=1000)
        self.trades_list = []
        self.portfolio = 100000  # budget with which the agent starts
        self.model_name = model_name
        self.is_evaluated = is_evaluated

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_evaluated else self.q_function()

    def q_function(self):  # returns the model
 
         model = Sequential()
         model.add(Dense(units=10, input_dim=self.state_size, activation="relu"))
         model.add(Dense(units=20, activation="relu"))
         model.add(Dense(units=8, activation="relu"))
         model.add(Dense(self.action_size, activation="relu"))
         model.compile(loss="mean_squared_logarithmic_error", optimizer=Adam(lr=0.002))
 

# =============================================================================
#           tsteps = 1
#           batch_size = 1
#           num_features = 1
#             
#           model = Sequential()
#           model.add(LSTM(5,
#                            input_shape=(1, num_features),
#                            return_sequences=True,
#                            stateful=False))
#           model.add(Dropout(0.5))
#             
#           model.add(LSTM(5,
#                            input_shape=(1, num_features),
#                            return_sequences=False,
#                            stateful=False))
#           model.add(Dropout(0.5))
#             
#           model.add(Dense(4, init='lecun_uniform'))
#           model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
#             
#           rms = RMSprop()
#           adam = Adam()
#           model.compile(loss='mse', optimizer=adam)
# =============================================================================
         return model

    def act(self, state):

        if self.is_evaluated is False and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):

        # mini_batch = random.sample(self.memory, batch_size)
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
            
                next_state=np.array(next_state)
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1,5),batch_size=1))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon equals \n",self.epsilon)
