from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque

num_simulations = 100
num_epochs = 4
batch_size = 32
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
replay_buffer_size = 2000
replay_buffer = deque(maxlen=replay_buffer_size)

model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

def epsilon_greedy_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(9)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def replay_and_train():
    if len(replay_buffer) < batch_size:
        return
    minibatch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

for simulation in range(num_simulations):
    state = np.zeros((1, 9))
    done = False

    while not done:
        action = epsilon_greedy_action(state, epsilon)
        next_state = np.zeros((1, 9))
        reward = 0
        done = True
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        replay_and_train()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

for epoch in range(num_epochs):
    replay_and_train()

model.save('2021A7PS2627G_m.H5')
