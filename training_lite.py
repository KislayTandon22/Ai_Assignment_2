import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='linear'))

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

replay_buffer = deque(maxlen=10000)

np.random.seed(42)
for _ in range(2000):
    state = np.random.rand(9)
    action = np.random.randint(0, 9)
    reward = np.random.rand()
    next_state = np.random.rand(9)
    done = np.random.choice([True, False])
    replay_buffer.append((state, action, reward, next_state, done))

BATCH_SIZE = 32
GAMMA = 0.99
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    for step in range(len(replay_buffer) // BATCH_SIZE):
        mini_batch = random.sample(replay_buffer, BATCH_SIZE)

        states = np.zeros((BATCH_SIZE, 9))
        q_values_batch = np.zeros((BATCH_SIZE, 9))

        for j, (state, action, reward, next_state, done) in enumerate(mini_batch):
            q_values = model.predict(state[np.newaxis, :], verbose=0)
            q_next = model.predict(next_state[np.newaxis, :], verbose=0)

            if done:
                q_target = reward
            else:
                q_target = reward + GAMMA * np.max(q_next[0])

            q_values[0, action] = q_target

            states[j] = state
            q_values_batch[j] = q_values

        model.fit(states, q_values_batch, epochs=1, verbose=0)

print("Training completed.")

model.save("2021A7PS2627G_lite.h5")

test_input = np.random.rand(1, 9)
predicted_q_values = model.predict(test_input, verbose=0)
print("Predicted Q-values for the test input:", predicted_q_values)
