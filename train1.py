import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from TicTacToe import TicTacToe

gamma = 0.95
learning_rate = 0.001
batch_size = 32
episodes = 5000

replay_buffer = deque(maxlen=2000)

model = Sequential([
    Dense(64, input_dim=9, activation='relu'),
    Dense(64, activation='relu'),
    Dense(9, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

def generate_random_state():
    env = TicTacToe(smartMovePlayer1=0.5)  # Player 1 makes random/smart moves
    board = [0] * 9
    num_moves = random.randint(1, 9)
    moves_made = 0
    player_turn = 1
    
    while moves_made < num_moves:
        valid_actions = [i for i in range(9) if board[i] == 0]
        pos = random.choice(valid_actions)
        board[pos] = player_turn
        env.board = board
        if env.check_winner(player_turn):
            break
        moves_made += 1
        player_turn = 3 - player_turn
    
    env.board = board
    return env, player_turn

def epsilon_greedy_action(state, env, epsilon=1.0):
    valid_actions = env.empty_positions()
    if not valid_actions:
        return None
    if np.random.rand() <= epsilon:
        return random.choice(valid_actions)
    q_values = model.predict(state, verbose=0)
    q_values_valid = [q_values[0][i] for i in valid_actions]
    return valid_actions[np.argmax(q_values_valid)]

def train_on_replay():
    if len(replay_buffer) < batch_size:
        return
    minibatch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.max(model.predict(next_state, verbose=0)[0])
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

valid_episodes = []
for _ in range(5000):
    env, player_turn = generate_random_state()
    valid_episodes.append((env, player_turn))

win, draw, loss = 0, 0, 0
for e, (env, current_player) in enumerate(valid_episodes):
    state = np.reshape(env.board, [1, 9])
    done = False
    epsilon = max(1.0 - (e / episodes), 0.01)
    
    while not done:
        if current_player == 2:  # Player 2 (Model)
            action = epsilon_greedy_action(state, env, epsilon)
            if action is None:
                break
            env.make_move(action, 2)
            reward = env.get_reward()  # Reward is only given when Player 2 (the model) moves
            done = env.is_full() or env.current_winner is not None
            next_state = np.reshape(env.board, [1, 9])
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
        else:  # Player 1 (Smart/Random)
            if not env.is_full() and env.current_winner is None:
                env.player1_move()
            done = env.is_full() or env.current_winner is not None
        
        # Calculate the result of the game once it's done
        if done:
            if reward == 1:
                loss += 1
            elif reward == -1:
                win += 1
            else:
                draw += 1

        current_player = 3 - current_player  # Switch player
    
    train_on_replay()
    
    print(f"Episode {e+1}/{episodes}, Win: {win}, Draw: {draw}, Loss: {loss}")

model.save('YourBITSid_MODEL.h5')
