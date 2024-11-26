import sys
import numpy as np
import random
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from TicTacToe import TicTacToe
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class PlayerSQN:
    def __init__(self, model="2021A7PS2627G_MODEL.h5"):
        try:
            custom_objects = {'mse': mse}
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model)
            self.model = load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error: Could not load model. {e}")
            sys.exit(1)
    
    def _preprocess_state(self, state):
        processed_state = np.array(state).copy()
        processed_state[processed_state == 0] = -1
        processed_state[processed_state == 1] = 0
        processed_state[processed_state == 2] = 1
        return processed_state

    def move(self, state):
        processed_state = self._preprocess_state(state)
        valid_moves = [i for i, val in enumerate(state) if val == 0]
        
        q_values = self.model.predict(processed_state.reshape(1, -1), verbose=0)[0]
        for i in range(len(q_values)):
            if i not in valid_moves:
                q_values[i] = float('-inf')
        return np.argmax(q_values)

def simulate_games(smartMovePlayer1, num_games=20):
    playerSQN = PlayerSQN()
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games):
        game = TicTacToe(smartMovePlayer1, playerSQN)
        game.play_game()
        
        reward = game.get_reward()
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
    print(f"Results after {num_games} games:")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")

class SQNAgent:
    def __init__(self, state_size=9, action_size=9, gamma=0.95, model_path='model6.h5'):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 32
        self.replay_buffer = deque(maxlen=10000)
        self.model_path = model_path
        if os.path.exists(model_path):
            try:
                print(f'{model_path} exists. Loading the model.')
                self.model = load_model(model_path, compile=False)
                self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead.")
                self.model = self._build_model()
        else:
            print(f'{model_path} does not exist. Starting new training.')
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def process_state(self, state):
        processed_state = np.array(state).copy()
        processed_state[processed_state == 0] = -1
        processed_state[processed_state == 1] = 0
        processed_state[processed_state == 2] = 1
        return processed_state

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        processed_state = self.process_state(state)
        q_values = self.model.predict(processed_state.reshape(1, -1), verbose=0)[0]
        valid_q_values = [(q_values[action], action) for action in valid_actions]
        return max(valid_q_values, key=lambda x: x[0])[1]

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([self.process_state(exp[0]) for exp in mini_batch])
        next_states = np.array([self.process_state(exp[3]) for exp in mini_batch])
        
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        x = []
        y = []

        for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if done:
                target = reward
            else:
                next_valid_actions = [j for j, val in enumerate(next_state) if val == 0]
                if next_valid_actions:
                    max_next_q = max(next_q_values[i][action] for action in next_valid_actions)
                    target = reward + self.gamma * max_next_q
                else:
                    target = reward
            
            current_q = current_q_values[i].copy()
            current_q[action] = target
            x.append(self.process_state(state))
            y.append(current_q)

        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=1, verbose=0)

def train_agent(episodes=40000, model_path='2021A7PS2627G_MODEL.h5'):
    agent = SQNAgent(model_path=model_path)
    history = {'wins': 0, 'losses': 0, 'draws': 0}
    initial_epsilon = 1.0
    min_epsilon = 0.01
    decay_episodes = 1000
    smartness = 0
    for episode in range(episodes):
        if episode % 1000 == 0 and episode > 0:
            print("Resetting epsilon to 1.0")
        else:
            epsilon_decay = (initial_epsilon - min_epsilon) / decay_episodes
            agent.epsilon = max(min_epsilon, initial_epsilon - (episode % 1000) * epsilon_decay)
        
        smartness = min(1, episode / (episodes * 0.9))
        game = TicTacToe(smartMovePlayer1=smartness)
        state = np.array(game.board)
        
        game.player1_move()
        state = np.array(game.board)
        
        while True:
            valid_actions = game.empty_positions()
            if not valid_actions:
                history['draws'] += 1
                break
                
            action = agent.select_action(state, valid_actions)
            game.make_move(action, player=2)
            
            reward = 0
            if game.current_winner == 2:
                reward = 1.0
                history['wins'] += 1
                agent.store_experience(state, action, reward, game.board, True)
                break
            
            game.player1_move()
            if game.current_winner == 1:
                reward = -1.0
                history['losses'] += 1
                agent.store_experience(state, action, reward, game.board, True)
                break
            elif game.is_full():
                reward = min(-0.1, -0.5 * smartness)
                history['draws'] += 1
                agent.store_experience(state, action, reward, game.board, True)
                break
            
            agent.store_experience(state, action, reward, game.board, False)
            state = np.array(game.board)
        
        agent.train()
        
        if episode % 100 == 0:
            win_rate = history['wins'] / (episode + 1)
            print(f"Episode: {episode}, Win Rate: {win_rate:.2f}, Epsilon: {agent.epsilon:.3f}, smartmove: {smartness:.2f}")
            print(f"Wins: {history['wins']}, Losses: {history['losses']}, Draws: {history['draws']}")
        
        if episode % 1000 == 0:
            try:
                agent.model.save(f'model6_retrain_retrain_retrain_episode_{episode}.h5')
                print(f"Checkpoint saved at episode {episode}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    try:
        agent.model.save("2021A7PS2627G_MODEL.h5")
        print(f'Final model saved as 2021A7PS2627G_MODEL.h5')
    except Exception as e:
        print(f"Error saving final model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python <script_name> <smartMovePlayer1Probability or 'train'>")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == 'train':
        train_agent()
    else:
        try:
            smartMovePlayer1 = float(arg)
            assert 0 <= smartMovePlayer1 <= 1
            simulate_games(smartMovePlayer1)
        except ValueError:
            print("Error: Probability must lie between 0 and 1.")
            sys.exit(1)
