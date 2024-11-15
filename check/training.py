# training.py
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from TicTacToe import TicTacToe

def create_model():
    """Creates and returns the SQN model with specified architecture."""
    model = Sequential([
        Dense(64, input_dim=9, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def train_sqn(episodes=5000, model_file='2021A7PS2627G_MODEL.h5'):
    """
    Trains the SQN model for Tic-Tac-Toe.
    
    Args:
        episodes: Number of training episodes
        model_file: Path to save the trained model
    """
    model = create_model()
    replay_buffer = deque(maxlen=10000)
    
    # Training hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    
    # Training statistics
    wins = draws = losses = 0
    smart_move_prob = 0.0  # Start with random opponent
    
    def preprocess_state(state):
        """Convert board state from [0,1,2] to [-1,0,1] format"""
        processed = np.array(state).copy()
        processed[processed == 2] = -1
        return processed
    
    def select_action(state, valid_moves):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.choice(valid_moves)
        
        state = preprocess_state(state)
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid moves
        for i in range(9):
            if i not in valid_moves:
                q_values[i] = float('-inf')
        return np.argmax(q_values)
    
    def train_on_batch():
        """Train model on a batch from replay buffer"""
        if len(replay_buffer) < batch_size:
            return
        
        batch = random.sample(replay_buffer, batch_size)
        states = np.zeros((batch_size, 9))
        targets = np.zeros((batch_size, 9))
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            proc_state = preprocess_state(state)
            target = model.predict(proc_state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                proc_next_state = preprocess_state(next_state)
                next_valid_moves = [i for i, val in enumerate(next_state) if val == 0]
                next_q_values = model.predict(proc_next_state.reshape(1, -1), verbose=0)[0]
                next_valid_q_values = [next_q_values[i] for i in next_valid_moves]
                target[action] = reward + gamma * max(next_valid_q_values)
            
            states[i] = proc_state
            targets[i] = target
        
        model.fit(states, targets, epochs=1, verbose=0)
    
    print("Starting training...")
    for episode in range(episodes):
        # Increase difficulty every 1000 episodes
        if episode > 0 and episode % 1000 == 0:
            smart_move_prob = min(0.8, smart_move_prob + 0.2)
            print(f"\nIncreasing opponent difficulty to {smart_move_prob}")
        
        game = TicTacToe(smart_move_prob)
        state = game.board.copy()
        done = False
        
        while not done:
            valid_moves = [i for i, val in enumerate(state) if val == 0]
            
            if not valid_moves:
                break
                
            # SQN's turn (Player 2)
            if state.count(1) == state.count(2):
                action = select_action(state, valid_moves)
                old_state = state.copy()
                
                game.make_move(action, 2)
                new_state = game.board.copy()
                reward = game.get_reward()
                done = game.is_full() or game.current_winner is not None
                
                replay_buffer.append((old_state, action, reward, new_state, done))
                state = new_state
                
                if len(replay_buffer) >= batch_size:
                    train_on_batch()
            
            # Player 1's turn
            else:
                game.player1_move()
                state = game.board.copy()
                done = game.is_full() or game.current_winner is not None
        
        # Update statistics
        if game.current_winner == 2:
            wins += 1
        elif game.current_winner == 1:
            losses += 1
        else:
            draws += 1
        
        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            win_rate = wins / 100
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Win Rate: {win_rate:.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            wins = draws = losses = 0
    
    model.save(model_file)
    print(f"\nTraining completed. Model saved as {model_file}")
    return model

if __name__ == "__main__":
    train_sqn()