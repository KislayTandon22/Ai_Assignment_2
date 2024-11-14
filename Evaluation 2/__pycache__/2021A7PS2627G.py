import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from TicTacToe import TicTacToe
import os

@register_keras_serializable()

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class PlayerSQN:
    def __init__(self,model="model21.h5"):
        """
        Initializes the PlayerSQN class and loads the pre-trained model.
        """
        try:
            custom_objects = {'mse': mse}
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir,model)
            self.model = load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error: Could not load model. {e}")
            sys.exit(1)
    
    def _preprocess_state(self, state):
        """
        Converts the state from [0,1,2] format to [-1,0,1] format for better learning.
        """
        processed_state = np.array(state).copy()
        processed_state[processed_state == 2] = -1
        return processed_state

    def move(self, state):
        """
        Selects the best move based on the current state.
        """
        processed_state = self._preprocess_state(state)
        valid_moves = [i for i, val in enumerate(state) if val == 0]
        
        q_values = self.model.predict(processed_state.reshape(1, -1), verbose=0)[0]
        # Mask invalid moves
        for i in range(len(q_values)):
            if i not in valid_moves:
                q_values[i] = float('-inf')
        return np.argmax(q_values)

def simulate_games(smartMovePlayer1, num_games=20):
    """
    Simulates multiple TicTacToe games and tracks the results.

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
    num_games: Number of games to simulate.
    """
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

if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0 <= smartMovePlayer1 <= 1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1Probability>")
        print("Example: python 2021A7PS2627G.py 0.5")
        print("Error: Probability must lie between 0 and 1.")
        sys.exit(1)
    
    simulate_games(smartMovePlayer1)
