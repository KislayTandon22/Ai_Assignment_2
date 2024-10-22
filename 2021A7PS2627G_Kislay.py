import sys
import random
from Ai_Assignment_2.TicTacToe import *
"""
You may import additional, commonly used libraries that are widely installed.
Please do not request the installation of new libraries to run your program.
"""

class PlayerSQN:
    def __init__(self):
        """
        Initializes the PlayerSQN class.
        """
        pass

    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.

        Parameters:
        state (list): A list representing the current state of the TicTacToe board.

        Returns:
        int: The position (0-8) where Player 2 wants to make a move.
        """
        # In your final submission, PlayerSQN must be controlled by an SQN. Use an epsilon-greedy action selection policy.
        # In this implementation, PlayerSQN is controlled by terminal input.
        print(f"Current state: {state}")
        action = int(input("Player 2 (You), enter your move (1-9): ")) - 1
        return action


def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
#    random.seed(42)
    playerSQN = PlayerSQN()
    game = TicTacToe(smartMovePlayer1,playerSQN)
    game.play_game()
    
    # Get and print the reward at the end of the episode
    reward = game.get_reward()
    print(f"Reward for Player 2 (You): {reward}")
    
if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0<=smartMovePlayer1<=1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1Probability>")
        print("Example: python 2020A7PS0001.py 0.5")
        print("There is an error. Probability must lie between 0 and 1.")
        sys.exit(1)
    
    main(smartMovePlayer1)