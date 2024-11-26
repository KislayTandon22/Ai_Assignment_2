import importlib
import numpy as np
import os
import csv
from TicTacToe import *
import random

def play_full_game(playerSQN, smartness):
    """Plays a full game of TicTacToe"""
    game = TicTacToe(smartness, playerSQN)
    game.play_game()
    reward = game.get_reward()
    return reward

def check_model(model_name):
    """Checks the submission"""
    # Dynamically import the module
    module_name = "2021A7PS2627G"  # Replace with the actual module name
    my_module = importlib.import_module(module_name)

    results = {0: {'wins': 0, 'losses': 0, 'draws': 0},
               0.8: {'wins': 0, 'losses': 0, 'draws': 0},
               1: {'wins': 0, 'losses': 0, 'draws': 0}}

    random_seeds = random.sample(range(1000, 10000), 1000)

    for seed in random_seeds:
        random.seed(seed)
        print(f"Testing with seed value: {seed}")
        playerSQN = my_module.PlayerSQN(model_name)
        for smartness in results.keys():
            game_reward = play_full_game(playerSQN, smartness)
            if game_reward == 1:
                results[smartness]['wins'] += 1
            elif game_reward == -1:
                results[smartness]['losses'] += 1
            else:
                results[smartness]['draws'] += 1
        del playerSQN

    return model_name, results

if __name__ == "__main__":
    model_names = ["2021A7PS2627G_MODEL.h5","model6_retrain_retrain_retrain_retrain_retrain.h5"]
    
    all_results = []

    # Check if all models exist
    for model_name in model_names:
        if not os.path.exists(model_name):
            print(f"Model file {model_name} does not exist.")
            continue
        
        print(f"Running on model: {model_name}")
        all_results.append(check_model(model_name))
    
    # Print all results at the end
    for model_name, results in all_results:
        print(f"Model: {model_name}")
        for smartness, stats in results.items():
            print(f"Smartness: {smartness}")
            print(f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")

    # Save results to CSV
    with open('results_model6.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Smartness', 'Wins', 'Losses', 'Draws'])
        for model_name, results in all_results:
            for smartness, stats in results.items():
                writer.writerow([model_name, smartness, stats['wins'], stats['losses'], stats['draws']])
