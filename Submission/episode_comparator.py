import importlib
import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt
from TicTacToe import *

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
               0.5: {'wins': 0, 'losses': 0, 'draws': 0},
               1: {'wins': 0, 'losses': 0, 'draws': 0}}

    random_seeds = random.sample(range(1000, 10000), 20)

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
    base_model_name = "model6"
    episode_intervals = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    model_names = [f"{base_model_name}_episode_{i}.h5" for i in episode_intervals]
    model_names.append(f"{base_model_name}.h5")  # Add the final model

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
    with open(f'results_{model_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Smartness', 'Wins', 'Losses', 'Draws'])
        for model_name, results in all_results:
            for smartness, stats in results.items():
                writer.writerow([model_name, smartness, stats['wins'], stats['losses'], stats['draws']])

    # Generate bar charts
    for smartness in [0, 0.5, 1]:
        wins = []
        losses = []
        draws = []
        episodes = []
        epi=0
        for model_name, results in all_results:
            if model_name.endswith('.h5'):
                episodes.append(epi)
                epi+=1000
            else:
                episodes.append(int(model_name.split('_')[-1].split('.')[0]))
            wins.append(results[smartness]['wins'])
            losses.append(results[smartness]['losses'])
            draws.append(results[smartness]['draws'])

        bar_width = 0.25
        r1 = np.arange(len(episodes))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        plt.figure(figsize=(12, 8))
        plt.bar(r1, wins, color='b', width=bar_width, edgecolor='grey', label='Wins')
        plt.bar(r2, losses, color='r', width=bar_width, edgecolor='grey', label='Losses')
        plt.bar(r3, draws, color='g', width=bar_width, edgecolor='grey', label='Draws')

        plt.xlabel('Episodes', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(episodes))], episodes)
        plt.title(f'Performance over Episodes (Smartness: {smartness})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'performance_smartness_{smartness}.png')
        plt.show()
