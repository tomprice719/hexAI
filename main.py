from make_games5 import make_games, make_training_data
from board import Board
from utils import rb
import numpy as np
board_size = 5

def compare_models(model1, model2, half_num_games, num_initial_moves):
    red_wins = [winner for game, winner in
                make_games(model1, model2, half_num_games, num_initial_moves)].count(0) + \
               [winner for game, winner in
                make_games(model2, model1, half_num_games, num_initial_moves)].count(1)
    return red_wins / (half_num_games * 2)

def show_game(red_model, blue_model, num_initial_moves):
    game, winner = make_games
    b = Board(board_size)
    for i, move in enumerate(game):
        b.update(rb[i % 2], b.point_to_index(move))
        print(b)

def train(model, epoch_size, initial_games, new_games_per_epoch, num_iterations):
    positions, winners = make_training_data(model, initial_games, 4)

    for _ in range(num_iterations):
        model.fit(
            positions,
            winners,
            batch_size=32,
            epochs=1,
            shuffle=True,
            verbose=False
        )

        new_positions, new_winners = make_training_data(model, new_games_per_epoch, 4)
        positions = np.append(new_positions, positions)[:epoch_size]
        winners = np.append(new_winners, winners)[:epoch_size]
