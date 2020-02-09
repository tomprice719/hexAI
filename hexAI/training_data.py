"""
The purpose of this module is the make_training_data function,
which makes the data needed to train a model, given a list of games.
"""

import numpy as np
from .board_utils import Player
from .model_input import new_array_position, update_array_position, new_model_input, fill_model_input


def _add_game(model_input, winners, move_numbers, moves, winner, num_initial_moves, starting_index):
    position = new_array_position()

    for i, (move, annotation) in enumerate(moves):
        update_array_position(position, Player(i % 2), move)

        # update training data
        j = i - num_initial_moves
        if j >= 0:
            fill_model_input(model_input, position, Player(i % 2), starting_index + j)
            winners[starting_index + j] = winner == Player(i % 2)
            move_numbers[starting_index + j] = i + 1


def make_training_data(games, num_initial_moves, filename=None):
    """
    Make the data needed to train a model.
    parameters:
        games: a list of games
        num_initial_moves: the length of the randomized phase at the beginning of each game.
        This section of the game will not be included in the training data.
        filename: if this parameter is provided, save the data to a file in the data folder,
        in addition to returning it.

        returns: (model_input, winners, move_numbers) tuple
        model_input can be used as the input of a model, and winners can be used as the target.
        move_numbers is the number of moves that have been played at each sample of training data,
        which can be used for weighing samples differently depending on the phase of the game.
    """
    total_moves = sum(len(moves[num_initial_moves:]) for moves, winner, swapped in games)

    model_input = new_model_input(total_moves)

    winners = np.zeros(total_moves, dtype="float32")
    move_numbers = np.zeros(total_moves, dtype="float32")

    total_moves_counter = 0

    while games:
        if len(games) % 1000 == 0:
            print(len(games), "more games to process.")
        moves, winner, swapped = games.pop()
        counter_diff = len(moves[num_initial_moves:])
        _add_game(model_input,
                  winners,
                  move_numbers,
                  moves,
                  winner,
                  num_initial_moves,
                  total_moves_counter)
        total_moves_counter += counter_diff

    if filename is not None:
        np.savez("data/%s" % filename,
                 winners=winners,
                 move_numbers=move_numbers,
                 **model_input)

    return model_input, winners, move_numbers
