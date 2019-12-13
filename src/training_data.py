import numpy as np
from board_utils import Player
from position_utils import create_position, update_position, initialize_model_input, fill_model_input


def _add_game(model_input, winners, move_numbers, moves, winner, num_initial_moves, starting_index):
    position = create_position()

    for i, (move, annotation) in enumerate(moves):
        update_position(position, Player(i % 2), move)

        # update training data
        j = i - num_initial_moves
        if j >= 0:
            fill_model_input(model_input, position, Player(i % 2), starting_index + j)
            winners[starting_index + j] = winner == i % 2
            move_numbers[starting_index + j] = i + 1


def make_training_data(games, num_initial_moves, filename=None):
    total_moves = sum(len(moves[num_initial_moves:]) for moves, winner, swapped in games)

    model_input = initialize_model_input(total_moves)

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
        np.savez("../data/%s" % filename,
                 winners=winners,
                 move_numbers=move_numbers,
                 **model_input)

    return model_input, winners, move_numbers
