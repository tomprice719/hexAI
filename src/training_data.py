from utils import input_names, board_size, initial_position
import itertools
import numpy as np


def add_training_data(moves, winner, num_initial_moves, positions, winners, slice_):
    temp_positions = dict(((player_perspective, flipped), np.copy(initial_position))
                          for player_perspective, flipped in itertools.product((0, 1), (False, True)))

    for i, (move, annotation) in enumerate(moves):
        # update temporary positions
        a, b = move
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (board_size - a1, board_size - b1) if flipped else (a1 + 1, b1 + 1)
            temp_positions[(player_perspective, flipped)][a2, b2, (i + player_perspective) % 2] = 1

        # update training data
        j = i - num_initial_moves
        if j >= 0:
            for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                positions[input_names[((player_perspective + i) % 2, flipped)]][slice_][j] = \
                    temp_positions[(player_perspective, flipped)]
            winners[j] = winner == i % 2


def make_training_data(games, num_initial_moves, filename=None):
    total_moves = sum(len(moves[num_initial_moves:]) for moves, winner, swapped in games)

    positions = dict((input_names[k],
                      np.zeros([total_moves,
                                board_size + 1,
                                board_size + 1,
                                2],
                               dtype="float32"))
                     for k in itertools.product((0, 1), (False, True)))

    winners = np.zeros(total_moves, dtype="float32")

    total_moves_counter = 0

    while games:
        if len(games) % 1000 == 0:
            print(len(games), "more games to process.")
        moves, winner, swapped = games.pop()
        counter_diff = len(moves[num_initial_moves:])
        slice_ = np.s_[total_moves_counter: total_moves_counter + counter_diff]
        add_training_data(moves, winner, num_initial_moves,
                          positions,
                          winners[slice_],
                          slice_)
        # total_moves_counter += len(moves)
        total_moves_counter += counter_diff

    if filename is not None:
        np.savez("../data/%s" % filename,
                 winners=winners,
                 **positions)

    return positions, winners
