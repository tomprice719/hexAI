import itertools
from utils import board_size, input_names
import numpy as np


def transform_coordinates(point, player_perspective, flip):
    a, b = point
    a1, b1 = (a, b) if player_perspective == 0 else (b, a)
    return (board_size - a1, board_size - b1) if flip is True else (a1 + 1, b1 + 1)


def minimax_move(current_positions, current_player, model, valid_moves):
    def _get_position(player_perspective, flipped):
        return current_positions[(player_perspective, flipped)]

    hypothetical_positions = dict((input_names[k],
                                   np.zeros([len(valid_moves) * (len(valid_moves) - 1),
                                             board_size + 1,
                                             board_size + 1,
                                             2],
                                            dtype="float32"))
                                  for k in itertools.product((0, 1), (False, True)))

    # initialize positions array
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        hypothetical_positions[input_names[((current_player + player_perspective + 1) % 2, flipped)]] = \
            _get_position(player_perspective, flipped)
    # add hypothetical moves
    move_pairs = [(move1, move2) for move1, move2 in itertools.product(valid_moves, valid_moves) if move1 != move2]
    for i, (move1, move2) in enumerate(move_pairs):
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = transform_coordinates(move1, player_perspective, flipped)
            a2, b2 = transform_coordinates(move2, player_perspective, flipped)
            hypothetical_positions[input_names[((current_player + player_perspective + 1) % 2, flipped)]] \
                [i, a1, b1, (current_player + player_perspective) % 2] = 1
            hypothetical_positions[input_names[((current_player + player_perspective + 1) % 2, flipped)]] \
                [i, a2, b2, (current_player + player_perspective + 1) % 2] = 1

    # infer

    win_logits = model.predict(hypothetical_positions)

    # get best move via minimax

    maximums = []
    for i in range(len(valid_moves)):
        maximums.append(max(win_logits[:len(valid_moves) - 1]))
        win_logits = win_logits[len(valid_moves) - 1:]
    assert len(win_logits) == 0

    best_move = valid_moves[min(range(len(valid_moves)), key=lambda x: maximums[x])]

    # update valid_moves, current position

    valid_moves.remove(best_move)

    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        a, b = transform_coordinates(best_move, player_perspective, flipped)
        _get_position(player_perspective, flipped)[a, b, (current_player + player_perspective) % 2] = 1

    return best_move
