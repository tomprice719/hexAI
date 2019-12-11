import itertools
from utils import board_size, input_names, initial_position
import numpy as np
from board_utils import Board, Player
from model import create_model


def transform_coordinates(point, player_perspective, flip):
    a, b = point
    a1, b1 = (a, b) if player_perspective == 0 else (b, a)
    return (board_size - a1, board_size - b1) if flip is True else (a1 + 1, b1 + 1)


def _get_position(positions, player_perspective, flipped):
    return positions[(player_perspective, flipped)]


def update_current_positions(positions, point, current_player):
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        a, b = transform_coordinates(point, player_perspective, flipped)
        _get_position(positions, player_perspective, flipped)[a, b, (current_player + player_perspective) % 2] = 1


def minimax_move(current_positions, current_player, model, valid_moves):
    hypothetical_positions = dict((input_names[k],
                                   np.zeros([len(valid_moves) * (len(valid_moves) - 1),
                                             board_size + 1,
                                             board_size + 1,
                                             2],
                                            dtype="float32"))
                                  for k in itertools.product((0, 1), (False, True)))

    # initialize positions array
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        hypothetical_positions[input_names[((current_player + player_perspective + 1) % 2, flipped)]][:] = \
            _get_position(current_positions, player_perspective, flipped)
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
    print([len(x) for x in hypothetical_positions.values()])
    win_logits = model.predict(hypothetical_positions)

    # get best move via minimax

    maximums = []
    for i in range(len(valid_moves)):
        maximums.append(max(win_logits[:len(valid_moves) - 1]))
        win_logits = win_logits[len(valid_moves) - 1:]
    assert len(win_logits) == 0

    print(min(maximums))

    best_move = valid_moves[min(range(len(valid_moves)), key=lambda x: maximums[x])]

    return best_move


def play(model):
    board = Board(board_size)
    valid_moves = list(board.all_points)
    current_positions = dict()
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        current_positions[(player_perspective, flipped)] = np.copy(initial_position)

    print(board)

    while board.winner is None:
        move = eval(input())
        valid_moves.remove(move)
        update_current_positions(current_positions, move, 0)
        board.update(Player(0), move)

        print(board)

        if board.winner is not None:
            break

        move = minimax_move(current_positions, 1, model, valid_moves)
        valid_moves.remove(move)
        update_current_positions(current_positions, move, 1)
        board.update(Player(1), move)

        print(board)


model = create_model(depth=18, breadth=40, learning_rate=0.0001)
model.load_weights('../data/better_model.h5')

play(model)
