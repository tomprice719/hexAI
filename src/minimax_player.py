from config import board_size
import numpy as np
from board_utils import Board, Player, opposite_player
from model import create_model
from position_utils import create_position, update_position, \
    initialize_model_input, fill_model_input, update_model_input


def minimax_move(position, current_player, model, valid_moves):
    model_input = initialize_model_input(len(valid_moves) * (len(valid_moves) - 1))

    fill_model_input(model_input, position, opposite_player(current_player), np.s_[:])

    for i, move in enumerate(valid_moves):
        responses = [response for response in valid_moves if response != move]
        slice_ = np.s_[i * len(responses): (i + 1) * len(responses)]
        update_model_input(model_input,
                           [move] * len(responses),
                           current_player,
                           opposite_player(current_player),
                           slice_)
        update_model_input(model_input,
                           responses,
                           opposite_player(current_player),
                           opposite_player(current_player),
                           slice_)

    print([len(x) for x in model_input.values()])
    win_logits = model.predict(model_input)

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
    position = create_position()

    print(board)

    while board.winner is None:
        move = eval(input())
        valid_moves.remove(move)
        update_position(position, Player.RED, move)
        board.update(Player.RED, move)

        print(board)

        if board.winner is not None:
            break

        move = minimax_move(position, Player.BLUE, model, valid_moves)
        valid_moves.remove(move)
        update_position(position, Player.BLUE, move)
        board.update(Player.BLUE, move)

        print(board)


model = create_model(depth=18, breadth=40, learning_rate=0.0001)
model.load_weights('../data/better_model.h5')

play(model)
