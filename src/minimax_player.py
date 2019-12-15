from config import board_size
import numpy as np
from board_utils import Board, Player, opposite_player
from model import get_main_model
from position_utils import create_position, update_position, \
    initialize_model_input, fill_model_input, update_model_input
import itertools
import yaml


def minimax_move(position, current_player, model, valid_moves):
    model_input = initialize_model_input(len(valid_moves) * (len(valid_moves) - 1))

    fill_model_input(model_input, position, opposite_player(current_player), np.s_[:])

    moves = [(move1, move2) for move1, move2 in itertools.product(valid_moves, valid_moves) if move1 != move2]

    update_model_input(model_input,
                       [move1 for move1, move2 in moves],
                       current_player,
                       opposite_player(current_player),
                       np.s_[:])

    update_model_input(model_input,
                       [move2 for move1, move2 in moves],
                       opposite_player(current_player),
                       opposite_player(current_player),
                       np.s_[:])

    print([len(x) for x in model_input.values()])
    win_logits = model.predict(model_input)

    # get best move via minimax

    maximums = []
    for i in range(len(valid_moves)):
        maximums.append(max(win_logits[:len(valid_moves) - 1]))
        win_logits = win_logits[len(valid_moves) - 1:]
    assert len(win_logits) == 0

    print(min(maximums))

    best_response = valid_moves[min(range(len(valid_moves)), key=lambda x: maximums[x])]

    return best_response, -min(maximums)


def play_with_swap(model):
    board = Board(board_size)
    valid_moves = list(board.all_points)
    position = create_position()

    with open('../data/opening_win_logits.yaml', 'r') as f:
        opening_win_logits = yaml.load(f)

    may_swap = True
    current_player = Player.RED

    print(board)

    while board.winner is None:
        move = eval(input())
        valid_moves.remove(move)
        update_position(position, current_player, move)
        board.update(current_player, move)
        current_player = opposite_player(current_player)
        print(board)

        if board.winner is not None:
            break

        if may_swap is True and opening_win_logits[str(move)] > 0.5:
            print("SWAPPED. You are now blue. It is your turn again.")
        else:
            move, win_logit = minimax_move(position, current_player, model, valid_moves)
            valid_moves.remove(move)
            update_position(position, current_player, move)
            board.update(current_player, move)
            current_player = opposite_player(current_player)
            print(board)

        may_swap = False


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

        move, win_logit = minimax_move(position, Player.BLUE, model, valid_moves)
        valid_moves.remove(move)
        update_position(position, Player.BLUE, move)
        board.update(Player.BLUE, move)

        print(board)


play(get_main_model())
