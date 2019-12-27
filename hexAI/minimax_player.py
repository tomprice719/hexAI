from .config import board_size
import numpy as np
from .board_utils import Player, opposite_player, all_points
from .model import get_main_model
from .model_input import ArrayBoard, new_model_input, fill_model_input, update_model_input
import itertools
import yaml
from string import ascii_uppercase


def minimax_move(board, current_player, model, valid_moves, breadth=None):
    """
    Get the best move to play, determined with a partial 2-ply minimax.
    First, depth-1 win probabilities are computed,
    then depth-2 probabilities are computed for the best moves found by the previous step,
    then minimax is applied to find the best move overall.

    parameters:
        board: Board object representing the current state of the game.
        current_player: element of Player enum representing the current player.
        model: model that computes the win probabilities
        valid_moves: a list of all unoccupied points on the board
        breadth: the number of moves, at the first ply, to compute 2-ply minimax win probabilities for. If none,
        does full 2-ply minimax.
    """

    if breadth is None:
        breadth = len(valid_moves)
    else:
        breadth = min(breadth, len(valid_moves))

    model_input = new_model_input(len(valid_moves))
    fill_model_input(model_input, board.array_position, current_player, np.s_[:])
    update_model_input(model_input,
                       valid_moves,
                       current_player,
                       current_player,
                       np.s_[:])
    one_ply_logits = model.predict(model_input)

    if breadth > 0:
        # Prevents equal win probabilities in case the board position is symmetrical
        one_ply_logits += np.random.uniform(0.0, 0.00001, one_ply_logits.shape)

        minimax_move_indices, minimax_moves = zip(*[(i, move)
                                                    for i, move in enumerate(valid_moves)
                                                    if one_ply_logits[i] >= sorted(one_ply_logits)[-breadth]])
        assert len(minimax_moves) == breadth
        moves = [(move1, move2)
                 for move1, move2 in itertools.product(minimax_moves, valid_moves)
                 if move1 != move2]

        model_input = new_model_input(len(moves))
        fill_model_input(model_input, board.array_position, opposite_player(current_player), np.s_[:])
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
        two_ply_logits = model.predict(model_input)

        minimax_logits = []
        for i in range(len(minimax_moves)):
            minimax_logits.append(-max(two_ply_logits[:len(valid_moves) - 1]))
            two_ply_logits = two_ply_logits[len(valid_moves) - 1:]
        assert len(two_ply_logits) == 0
        assert len(minimax_logits) == len(minimax_moves)

        for j, i in enumerate(minimax_move_indices):
            one_ply_logits[i] = minimax_logits[j]

    best_response_index, best_response = max(enumerate(valid_moves), key=lambda x: one_ply_logits[x[0]])
    assert one_ply_logits[best_response_index] == max(one_ply_logits)

    return best_response, one_ply_logits[best_response_index]


def _get_move(valid_moves):
    while True:
        try:
            print("Your move: ", end="")
            input_ = input().strip().upper()
            if len(input_) < 2:
                raise ValueError
            a = ascii_uppercase.index(input_[0])
            b = int(input_[1:]) - 1
            if (a, b) not in valid_moves:
                raise ValueError
            return a, b
        except ValueError:
            print("Invalid move.")


def _get_breadth():
    while True:
        try:
            print("Search breadth: ", end="")
            breadth = int(input())
            if breadth < 0 or breadth > board_size ** 2:
                raise ValueError
            return breadth
        except ValueError:
            print("Invalid input.")


def play_auto(model, starting_move, breadth):
    """
    Have the AI play against itself, with a provided starting move and search breadth.
    """
    board = ArrayBoard(board_size)
    valid_moves = list(all_points(board_size))

    valid_moves.remove(starting_move)
    board.update(Player.RED, starting_move)

    print(board)

    while board.winner is None:

        move, win_logit = minimax_move(board, Player.BLUE, model, valid_moves)
        valid_moves.remove(move)
        board.update(Player.BLUE, move)

        print(board)

        if board.winner is not None:
            break

        move, win_logit = minimax_move(board, Player.RED, model, valid_moves, breadth)
        valid_moves.remove(move)
        board.update(Player.RED, move)

        print(board)

        if board.winner is not None:
            break


def play_with_swap(model):
    """
    Play as first player against the AI, using the pie rule.
    """
    board = ArrayBoard(board_size)
    valid_moves = list(all_points(board_size))

    with open('data/opening_win_logits.yaml', 'r') as f:
        opening_win_logits = yaml.load(f)

    may_swap = True
    current_player = Player.RED

    print("Please enter the search breadth, a whole number from 0 to %d." % board_size ** 2)
    print("With a larger search breadth, I will think longer and play better moves.")
    print("The recommended value is 10.")
    breadth = _get_breadth()

    print(board)

    print("It is your move. You are red.")
    print("Please enter your move coordinates, e.g. A1 to play in the top left corner.")
    print("We are playing with the pie rule, \
so if your first move is too good, I can choose to swap positions with you.")

    while board.winner is None:
        move = _get_move(valid_moves)
        valid_moves.remove(move)
        board.update(current_player, move)
        if board.winner == current_player:
            print("You win!")
            return
        current_player = opposite_player(current_player)
        print(board)

        if may_swap is True and opening_win_logits[str(move)] > 0:
            print("SWAPPED. You are now blue. It is your turn again.")
        else:
            print("Thinking...")
            move, win_logit = minimax_move(board, current_player, model, valid_moves, breadth)
            valid_moves.remove(move)
            board.update(current_player, move)
            current_player = opposite_player(current_player)
            print(board)

        may_swap = False

    print("I win!")


def play_without_swap(model):
    """
    Play as first player against the AI, with no pie rule.
    """
    board = ArrayBoard(board_size)
    valid_moves = list(all_points(board_size))

    print("Please enter the search breadth, a whole number between 0 and %d." % board_size ** 2)
    print("With a larger search breadth, I will think longer and play better moves.")
    breadth = _get_breadth()

    print(board)

    print("It is your move. You are red.")
    print("Please enter your move coordinates, e.g. A1 to play in the top left corner.")

    while board.winner is None:
        move = _get_move(valid_moves)
        valid_moves.remove(move)
        board.update(Player.RED, move)

        print(board)

        if board.winner == Player.RED:
            print("You win!")
            return

        print("Thinking...")
        move, win_logit = minimax_move(board, Player.BLUE, model, valid_moves, breadth)
        valid_moves.remove(move)
        board.update(Player.BLUE, move)

        print(board)
    print("I win!")


if __name__ == "__main__":
    play_with_swap(get_main_model())
