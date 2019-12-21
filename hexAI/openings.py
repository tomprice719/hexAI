"""
Use 2-ply minimax to compute win probabilities for each opening move.
Used to determine when to swap (see https://en.wikipedia.org/wiki/Pie_rule).
"""

if __name__ == "__main__":
    from .config import board_size
    from .board_utils import Player, all_points
    from .model import get_main_model
    from .model_input import ArrayBoard
    from .minimax_player import minimax_move
    import yaml

    model = get_main_model()

    opening_win_logits = dict()

    for move in all_points(board_size):
        board = ArrayBoard(board_size)
        board.update(Player.RED, move)
        valid_moves = list(all_points(board_size))
        valid_moves.remove(move)
        opening_win_logits[str(move)] = float(-minimax_move(board, Player.BLUE, model, valid_moves)[1])
        print("move: %s win logit: %s" % (move, opening_win_logits[(str(move))]))

    with open('data/opening_win_logits.yaml', 'w') as f:
        yaml.dump(opening_win_logits, f)
