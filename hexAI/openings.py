if __name__ == "__main__":
    from .config import board_size
    from .board_utils import Player
    from .model import get_main_model
    from .position_utils import ArrayBoard
    from .minimax_player import minimax_move
    import yaml

    board = ArrayBoard(board_size)

    model = get_main_model()

    opening_win_logits = dict()

    for move in board.all_points:
        valid_moves = list(board.all_points)
        valid_moves.remove(move)
        opening_win_logits[str(move)] = float(-minimax_move(board, Player.BLUE, model, valid_moves)[1])
        print(move, opening_win_logits[(str(move))])

    with open('data/opening_win_logits.yaml', 'w') as f:
        yaml.dump(opening_win_logits, f)
