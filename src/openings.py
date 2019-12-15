from config import board_size
from board_utils import Board, Player
from model import get_main_model
from position_utils import create_position, update_position
from minimax_player import minimax_move
import yaml

board = Board(board_size)

model = get_main_model()

opening_win_logits = dict()

for move in board.all_points:
    valid_moves = list(board.all_points)
    valid_moves.remove(move)
    position = create_position()
    update_position(position, Player.RED, move)
    opening_win_logits[str(move)] = -minimax_move(position, Player.BLUE, model, valid_moves)[1]
    print(move, opening_win_logits[(str(move))])

with open('../data/opening_win_logits.yaml', 'w') as f:
    yaml.dump(opening_win_logits, f)
