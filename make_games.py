from board import Board
import numpy as np
from utils import rb, player_sign

board_size = 5
num_games = 10
b = Board(board_size)

positions = np.zeros((num_games, board_size + 1, board_size + 1, 2))
positions [:, 1:, 0, 0] = 1
positions [:, 0, 1:, 1] = 1
winners = np.zeros(num_games)


def make_game():
    b.refresh()
    p = list(np.random.permutation(board_size * board_size))
    for i, move in enumerate(p):
        wm = b.winning_move(rb[i % 2])
        if wm:
            b.update(rb[i % 2], wm)
            moves = [b.index_to_point(m) for m in p[:i] + [wm]]
            break
        b.update(rb[i % 2], move)
    return (moves[::2], moves[1::2], b.winner)


for i in range(num_games):
    red_moves, blue_moves, winner = make_game()
    for row, column in red_moves:
        positions[i, row + 1, column + 1, 0] = 1
    for row, column in blue_moves:
        positions[i, row + 1, column + 1, 1] = 1
    winners[i] = player_sign[winner]

for i in range(num_games):
    print(positions[i, :, :, 0])
    print(positions[i, :, :, 1])
    print(winners[i])
