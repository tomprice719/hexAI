from board import Board
import numpy as np
from utils import rb, player_sign

board_size = 5
num_games = 50000

positions = np.zeros((num_games, 2, board_size + 1, board_size + 1))
positions [:, 0, 1:, 0] = 1
positions [:, 1, 0, 1:] = 1
winning_move_array = np.zeros((num_games, board_size, board_size))

b = Board(board_size)

def make_game():
    b.refresh()
    p = list(np.random.permutation(board_size * board_size))
    for i, move in enumerate(p):
        wm = b.winning_moves(rb[i % 2])
        if wm:
            moves = [b.index_to_point(m) for m in p[:i]]
            if i % 2 == 0:
                return (moves[::2], moves[1::2], wm, p[:i])
            if i % 2 == 1:
                moves = [(b, a) for a, b in moves]
                wm = [(b, a) for a, b in wm]
                return (moves[1::2], moves[::2], wm, p[:i])
        b.update(rb[i % 2], move)
    print(b)

for i in range(num_games):
    if i % 1000 == 0:
        print(i)
    red_moves, blue_moves, winning_moves, p = make_game()
    for row, column in red_moves:
        positions[i, 0, row + 1, column + 1] = 1
    for row, column in blue_moves:
        positions[i, 1, row + 1, column + 1] = 1
    for row, column in winning_moves:
        winning_move_array[i, row, column] = 1

    # print(positions[i, 0])
    # print(positions[i, 1])
    # print(labels[i])
    # print(len(p))
    # print("----------------------------------------")

np.savez("training_data2.npz", positions = positions, winning_moves = winning_move_array)