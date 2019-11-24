from board import Board
import numpy as np
from utils import rb, player_index
from random import randint

board_size = 5
num_games = 50000

positions = np.zeros((num_games, board_size + 1, board_size + 1, 2))
positions[:, 1:, 0, 0] = 1
positions[:, 0, 1:, 1] = 1
winners = np.zeros(num_games)

b = Board(board_size)


# def make_game():
#     b.refresh()
#     p = list(np.random.permutation(board_size * board_size))
#     for i, move in enumerate(p):
#         wm = b.winning_move(rb[i % 2])
#         if wm is not None:
#             b.update(rb[i % 2], wm)
#             #print(b)
#             moves = [b.index_to_point(m) for m in p[:i] + [wm]]
#             return (moves[::2], moves[1::2], b.winner)
#         b.update(rb[i % 2], move)
#         #print(b)
#     b.refresh()
#     for i, move in enumerate(p):
#         b.update(rb[i % 2], move)
#         print(b.winning_move("red"))
#         print(b.winning_move("blue"))
#         print(b)
#         b.print_stuff()
#         print("----------------------------------------------")
#         print("----------------------------------------------")

def make_game():
    b.refresh()
    p = list(np.random.permutation(board_size * board_size))
    for i, move in enumerate(p):
        b.update(rb[i % 2], move)
        if i % 2 == 1 and b.winner != None:
            moves = [b.index_to_point(m) for m in p[:i + 1]]
            return moves[::2], moves[1::2], b.winner
    return make_game()


for i in range(num_games):
    if i % 1000 == 0:
        print(i)
    red_moves, blue_moves, winner = make_game()
    for row, column in red_moves:
        positions[i, row + 1, column + 1, 0] = 1
    for row, column in blue_moves:
        positions[i, row + 1, column + 1, 1] = 1
    winners[i] = player_index[winner]
    #winners[i] = randint(0, 1)

np.savez("training_data.npz", positions=positions, winners=winners)

# for i in range(num_games):
#     print(positions[i, 0, :, :])
#     print(positions[i, 1, :, :])
#     print(np.sum(positions[i, 0, :, :]))
#     print(np.sum(positions[i, 1, :, :]))
#     print("Winner: ", winners[i])
