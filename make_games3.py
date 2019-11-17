from board import Board
import numpy as np
from utils import rb, player_sign, neighbour_difference, opposite_player
import matplotlib.pyplot as plt

board_size = 5
num_games = 1

positions = np.zeros((num_games, board_size + 1, board_size + 1, 2))
positions[:, 1:, 0, 0] = 1
positions[:, 0, 1:, 1] = 1
winners = np.zeros(num_games)

b = Board(board_size)

def get_bridge_saving_moves(board, player, move):
    def get_colour(point):
        a, b = point
        if b < 0 or b >= board_size or board.has_hex("red", point):
            return "red"
        if a < 0 or a >= board_size or board.has_hex("blue", point):
            return "blue"
        return None
    a, b = board.index_to_point(move)
    neighbours = [(a + da, b + db) for da, db in neighbour_difference]
    neighbour_colours = [get_colour(point) for point in neighbours]
    opp = opposite_player(player)
    for i in range(6):
        prev_colour = neighbour_colours[(i - 1) % 6]
        colour = neighbour_colours[i % 6]
        next_colour = neighbour_colours[(i + 1) % 6]
        if colour is None and prev_colour == opp and next_colour == opp:
            yield board.point_to_index(neighbours[i % 6])


def make_game(starting_moves=()):
    b.refresh()
    random_moves = list(np.random.permutation(board_size * board_size))
    already_played_set = set()
    already_played_list = []
    bridge_saving_moves = None
    for i in range(board_size * board_size):
        if i < len(starting_moves):
            next_move = starting_moves[i]
        else:
            # If there is a move that wins immediately, play it and return
            wm = [b.point_to_index(p) for p in b.winning_moves(rb[i % 2])]
            if wm:
                already_played_list.append(np.random.choice(wm))
                already_played_list = [b.index_to_point(move) for move in already_played_list]
                return already_played_list[::2], already_played_list[1::2], i % 2
            # Otherwise, prevent an immediate win by the opponent, or save a bridge, or play a random move, in that priority.
            next_move = None
            wm = [b.point_to_index(p) for p in b.winning_moves(rb[(i + 1) % 2])]
            if wm:
                next_move = np.random.choice(wm)
            if next_move is None and bridge_saving_moves:
                next_move = np.random.choice(bridge_saving_moves)
            while next_move == None:
                candidate_move = random_moves.pop()
                if candidate_move not in already_played_set:
                    next_move = candidate_move
        b.update(rb[i % 2], next_move)
        already_played_set.add(next_move)
        already_played_list.append(next_move)
        bridge_saving_moves = list(get_bridge_saving_moves(b, rb[i % 2], next_move))
        # print(b)
    print(b)  # should never get here, printing board might give useful debugging information if we do


win_probs = np.array([sum(1 - winner
                          for red_moves, blue_moves, winner
                          in [make_game([starting_move]) for _ in range(100)])
                      for starting_move in range(board_size * board_size)]).reshape(((board_size, board_size)))

print(win_probs)

plt.imshow(win_probs)
plt.show()

# for i in range(num_games):
#     if i % 1000 == 0:
#         print(i)
#     red_moves, blue_moves, winner = make_game()
#     for row, column in red_moves:
#         positions[i, row + 1, column + 1, 0] = 1
#     for row, column in blue_moves:
#         positions[i, row + 1, column + 1, 1] = 1
#     winners[i] = winner
#     #winners[i] = randint(0, 1)
#
#     # print(positions[i, 0])
#     # print(positions[i, 1])
#     # print(labels[i])
#     # print(len(p))
#     # print("----------------------------------------")
#
# np.savez("training_data3.npz", positions=positions, winners=winners)