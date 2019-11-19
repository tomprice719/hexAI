from board import Board
import numpy as np
from utils import rb, neighbour_difference, opposite_player

board_size = 5
num_games = 10000

board = Board(board_size)

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


def make_game(starting_moves=(), index=None):
    if index is not None and index % 1000 == 0:
        print("Creating game", index)
    board.refresh()
    random_moves = list(np.random.permutation(board_size * board_size))
    already_played_set = set()
    already_played_list = []
    bridge_saving_moves = None
    for i in range(board_size * board_size):
        if i < len(starting_moves):
            next_move = starting_moves[i]
        else:
            # If there is a move that wins immediately, play it and return
            wm = [board.point_to_index(p) for p in board.winning_moves(rb[i % 2])]
            if wm:
                already_played_list.append(np.random.choice(wm))
                already_played_list = [board.index_to_point(move) for move in already_played_list]
                return already_played_list, i % 2
            # Otherwise, prevent an immediate win by the opponent, or save a bridge, or play a random move, in that priority.
            next_move = None
            wm = [board.point_to_index(p) for p in board.winning_moves(rb[(i + 1) % 2])]
            if wm:
                next_move = np.random.choice(wm)
            if next_move is None and bridge_saving_moves:
                next_move = np.random.choice(bridge_saving_moves)
            while next_move == None:
                candidate_move = random_moves.pop()
                if candidate_move not in already_played_set:
                    next_move = candidate_move
        board.update(rb[i % 2], next_move)
        already_played_set.add(next_move)
        already_played_list.append(next_move)
        bridge_saving_moves = list(get_bridge_saving_moves(board, rb[i % 2], next_move))
        # print(b)
    print(board)  # should never get here, printing board might give useful debugging information if we do


def add_training_data(moves, winner, positions_array, winners_array):
    p, p_swapped = np.zeros_like(positions_array[0, 1:, 1:]), np.zeros_like(positions_array[0, 1:, 1:])
    for i, move in enumerate(moves):
        a, b = move
        # Update temporary boards
        p[a, b, i % 2] = 1
        p_swapped[b, a, (i + 1) % 2] = 1
        # update training data, possibly swapping colours so that current player is always red
        positions_array[i, 1:, 1:] = (p, p_swapped)[i % 2]
        winners_array[i] = winner == i % 2


games = [make_game(index=i) for i in range(num_games)]
total_moves = sum(len(moves) for moves, winner in games)

positions_array = np.zeros((total_moves, board_size + 1, board_size + 1, 2))
winners_array = np.zeros(total_moves)

positions_array[:, 1:, 0, 0] = 1
positions_array[:, 0, 1:, 1] = 1

total_moves_counter = 0

while games:
    if len(games) % 1000 == 0:
        print(len(games), "more games to process.")
    moves, winner = games.pop()
    add_training_data(moves, winner,
                      positions_array[total_moves_counter: total_moves_counter + len(moves)],
                      winners_array[total_moves_counter: total_moves_counter + len(moves)])
    #total_moves_counter += len(moves)
    total_moves_counter += len(moves)

# for i in range(100):
#     print(positions_array[i, :, :, 0])
#     print(positions_array[i, :, :, 1])
#     print(winners_array[i])
#     print("--------------------------------------------")

np.savez("training_data4.npz",
         positions=positions_array,
         winners=winners_array)
