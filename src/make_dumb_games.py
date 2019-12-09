from board_utils import Board, neighbour_difference, Player, opposite_player
import numpy as np
from utils import board_size


def get_bridge_saving_moves(board, player, move):
    def get_colour(point):
        a1, b1 = point
        if b1 < 0 or b1 >= board_size or board.has_hex(Player.RED, point):
            return Player.RED
        if a1 < 0 or a1 >= board_size or board.has_hex(Player.Blue, point):
            return Player.Blue
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


def make_game(board, starting_moves=(), index=None):
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
            wm = [board.point_to_index(p) for p in board.winning_moves(Player(i % 2))]
            if wm:
                already_played_list.append(np.random.choice(wm))
                already_played_list = [(board.index_to_point(move), None) for move in already_played_list]
                return already_played_list, i % 2, None
            # Otherwise, prevent an immediate win by the opponent
            # or save a bridge, or play a random move, in that priority.
            next_move = None
            wm = [board.point_to_index(p) for p in board.winning_moves(Player((i + 1) % 2))]
            if wm:
                next_move = np.random.choice(wm)
            if next_move is None and bridge_saving_moves:
                next_move = np.random.choice(bridge_saving_moves)
            while next_move is None:
                candidate_move = random_moves.pop()
                if candidate_move not in already_played_set:
                    next_move = candidate_move
        board.update(Player(i % 2), next_move)
        already_played_set.add(next_move)
        already_played_list.append(next_move)
        bridge_saving_moves = list(get_bridge_saving_moves(board, Player(i % 2), next_move))
    print(board)  # should never get here, printing board might give useful debugging information if we do
    assert False


def make_games(num_games):
    games = []
    board = Board(board_size)
    for i in range(num_games):
        if i % 1000 == 0:
            print("Making game %d" % i)
        games.append(make_game(board))
    return games
