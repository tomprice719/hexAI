"""
Generate games according to a simple method that requires no trained model.
These games can be used to start the training process.
Moves are selected according to the following procedure:
1. If there is a move that will cause the current player to win immediately, play there,
or randomly choose such a move if there are multiple.
2. Otherwise, if there is a move that will cause the opponent to win immediately, play there,
or randomly choose such a move if there are multiple.
3. Otherwise, if there is a move that will save a threatened bridge, play there,
or randomly choose such a move if there are multiple.
4. Otherwise, randomly play an arbitrary legal move.

Games are represented as triples (moves, winner, swapped).
The winner is represented with an element of the Player enum.
As the pie rule is not used in this module, swapped is always None.

Moves are represented as a list of (point, annotation) tuples.
Since no annotations are implemented in this module, the annotation is always None.
"""

from .board_utils import Board, neighbour_difference, Player, opposite_player, all_points
from .config import board_size
from random import shuffle, choice


def _get_bridge_saving_moves(board, player, move):
    def get_colour(point):
        a1, b1 = point
        if b1 < 0 or b1 >= board_size or board.has_hex(Player.RED, point):
            return Player.RED
        if a1 < 0 or a1 >= board_size or board.has_hex(Player.BLUE, point):
            return Player.BLUE
        return None

    a, b = move
    neighbours = [(a + da, b + db) for da, db in neighbour_difference]
    neighbour_colours = [get_colour(point) for point in neighbours]
    opp = opposite_player(player)
    for i in range(6):
        prev_colour = neighbour_colours[(i - 1) % 6]
        colour = neighbour_colours[i]
        next_colour = neighbour_colours[(i + 1) % 6]
        if colour is None and prev_colour == opp and next_colour == opp:
            yield neighbours[i]


def make_game(starting_moves=()):
    """
    Generate a single game.
    You may provide a list of points to the starting_moves parameter
    to force the games to begin with that sequence of moves.
    See the module docstring for more information.
    """
    board = Board(board_size)
    random_moves = list(all_points(board_size))
    shuffle(random_moves)
    already_played_set = set()
    already_played_list = []
    bridge_saving_moves = None
    for i in range(board_size ** 2):
        if i < len(starting_moves):
            next_move = starting_moves[i]
        else:
            # If there is a move that wins immediately, play it and return
            wm = board.winning_moves(Player(i % 2))
            if wm:
                already_played_list.append(choice(wm))
                already_played_list = [(move, None) for move in already_played_list]
                return already_played_list, Player(i % 2), None
            # Otherwise, prevent an immediate win by the opponent if possible
            next_move = None
            wm = board.winning_moves(Player((i + 1) % 2))
            if wm:
                next_move = choice(wm)
            # Otherwise, save a bridge if possible
            if next_move is None and bridge_saving_moves:
                next_move = choice(bridge_saving_moves)
            # Otherwise, play a random move
            while next_move is None:
                candidate_move = random_moves.pop()
                if candidate_move not in already_played_set:
                    next_move = candidate_move
        board.update(Player(i % 2), next_move)
        already_played_set.add(next_move)
        already_played_list.append(next_move)
        bridge_saving_moves = list(_get_bridge_saving_moves(board, Player(i % 2), next_move))
    print(board)  # should never get here, printing board might give useful debugging information if we do
    assert False


def make_games(num_games):
    """
    Generate a list of games.
    See the module docs for more information.
    """
    games = []
    for i in range(num_games):
        if i % 1000 == 0:
            print("Making game %d" % i)
        games.append(make_game())
    return games
