"""Utilities for representing the state of a hex game."""

from enum import Enum
import itertools
from colorama import Fore, Back, Style
from string import ascii_uppercase

neighbour_difference = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]


class Player(Enum):
    RED = 0
    BLUE = 1


def all_points(board_size):
    return itertools.product(range(board_size), range(board_size))


def opposite_player(player):
    return Player(1 - player.value)


_symbols = {(Player.RED, False): Fore.LIGHTRED_EX + u"\u25CF " + Style.RESET_ALL,
            (Player.BLUE, False): Fore.LIGHTBLUE_EX + u"\u25CF " + Style.RESET_ALL,
            "empty": Fore.WHITE + u"\u25CF " + Style.RESET_ALL,
            (Player.RED, True): Fore.LIGHTRED_EX + u"\u25C9 " + Style.RESET_ALL,
            (Player.BLUE, True): Fore.LIGHTBLUE_EX + u"\u25C9 " + Style.RESET_ALL}


class Board:
    """Represents the board state during a game of hex."""

    def __init__(self, board_size):
        """board size: int representing number of tiles along each side of the board."""
        self.board_size = board_size
        self.winner = None
        self._hexes = dict((p, set()) for p in Player)
        self._connected = dict((p, set()) for p in Player)
        self._last_move = None
        self._top = [(x, 0) for x in range(board_size)]
        self._bottom = [(x, board_size - 1) for x in range(board_size)]
        self._left = [(0, y) for y in range(board_size)]
        self._right = [(board_size - 1, y) for y in range(board_size)]

        self._top_connected = set()
        self._top_boundary = set(self._top)
        self._bottom_connected = set()
        self._bottom_boundary = set(self._bottom)
        self._left_connected = set()
        self._left_boundary = set(self._left)
        self._right_connected = set()
        self._right_boundary = set(self._right)

    def update(self, player, point):
        """
        Adds a hex tile to the board.

        player: element of Player enum representing which player adds the tile.
        point: a tuple of integers representing the coordinates of the point.
        """
        assert (all(point not in self._hexes[p] for p in Player))
        self._hexes[player].add(point)
        self._update_sets(player, point)
        self._last_move = point

    def has_hex(self, player, point):
        """
        Checks if a point is occupied by a given player's tile.

        player: element of Pleyer enum.
        point: a tuple of integers representing the coordinates of the point.
        """
        return point in self._hexes[player]

    def legal_move(self, point):
        """
        Checks if a particular point on the board can legally be played in.
        point: a tuple of integers representing the coordinates of the point.
        """
        return not any(self.has_hex(p, point) for p in Player)

    def _starting_side(self, player, point):
        i, j = point
        k = (j if player == Player.RED else i)
        return k == 0

    def _finishing_side(self, player, point):
        i, j = point
        k = (j if player == Player.RED else i)
        return k == self.board_size - 1

    def _neighbours(self, point):
        i, j = point
        return [(i + di, j + dj) for di, dj in neighbour_difference
                if 0 <= i + di < self.board_size
                and 0 <= j + dj < self.board_size]

    def _expand(self, interior, boundary, player, starting_point):
        new_points = set()

        if starting_point in boundary:
            interior.add(starting_point)
            new_points.add(starting_point)

        while new_points:
            point = new_points.pop()
            for neighbour in self._neighbours(point):
                if neighbour not in interior:
                    if neighbour in self._hexes[player]:
                        interior.add(neighbour)
                        new_points.add(neighbour)
                    elif neighbour not in self._hexes[opposite_player(player)]:
                        boundary.add(neighbour)

    def winning_moves(self, player):
        """
        Gives a list of all points that would cause the given player to immediately win they played there.
        player: element of the Player enum.
        """
        if player == Player.RED:
            return list(self._top_boundary.intersection(self._bottom_boundary))
        if player == Player.BLUE:
            return list(self._left_boundary.intersection(self._right_boundary))
        raise Exception("Player must be an element of the Player enum.")

    def _update_sets(self, player, move_point):
        if player == Player.RED:
            self._expand(self._top_connected, self._top_boundary, Player.RED, move_point)
            self._expand(self._bottom_connected, self._bottom_boundary, Player.RED, move_point)
            if move_point in self._top_connected and move_point in self._bottom_connected:
                self.winner = Player.RED
        elif player == Player.BLUE:
            self._expand(self._left_connected, self._left_boundary, Player.BLUE, move_point)
            self._expand(self._right_connected, self._right_boundary, Player.BLUE, move_point)
            if move_point in self._left_connected and move_point in self._right_connected:
                self.winner = Player.BLUE
        for boundary in [self._top_boundary,
                         self._bottom_boundary,
                         self._left_boundary,
                         self._right_boundary]:
            boundary.discard(move_point)

    def print_debug_data(self):
        def print_points(set_, label):
            s = label + "\n"
            for i in range(self.board_size):
                s += " " * i
                for j in range(self.board_size):
                    symbol = " ."
                    if (j, i) in set_:
                        symbol = " x"
                    s += symbol
                s += "\n"
            s += "------------------------------------"
            print(s)

        print_points(self._top_connected, "top connected")
        print_points(self._bottom_connected, "bottom connected")
        print_points(self._left_connected, "left connected")
        print_points(self._right_connected, "right connected")
        print_points(self._top_boundary, "top boundary")
        print_points(self._bottom_boundary, "bottom boundary")
        print_points(self._left_boundary, "left boundary")
        print_points(self._right_boundary, "right boundary")

    def __repr__(self):
        rep = "   " + Back.RED + Fore.WHITE \
              + "".join(["%s " % c for c in ascii_uppercase[:self.board_size]]) \
              + Style.RESET_ALL + "\n"
        for i in range(self.board_size):
            rep += " " * i + Back.BLUE + Fore.WHITE + "%d " % (i + 1) + Style.RESET_ALL + " "
            for j in range(self.board_size):
                symbol = _symbols["empty"]
                for player in Player:
                    if self.has_hex(player, (j, i)):
                        new = (j, i) == self._last_move
                        symbol = _symbols[(player, new)]
                rep += symbol
            rep += Back.BLUE + "  " + Style.RESET_ALL + "\n"
        rep += " " * (self.board_size + 2) + Back.RED + " " * self.board_size * 2 + Style.RESET_ALL
        return rep
