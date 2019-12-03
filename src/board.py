from utils import opposite_player, neighbour_difference

symbols = {("red", False) : "o ", ("blue", False) : "* ", "empty": ". ",
           ("red", True): "@ ", ("blue", True) : "# "}

class Board:

    def __init__(self, board_size):
        self.board_size = board_size
        self.winner = None
        self._hexes = {"red": set(), "blue": set()}
        self._connected = {"red": set(), "blue": set()}
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

    def is_empty(self):
        return (not self._hexes["red"]) and (not self._hexes["blue"])
    
    def update(self, player, move):
        point = self.index_to_point(move)
        assert(point not in self._hexes["red"])
        assert (point not in self._hexes["blue"])
        self._hexes[player].add(point)
        self._update_sets(player, point)
        self._last_move = point

    def legal_move(self, index):
        point = self.index_to_point(index)
        return not (self.has_hex("red", point) or self.has_hex("blue", point))

    def has_hex(self, player, point):
        return point in self._hexes[player]

    def refresh(self):
        self._hexes["red"].clear()
        self._hexes["blue"].clear()
        self._top_connected.clear()
        self._top_boundary.clear()
        self._top_boundary.update(self._top)
        self._bottom_connected.clear()
        self._bottom_boundary.clear()
        self._bottom_boundary.update(self._bottom)
        self._left_connected.clear()
        self._left_boundary.clear()
        self._left_boundary.update(self._left)
        self._right_connected.clear()
        self._right_boundary.clear()
        self._right_boundary.update(self._right)
        self.winner = None

    def index_to_point(self, i):
        return (i % self.board_size, i // self.board_size)

    def point_to_index(self, point):
        x, y = point
        return x + y * self.board_size

    def _starting_side(self, player, point):
        i, j = point
        k = (j if player == "red" else i)
        return k == 0

    def _finishing_side(self, player, point):
        i, j = point
        k = (j if player == "red" else i)
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
        if player == "red":
            return self._top_boundary.intersection(self._bottom_boundary)
        if player == "blue":
            return self._left_boundary.intersection(self._right_boundary)

    def _update_sets(self, player, move_point):
        if player == "red":
            self._expand(self._top_connected, self._top_boundary, "red", move_point)
            self._expand(self._bottom_connected, self._bottom_boundary, "red", move_point)
            if move_point in self._top_connected and move_point in self._bottom_connected:
                self.winner = "red"
        elif player == "blue":
            self._expand(self._left_connected, self._left_boundary, "blue", move_point)
            self._expand(self._right_connected, self._right_boundary, "blue", move_point)
            if move_point in self._left_connected and move_point in self._right_connected:
                self.winner = "blue"
        for boundary in [self._top_boundary,
                         self._bottom_boundary,
                         self._left_boundary,
                         self._right_boundary]:
            boundary.discard(move_point)

    def to_string(self, set_):
        rep = ""
        for i in range(self.board_size):
            rep += " " * i
            for j in range(self.board_size):
                symbol = " ."
                if (j, i) in set_:
                    symbol = " x"
                rep += symbol
            rep += "\n"
        rep += "---------------------------------------------------------------"
        return rep

    def print_stuff(self):
        print(self.to_string(self._top_connected))
        print(self.to_string(self._bottom_connected))
        print(self.to_string(self._left_connected))
        print(self.to_string(self._right_connected))
        print(self.to_string(self._top_boundary))
        print(self.to_string(self._bottom_boundary))
        print(self.to_string(self._left_boundary))
        print(self.to_string(self._right_boundary))

    def __repr__(self):
        rep = ""
        for i in range(self.board_size):
            rep += " " * i
            for j in range(self.board_size):
                symbol = symbols["empty"]
                for player in ["red", "blue"]:
                    if self.has_hex(player, (j, i)):
                        new = (j, i) == self._last_move
                        symbol = symbols[(player, new)]
                rep += symbol
            rep += "\n"
        return rep
