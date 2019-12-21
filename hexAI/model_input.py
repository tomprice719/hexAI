import numpy as np
from .config import board_size
from .board_utils import Board
import itertools

input_names = {(0, False): "current_player_no_flip",
               (0, True): "current_player_180_flip",
               (1, False): "opposite_player_no_flip",
               (1, True): "opposite_player_180_flip"}

_initial_quarter_position = np.zeros((board_size + 1, board_size + 1, 2), dtype="float32")

_initial_quarter_position[1:, 0, 0] = 1
_initial_quarter_position[0, 1:, 1] = 1


def _transform_coordinates(point, player_perspective, flip):
    a, b = point
    a1, b1 = (a, b) if player_perspective == 0 else (b, a)
    return (board_size - a1, board_size - b1) if flip is True else (a1 + 1, b1 + 1)


def create_array_position():
    return dict((k, np.copy(_initial_quarter_position)) for k in itertools.product((0, 1), (False, True)))


def update_array_position(position, player, point):
    player = player.value
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        a, b = _transform_coordinates(point, player_perspective, flipped)
        position[player_perspective, flipped][a, b, (player + player_perspective) % 2] = 1


class ArrayBoard(Board):
    def __init__(self, board_size):
        Board.__init__(self, board_size)
        self.array_position = create_array_position()

    def update(self, player, point):
        Board.update(self, player, point)
        update_array_position(self.array_position, player, point)


def initialize_model_input(num_positions):
    return dict((input_names[k],
                 np.zeros([num_positions,
                           board_size + 1,
                           board_size + 1,
                           2],
                          dtype="float32"))
                for k in itertools.product((0, 1), (False, True)))


def fill_model_input(model_input, array_position, player_perspective, slice_):
    player_perspective = player_perspective.value
    for inner_player_perspective, flipped in itertools.product((0, 1), (False, True)):
        model_input[input_names[((player_perspective + inner_player_perspective) % 2, flipped)]][slice_] = \
            array_position[(inner_player_perspective, flipped)]


def update_model_input(model_input, moves, player, player_perspective, slice_):
    player = player.value
    player_perspective = player_perspective.value
    for i, move in enumerate(moves):
        for inner_player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a, b = _transform_coordinates(move, inner_player_perspective, flipped)
            model_input[input_names[((player_perspective + inner_player_perspective) % 2, flipped)]]\
                [slice_][i, a, b, (player + inner_player_perspective) % 2] = 1
