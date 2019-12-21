"""
Utilities for creating input data for models.

Terminology:

model_input: something that can be used as the input of a model, e.g. as the argument of the predict method.

array_position: a representation of the state of hex board that can be used for creating a model_input.
This contrasts with the Board class, which also represents the state of a hex board,
but is better suited to determining if the game has been won.
"""

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


def new_array_position():
    """
    Creates an array position corresponding to an empty board.
    """
    return dict((k, np.copy(_initial_quarter_position)) for k in itertools.product((0, 1), (False, True)))


def update_array_position(array_position, player, point):
    """
    Adds a hex tile to an array position.

    parameters:

        array_position: the array_position to add the tile to.

        player: element of the board_utils.Player enum corresponding to the colour of the hex tile played.

        point: tuple of integers representing the point at which to play the tile.
    """
    player = player.value
    for player_perspective, flipped in itertools.product((0, 1), (False, True)):
        a, b = _transform_coordinates(point, player_perspective, flipped)
        array_position[player_perspective, flipped][a, b, (player + player_perspective) % 2] = 1


class ArrayBoard(Board):
    """
    Adds an array_position to the board class which automatically updates with calls to the Board.update method.
    """
    def __init__(self, board_size):
        Board.__init__(self, board_size)
        self.array_position = new_array_position()

    def update(self, player, point):
        Board.update(self, player, point)
        update_array_position(self.array_position, player, point)


def new_model_input(num_positions):
    """
    Creates a new model_input.
    parameters:

        num_positions: number of hex board states represented in the input. The output of the model should then be
        an array of length num_positions, representing the win probability (in logits) at each position.
    """
    return dict((input_names[k],
                 np.zeros([num_positions,
                           board_size + 1,
                           board_size + 1,
                           2],
                          dtype="float32"))
                for k in itertools.product((0, 1), (False, True)))


def fill_model_input(model_input, array_position, player_perspective, slice_):
    """
    fills a model_input with copies of an array position.
    parameters:

        model_input: the model_input to alter

        array_position: the array_position to fill it with

        player_perspective: element of the board_utils.Player enum, representing the player whose turn it is,
        and for whom the model is computing a win probability.

        slice_: slice object describing at which sample indices we should fill with array_position
    """
    player_perspective = player_perspective.value
    for inner_player_perspective, flipped in itertools.product((0, 1), (False, True)):
        model_input[input_names[((player_perspective + inner_player_perspective) % 2, flipped)]][slice_] = \
            array_position[(inner_player_perspective, flipped)]


def update_model_input(model_input, moves, player, player_perspective, slice_):
    """
    Adds a single hex tile to a number of positions in a model_input.
    parameters:

        model_input: the model_input to alter

        moves: a list of points describing where the tiles should be added. Though each position only receives at most
        one new tile, multiple points can be given in order to update multiple positions. The ith point is used to
        update the ith position in the specified slice.

        player: An element of the board_utils.Player enum representing the colour of the new tile.

        player_perspective: element of the board_utils.Player enum, representing the player whose turn it is,
        and for whom the model is computing a win probability. This is assumed to be constant across all samples in the
        specified slice. If it isn't, multiple calls to update_model_input must be used.

        slice_: slice object describing at which sample indices we should add a new tile.
    """
    player = player.value
    player_perspective = player_perspective.value
    for i, move in enumerate(moves):
        for inner_player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a, b = _transform_coordinates(move, inner_player_perspective, flipped)
            model_input[input_names[((player_perspective + inner_player_perspective) % 2, flipped)]]\
                [slice_][i, a, b, (player + inner_player_perspective) % 2] = 1
