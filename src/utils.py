import numpy as np

def opposite_player(player):
    if player == "red":
        return "blue"
    return "red"

board_size = 5

rb = ("red", "blue")

player_index = {"red": 0, "blue": 1}

neighbour_difference = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

input_names = {(0, False): "current player, no flip",
               (0, True): "current player, 180 degree flip",
               (1, False): "opposite player, no flip",
               (1, True): "opposite player, 180 degree flip"}

initial_position = np.zeros((board_size + 1, board_size + 1, 2), dtype="float32")

initial_position[1:, 0, 0] = 1
initial_position[0, 1:, 1] = 1