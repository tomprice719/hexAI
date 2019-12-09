import numpy as np


board_size = 9

input_names = {(0, False): "current_player_no_flip",
               (0, True): "current_player_180_flip",
               (1, False): "opposite_player_no_flip",
               (1, True): "opposite_player_180_flip"}

initial_position = np.zeros((board_size + 1, board_size + 1, 2), dtype="float32")

initial_position[1:, 0, 0] = 1
initial_position[0, 1:, 1] = 1
