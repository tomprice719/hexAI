def opposite_player(player):
    if player == "red":
        return "blue"
    return "red"


rb = ("red", "blue")

player_index = {"red": 0, "blue": 1}

neighbour_difference = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

input_names = {(0, False): "current player, no flip",
               (0, True): "current player, 180 degree flip",
               (1, False): "opposite player, no flip",
               (1, True): "opposite player, 180 degree flip"}
