def opposite_player(player):
    if player == "red":
        return "blue"
    return "red"


rb = ("red", "blue")

player_sign = {"red": 1, "blue": -1}

neighbour_difference = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
