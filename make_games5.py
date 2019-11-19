# Generates games by choosing moves with maximum win probability according to a trained model
# TODO: make GameMaker class, load model

import numpy as np

board_size = 5
batch_size = 5  # number of games to create simultaneously
games_required = 10000
game_makers = [GameMaker() for _ in range(batch_size)]
games = []

while len(games) < games_required:

    [g.refresh() for g in game_makers if g.finished()]

    positions = np.zeros([sum(g.num_positions_required() for g in game_makers), board_size, board_size, 2])

    position_counter = 0
    for g in game_makers:
        g.fill_positions(positions[position_counter: position_counter + g.num_positions_required()])
        position_counter += g.num_positions_required()

    win_probs = model.predict(positions)

    position_counter = 0
    for g in game_makers:
        g.update(win_probs[position_counter: position_counter + g.num_positions_required()])
        position_counter += g.num_positions_required()

    games += [g.game() for g in game_makers if g.finished()]


def add_training_data(moves, winner, positions_array, winners_array):
    p, p_swapped = np.zeros_like(positions_array[0, 1:, 1:]), np.zeros_like(positions_array[0, 1:, 1:])
    for i, move in enumerate(moves):
        a, b = move
        # Update temporary boards
        p[a, b, i % 2] = 1
        p_swapped[b, a, (i + 1) % 2] = 1
        # update training data, possibly swapping colours so that current player is always red
        positions_array[i, 1:, 1:] = (p, p_swapped)[i % 2]
        winners_array[i] = winner == i % 2


total_moves = sum(len(moves) for moves, winner in games)

positions_array = np.zeros((total_moves, board_size + 1, board_size + 1, 2))
winners_array = np.zeros(total_moves)

positions_array[:, 1:, 0, 0] = 1
positions_array[:, 0, 1:, 1] = 1

total_moves_counter = 0

while games:
    if len(games) % 1000 == 0:
        print(len(games), "more games to process.")
    moves, winner = games.pop()
    add_training_data(moves, winner,
                      positions_array[total_moves_counter: total_moves_counter + len(moves)],
                      winners_array[total_moves_counter: total_moves_counter + len(moves)])
    # total_moves_counter += len(moves)
    total_moves_counter += len(moves)
