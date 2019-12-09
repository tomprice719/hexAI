import make_smart_games
import make_dumb_games
from training_data import make_training_data
from board_utils import Board, Player
from utils import board_size
import numpy as np
from model import create_model
from collections import defaultdict
import itertools
import time

model1 = create_model(depth=18, breadth=40)
model1.load_weights('../data/model2.h5')

model2 = create_model(depth=18, breadth=40, learning_rate=0.0001)
model2.load_weights('../data/model3.h5')

num_initial_moves = 2


def compare_models(model1, model2, num_games):
    win_counter = defaultdict(int)
    for game, winner, swapped in make_smart_games.make_games(model1, model2, num_games, num_initial_moves):
        win_counter[(winner, swapped)] += 1
    for winner, swapped in itertools.product((0, 1), (0, 1)):
        print("winner: %d swapped: %d count: %d" %
              (winner, swapped, win_counter[(winner, swapped)]))
    print("win ratio %f" % ((win_counter[(0, 0)] + win_counter[(1, 1)]) / num_games))


def show_game(red_model, blue_model, num_games = 1):
    games = make_smart_games.make_games(red_model, blue_model, num_games, num_initial_moves, batch_size=1)
    for moves, winner, swapped in games:
        b = Board(board_size)
        for i, (move, annotation) in enumerate(moves):
            b.update(Player(i % 2), move)
            print(b)
            print(annotation)
            print("-------------------------------------")
        print("winner:", winner, "swapped:", swapped)
        print()


# def train_from_selfplay(model, epoch_size, new_games_per_epoch, num_iterations, training_data):
#     if training_data is not None:
#         positions, winners = training_data
#     else:
#         positions, winners = make_training_data(model, 0, num_initial_moves)
#
#     for i in range(num_iterations):
#         if i % 100 == 0:
#             print(i)
#         new_positions, new_winners = make_training_data(model, new_games_per_epoch, num_initial_moves)
#         positions = np.append(new_positions, positions, axis=0)[:epoch_size]
#         winners = np.append(new_winners, winners, axis=0)[:epoch_size]
#
#         model.fit(
#             positions,
#             winners,
#             batch_size=64,
#             epochs=1,
#             shuffle=True,
#             verbose=0
#         )
#
#     return positions, winners


def train_from_selfplay(model, new_games_per_epoch, num_iterations, use_weight=False):
    for i in range(num_iterations):
        if i % 10 == 0:
            print(i)
        games = make_smart_games.make_games(model, model, new_games_per_epoch, num_initial_moves)
        assert len(games) == new_games_per_epoch
        positions, winners, move_numbers = make_training_data(games, num_initial_moves)

        model.fit(
            positions,
            winners,
            batch_size=64,
            epochs=1,
            shuffle=True,
            verbose=0,
            sample_weight=1 / move_numbers if use_weight else None
        )


def train_from_file(model, filename, num_epochs):
    data = dict(np.load("../data/%s" % filename))

    model.fit(
        data,
        data,
        batch_size=64,
        epochs=num_epochs,
        shuffle=True,
        validation_split=0.05
    )


def make_initial_training_data(num_games, filename):
    games = make_dumb_games.make_games(num_games)
    make_training_data(games, 0, filename)


# make_initial_training_data(30000, "games1.npz")

# train_from_file(model1, "games1.npz", 1)
# model1.save_weights('../data/model1.h5')


start_time = time.time()

while (True):
    print(time.time() - start_time)
    train_from_selfplay(model2, 10, 300, False)
    model2.save_weights('../data/model4.h5')
    show_game(model2, model2)
    compare_models(model2, model2, 100)
    print("-")
    compare_models(model1, model2, 100)
    print("-")
    compare_models(model2, model1, 100)
