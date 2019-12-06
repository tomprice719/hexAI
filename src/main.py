import make_smart_games
import make_dumb_games
from training_data import make_training_data
from board import Board
from utils import rb, board_size
import numpy as np
from keras_model4 import create_model
from collections import defaultdict
import itertools

model1 = create_model(5, 40)
model1.load_weights('../data/my_model1.h5')

model2 = create_model(5, 40)
model2.load_weights('../data/my_model1.h5')

num_initial_moves = 2


def compare_models(model1, model2, num_games):
    win_counter = defaultdict(int)
    for game, winner, swapped in make_smart_games.make_games(model1, model2, num_games, num_initial_moves):
        win_counter[(winner, swapped)] += 1
    for winner, swapped in itertools.product((0, 1), (0, 1)):
        print("winner: %d swapped: %d count: %d" %
              (winner, swapped, win_counter[(winner, swapped)]))


def show_game(red_model, blue_model):
    moves, winner, swapped = make_smart_games.make_games(red_model, blue_model, 1, num_initial_moves, batch_size=1)[0]
    b = Board(board_size)
    for i, (move, annotation) in enumerate(moves):
        b.update(rb[i % 2], b.point_to_index(move))
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


def train_from_selfplay(model, new_games_per_epoch, num_iterations):
    for i in range(num_iterations):
        if i % 10 == 0:
            print(i)
        games = make_smart_games.make_games(model, model, new_games_per_epoch, num_initial_moves)
        positions, winners = make_training_data(games, num_initial_moves)

        model.fit(
            positions,
            winners,
            batch_size=64,
            epochs=1,
            shuffle=True,
            verbose=0
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

#make_initial_training_data(10000, "games1.npz")

# train_from_file(model1, "games1.npz", 1)
# model1.save_weights('../data/my_model1.h5')


while (True):
    train_from_selfplay(model2, 10, 300)
    model2.save_weights('../data/my_model2.h5')
    show_game(model2, model2)
    compare_models(model1, model1, 100)
    print("-")
    compare_models(model1, model2, 100)
    print("-")
    compare_models(model2, model1, 100)
