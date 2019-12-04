from make_games5 import make_games, make_training_data
from board import Board
from utils import rb
import numpy as np
from keras_model4 import create_model

model1 = create_model()
model1.load_weights('../data/my_model2.h5')

model2 = create_model(5, 40)
model2.load_weights('../data/my_model2.h5')

board_size = 5

num_initial_moves = 2

def compare_models(model1, model2, half_num_games):
    red_wins = [(winner + swapped) % 2 for game, winner, swapped in
                make_games(model1, model2, half_num_games, num_initial_moves)].count(0) + \
               [(winner + swapped) % 2 for game, winner, swapped in
                make_games(model2, model1, half_num_games, num_initial_moves)].count(1)
    return red_wins / (half_num_games * 2)


def show_game(red_model, blue_model):
    moves, winner, swapped = make_games(red_model, blue_model, 1, num_initial_moves, batch_size=1)[0]
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
        positions, winners = make_training_data(model, new_games_per_epoch, num_initial_moves)

        model.fit(
            positions,
            winners,
            batch_size=64,
            epochs=1,
            shuffle=True,
            verbose=0
        )


def train_from_file(model, filename, num_epochs):
    data = np.load("../data/%s" % filename)

    model.fit(
        data,
        data,
        batch_size=64,
        epochs=num_epochs,
        shuffle=True,
        validation_split=0.05
    )


# #make_training_data(model1, 10000, 4, "training_data.npz")
# train_from_file(model2, "training_data.npz", 3)
# model2.save_weights('../data/my_model4.h5')
#
# print("finished initializing model")

while (True):
    train_from_selfplay(model2, 10, 300)
    model2.save_weights('../data/my_model7.h5')
    show_game(model2, model2)
    print("WIN RATIO", compare_models(model2, model2, 100))
    print("WIN RATIO", compare_models(model2, model1, 500))

# model2.save_weights('../data/my_model3.h5')
# print(compare_models(model1, model2, 10000, 4))

# while True:
#     train(model2, 500000, 10000, 1)
#     print(compare_models(model1, model2, 500, 4))
