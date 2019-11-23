from make_games5 import make_games, make_training_data
from board import Board
from utils import rb
import numpy as np
from keras_model4 import create_model

model1 = create_model()
model1.load_weights('my_model2.h5')

model2 = create_model()
model2.load_weights('my_model2.h5')

board_size = 5


def compare_models(model1, model2, half_num_games, num_initial_moves):
    red_wins = [winner for game, winner in
                make_games(model1, model2, half_num_games, num_initial_moves)].count(0) + \
               [winner for game, winner in
                make_games(model2, model1, half_num_games, num_initial_moves)].count(1)
    return red_wins / (half_num_games * 2)


def show_game(red_model, blue_model, num_initial_moves):
    moves, winner = make_games(red_model, blue_model, 1, num_initial_moves, batch_size=1)[0]
    b = Board(board_size)
    for i, move in enumerate(moves):
        b.update(rb[i % 2], b.point_to_index(move))
        print(b)


def train(model, epoch_size, new_games_per_epoch, num_iterations):
    positions, winners = make_training_data(model, 0, 4)

    for _ in range(num_iterations):
        new_positions, new_winners = make_training_data(model, new_games_per_epoch, 4)
        positions = np.append(new_positions, positions, axis=0)[:epoch_size]
        winners = np.append(new_winners, winners, axis=0)[:epoch_size]

        model.fit(
            positions,
            winners,
            batch_size=16,
            epochs=1,
            shuffle=True
        )


while True:
    train(model2, 500000, 10000, 1)
    print(compare_models(model1, model2, 500, 4))
