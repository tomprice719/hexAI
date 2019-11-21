# Generates games by choosing moves with maximum win probability according to a trained model
# TODO: make GameMaker class, load model

import numpy as np
from utils import rb
import itertools
from board import Board
from keras_model4 import model
from random import randint
from utils import player_index

model.load_weights('my_model2.h5')

board_size = 5

initial_position = np.zeros((board_size + 1, board_size + 1, 2))

initial_position[1:, 0, 0] = 1
initial_position[0, 1:, 1] = 1


class GameMaker:

    def __init__(self, board_size, num_initial_moves):
        self.board = Board(board_size)
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        self.positions = dict()
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            self.positions[(player_perspective, flipped)] = np.copy(initial_position)
        self.num_initial_moves = num_initial_moves
        for i in range(self.num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))

    def _get_position(self, player_perspective, flipped):
        return self.positions[(player_perspective, flipped)]

    def num_positions_required(self):
        return 2 * len(self.valid_moves)

    def finished(self):
        return self.board.winner is not None

    def game(self):
        return self.moves_played, player_index[self.board.winner]

    def refresh(self):
        self.board.refresh()
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            np.copyto(self._get_position(player_perspective, flipped), initial_position)
        for i in range(self.num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))

    def fill_positions(self, position_array):
        position_array[:len(self.valid_moves)] = self._get_position(self.current_player, False)
        position_array[len(self.valid_moves):] = self._get_position(self.current_player, True)

        for i, move in enumerate(self.valid_moves):
            a, b = self.board.index_to_point(move)
            if self.current_player == 1:
                a, b = b, a
            position_array[i, a + 1, b + 1, 0] = 1
            position_array[i, self.board.board_size - a, self.board.board_size - b, 0] = 1

    def _play_move(self, move_index):
        """Plays a move on the board, where the move is specified by its index in valid_moves"""
        move = self.valid_moves[move_index]
        a, b = self.board.index_to_point(move)

        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (self.board.board_size - a1, self.board.board_size - b1) if flipped else (a1 + 1, b1 + 1)
            self._get_position(player_perspective, flipped)[a2, b2, (self.current_player + player_perspective) % 2] = 1

        self.board.update(rb[self.current_player], move)
        self.current_player = 1 - self.current_player
        self.valid_moves[move_index] = self.valid_moves[-1]
        del self.valid_moves[-1]
        self.moves_played.append((a, b))
        print(self.board)

    def update(self, win_probs):
        best_move_index = max(range(len(self.valid_moves)),
                              key=lambda x: win_probs[x] + win_probs[x + len(self.valid_moves)])
        self._play_move(best_move_index)


board_size = 5
batch_size = 1  # number of games to create simultaneously
games_required = 1
num_initial_moves = 4
game_makers = [GameMaker(board_size, num_initial_moves) for _ in range(batch_size)]
games = []

while len(games) < games_required:

    [g.refresh() for g in game_makers if g.finished()]

    positions = np.zeros([sum(g.num_positions_required() for g in game_makers), board_size + 1, board_size + 1, 2])

    position_counter = 0
    for g in game_makers:
        g.fill_positions(positions[position_counter: position_counter + g.num_positions_required()])
        position_counter += g.num_positions_required()

    win_probs = model.predict(positions)

    position_counter = 0
    for g in game_makers:
        g.update(win_probs[position_counter: position_counter + g.num_positions_required()])
        position_counter += g.num_positions_required()

    new_games = [g.game() for g in game_makers if g.finished()]
    if (len(games) + len(new_games)) // 100 > len(games) // 100:
        print(len(games) + len(new_games))
    games += new_games


def add_training_data(moves, winner, positions_array, winners_array):
    temp_positions = dict(((player_perspective, flipped), np.copy(initial_position))
                          for player_perspective, flipped in itertools.product((0, 1), (False, True)))

    for i, move in enumerate(moves):
        a, b = move
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (board_size - a1, board_size - b1) if flipped else (a1 + 1, b1 + 1)
            temp_positions[(player_perspective, flipped)][a2, b2, (i + player_perspective) % 2] = 1

        # update training data, possibly swapping colours so that current player is always red
        positions_array[i] = temp_positions[(i % 2, False)]
        positions_array[i + len(moves)] = temp_positions[(i % 2, True)]
        winners_array[i] = winner == i % 2
        winners_array[i + len(moves)] = winners_array[i]

total_moves = sum(len(moves) for moves, winner in games)

positions_array = np.zeros((total_moves * 2, board_size + 1, board_size + 1, 2))
winners_array = np.zeros(total_moves * 2)

total_moves_counter = 0

while games:
    if len(games) % 1000 == 0:
        print(len(games), "more games to process.")
    moves, winner = games.pop()
    add_training_data(moves, winner,
                      positions_array[total_moves_counter: total_moves_counter + 2 * len(moves)],
                      winners_array[total_moves_counter: total_moves_counter + 2 * len(moves)])
    # total_moves_counter += len(moves)
    total_moves_counter += 2 * len(moves)

# for i in range(total_moves * 2):
#     print(positions_array[i, :, :, 0])
#     print(positions_array[i, :, :, 1])
#     print("------------------------------")

# np.savez("training_data5.npz",
#          positions=positions_array,
#          winners=winners_array)

