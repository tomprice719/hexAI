# Generates games by choosing moves with maximum win probability according to a trained model
# TODO: make GameMaker class, load model

import numpy as np
from utils import rb
import itertools
from board import Board

board_size = 5

initial_position = np.zeros((board_size, board_size, 2))

initial_position[1:, 0, 0] = 1
initial_position[0, 1:, 1] = 1


class GameMaker:

    def __init__(self, board_size, num_initial_moves):
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        self.board = Board(board_size)
        self.positions = dict()
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            self.positions[(player_perspective, flipped)] = np.copy(initial_position)
        self.num_initial_moves = num_initial_moves
        for move_index in np.random.choice(self.valid_moves, size=num_initial_moves, replace=False):
            self._play_move(move_index)

    def _get_position(self, player_perspective, flipped):
        return self.positions[(player_perspective, flipped)]

    def num_positions_required(self):
        return 2 * len(self.valid_moves)

    def finished(self):
        return self.board.winner is not None

    def game(self):
        return self.moves_played, self.board.winner

    def refresh(self):
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        self.board.refresh()
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            np.copyto(self._get_position(player_perspective, flipped), initial_position)
        for move_index in np.random.choice(self.valid_moves, size=self.num_initial_moves, replace=False):
            self._play_move(move_index)

    def fill_positions(self, position_array):
        position_array[:len(self.valid_moves)] = self._get_position(self.current_player, False)
        position_array[len(self.valid_moves):] = self._get_position(self.current_player, True)

        for i, move in enumerate(self.valid_moves):
            a, b = self.board.index_to_point(move)
            if self.current_player == 1:
                a, b = b, a
            position_array[i, a, b, 0] = 1
            position_array[i, self.board.board_size - a, self.board.board_size - b, 0] = 1

    def _play_move(self, move_index):
        """Plays a move on the board, where the move is specified by its index in valid_moves"""
        move = self.valid_moves[move_index]
        a, b = self.board.index_to_point(move)

        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (self.board.board_size - a1, self.board.board_size - b1) if flipped else (a1, b1)
            self._get_position(player_perspective, flipped)[a2, b2, self.current_player + player_perspective % 2] = 1

        self.board.update(rb(self.current_player), move)
        self.current_player = 1 - self.current_player
        self.valid_moves[move_index] = self.valid_moves[-1]
        del self.valid_moves[-1]
        self.moves_played.append(move)

    def update(self, win_probs):
        best_move_index = max(range(len(self.valid_moves)),
                              key=lambda x: win_probs[x] + win_probs[x + len(self.valid_moves)])
        self._play_move(best_move_index)


board_size = 5
batch_size = 5  # number of games to create simultaneously
games_required = 10000
game_makers = [GameMaker(board_size, 4) for _ in range(batch_size)]
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
