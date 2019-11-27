# Generates games by choosing moves with maximum win probability according to a trained model
# TODO: make GameMaker class, load model

import numpy as np
from utils import rb, player_index
import itertools
from board import Board
from random import randint
import time
from enum import Enum

board_size = 5

initial_position = np.zeros((board_size + 1, board_size + 1, 2), dtype="float32")

initial_position[1:, 0, 0] = 1
initial_position[0, 1:, 1] = 1


class GamePhase(Enum):
    BEFORE_SWAP = 1
    MAY_SWAP = 2
    AFTER_SWAP = 3
    FINISHED = 4


class GameMaker:

    def __init__(self, board_size, num_initial_moves, allow_swap):
        self.board = Board(board_size)
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        self.positions = dict()
        self.allow_swap = allow_swap
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            self.positions[(player_perspective, flipped)] = np.copy(initial_position)
        self.num_initial_moves = num_initial_moves
        for i in range(self.num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))
        self.game_phase = GamePhase.BEFORE_SWAP
        self.swapped = None

    def _get_position(self, player_perspective, flipped):
        return self.positions[(player_perspective, flipped)]

    def num_positions_required(self):
        if self.game_phase == GamePhase.BEFORE_SWAP or self.game_phase == GamePhase.AFTER_SWAP:
            return 2 * len(self.valid_moves)
        if self.game_phase == GamePhase.MAY_SWAP:
            return 2 * len(self.valid_moves) + 2
        if self.game_phase == GamePhase.FINISHED:
            return 0
        raise Exception("Unrecognized game phase.")

    def finished(self):
        return self.board.winner is not None

    def game(self):
        return self.moves_played, player_index[self.board.winner], self.swapped

    def refresh(self):
        self.board.refresh()
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(range(self.board.board_size ** 2))
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            np.copyto(self._get_position(player_perspective, flipped), initial_position)
        for i in range(self.num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))
        self.game_phase = GamePhase.BEFORE_SWAP
        self.swapped = None

    def fill_positions(self, position_array):
        if self.game_phase == GamePhase.BEFORE_SWAP or self.game_phase == GamePhase.AFTER_SWAP:
            position_array[:len(self.valid_moves)] = self._get_position(self.current_player, False)
            position_array[len(self.valid_moves):] = self._get_position(self.current_player, True)

            for i, move in enumerate(self.valid_moves):
                a, b = self.board.index_to_point(move)
                if self.current_player == 1:
                    a, b = b, a
                position_array[i, a + 1, b + 1, 0] = 1
                position_array[i + len(self.valid_moves), self.board.board_size - a, self.board.board_size - b, 0] = 1
        if self.game_phase == GamePhase.MAY_SWAP:
            position_array[:len(self.valid_moves)] = self._get_position(self.current_player, False)
            position_array[len(self.valid_moves) + 1: -1] = self._get_position(self.current_player, True)
            position_array[len(self.valid_moves)] = self._get_position(1 - self.current_player, False)
            position_array[-1] = self._get_position(1 - self.current_player, True)

            for i, move in enumerate(self.valid_moves):
                a, b = self.board.index_to_point(move)
                if self.current_player == 1:
                    a, b = b, a
                position_array[i, a + 1, b + 1, 0] = 1
                position_array[i + len(self.valid_moves) + 1,
                               self.board.board_size - a,
                               self.board.board_size - b, 0] = 1
        if self.game_phase == GamePhase.FINISHED:
            return
        raise Exception("Unrecognized game phase.")

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
        if self.board.winner is not None:
            self.game_phase = GamePhase.FINISHED
        # print(self.board)

    def update(self, win_logits):
        if self.game_phase == GamePhase.BEFORE_SWAP:
            best_move_index = min(range(len(self.valid_moves)),
                                  key=lambda x: abs(win_logits[x] + win_logits[x + len(self.valid_moves)]))
            self._play_move(best_move_index)
            if self.allow_swap:
                self.game_phase = GamePhase.MAY_SWAP
            else:
                self.game_phase = GamePhase.AFTER_SWAP
        if self.game_phase == GamePhase.MAY_SWAP:
            best_move_index = max(range(len(self.valid_moves) + 1),
                                  key=lambda x: win_logits[x] + win_logits[x + len(self.valid_moves) + 1])
            if best_move_index < len(self.valid_moves):
                self._play_move(best_move_index)
                self.swapped = 0
            else:
                self.swapped = 1
        if self.game_phase == GamePhase.AFTER_SWAP:
            best_move_index = max(range(len(self.valid_moves)),
                                  key=lambda x: win_logits[x] + win_logits[x + len(self.valid_moves)])
            self._play_move(best_move_index)
        if self.game_phase == GamePhase.FINISHED:
            return

        raise Exception("Unrecognized game phase.")


def make_games(red_model, blue_model, games_required, num_initial_moves, batch_size=3, allow_swap = True):
    game_makers = [GameMaker(board_size, num_initial_moves, allow_swap) for _ in range(batch_size)]
    games = []
    start_time = time.time()

    if num_initial_moves % 2 == 0:
        models = [red_model, blue_model]
    else:
        models = [blue_model, red_model]

    while game_makers:

        for model in models:
            positions = np.zeros(
                [sum(g.num_positions_required() for g in game_makers), board_size + 1, board_size + 1, 2],
                dtype="float32")

            position_counter = 0

            for g in game_makers:
                g.fill_positions(positions[position_counter: position_counter + g.num_positions_required()])
                position_counter += g.num_positions_required()

            win_logits = model.predict(positions)

            position_counter = 0
            for g in game_makers:
                g.update(win_logits[position_counter: position_counter + g.num_positions_required()])
                position_counter += g.num_positions_required()

        new_games = [g.game() for g in game_makers if g.finished()]
        if (len(games) + len(new_games)) // 100 > len(games) // 100:
            print(time.time() - start_time, len(games) + len(new_games))
        games += new_games

        if len(games) > games_required:
            game_makers = [g for g in game_makers if not g.finished()]
        else:
            [g.refresh() for g in game_makers if g.finished()]

    return games


def add_training_data(moves, winner, num_initial_moves, positions_array, winners_array):
    temp_positions = dict(((player_perspective, flipped), np.copy(initial_position))
                          for player_perspective, flipped in itertools.product((0, 1), (False, True)))

    positions_array1 = positions_array[:len(moves[num_initial_moves + 1:])]
    positions_array_2 = positions_array[len(moves[num_initial_moves + 1:]):]
    winners_array_1 = winners_array[:len(moves[num_initial_moves + 1:])]
    winners_array_2 = winners_array[len(moves[num_initial_moves + 1:]):]

    for i, move in enumerate(moves):
        a, b = move
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (board_size - a1, board_size - b1) if flipped else (a1 + 1, b1 + 1)
            temp_positions[(player_perspective, flipped)][a2, b2, (i + player_perspective) % 2] = 1

        # update training data, possibly swapping colours so that current player is always red
        j = i - (num_initial_moves + 1)
        if j >= 0:
            positions_array1[j] = temp_positions[(i % 2, False)]
            positions_array_2[j] = temp_positions[(i % 2, True)]
            winners_array_1[j] = winner == i % 2
            winners_array_2[j] = winners_array[i]


def make_training_data(model, games_required, num_initial_moves, save_filename=None):
    games = make_games(model, model, games_required, num_initial_moves, allow_swap = False)
    total_moves = sum(len(moves[num_initial_moves + 1:]) for moves, winner in games)

    positions_array = np.zeros((total_moves * 2, board_size + 1, board_size + 1, 2), dtype="float32")
    winners_array = np.zeros(total_moves * 2, dtype="float32")

    total_moves_counter = 0

    while games:
        if len(games) % 1000 == 0:
            print(len(games), "more games to process.")
        moves, winner, swapped = games.pop()
        counter_diff = 2 * len(moves[num_initial_moves + 1:])
        add_training_data(moves, winner, num_initial_moves,
                          positions_array[total_moves_counter: total_moves_counter + counter_diff],
                          winners_array[total_moves_counter: total_moves_counter + counter_diff])
        # total_moves_counter += len(moves)
        total_moves_counter += counter_diff

    if save_filename is not None:
        np.savez("../data/%s" % save_filename,
                 positions=positions_array,
                 winners=winners_array)

    return positions_array, winners_array

# for i in range(total_moves * 2):
#     print(positions_array[i, :, :, 0])
#     print(positions_array[i, :, :, 1])
#     print("------------------------------")

# np.savez("training_data5.npz",
#          positions=positions_array,
#          winners=winners_array)
