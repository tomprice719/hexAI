# Generates games by choosing moves with maximum win probability according to a trained model
# TODO: make GameMaker class, load model

import numpy as np
from utils import rb, player_index, input_names
import itertools
from board import Board
from random import randint
import time
from enum import Enum
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
        self.noswap_threshold = None

    def _get_position(self, player_perspective, flipped):
        return self.positions[(player_perspective, flipped)]

    def num_positions_required(self):
        if self.game_phase == GamePhase.FINISHED:
            return 0
        else:
            return len(self.valid_moves)

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
        self.noswap_threshold = None

    def fill_positions(self, positions, slice_):
        if self.game_phase == GamePhase.FINISHED:
            return
        else:
            for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                positions[input_names[(player_perspective, flipped)]][slice_] = \
                    self._get_position((self.current_player + player_perspective) % 2, flipped)

            for i, move in enumerate(self.valid_moves):
                a, b = self.board.index_to_point(move)
                for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                    a1, b1 = (a, b) if player_perspective == 0 else (b, a)
                    a2, b2 = (self.board.board_size - a1, self.board.board_size - b1) if flipped else (a1 + 1, b1 + 1)
                    positions[input_names[(player_perspective, flipped)]][slice_][i, a2, b2, player_perspective] = 1

    def _play_move(self, move_index, annotation=None):
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
        self.moves_played.append(((a, b), annotation))
        # print(self.board)

    def update(self, win_logits, model_label):
        if self.game_phase == GamePhase.BEFORE_SWAP:
            medium_move_index = min(range(len(self.valid_moves)),
                                    key=lambda x: abs(win_logits[x]))
            self.noswap_threshold = float(win_logits[medium_move_index])
            self._play_move(medium_move_index, (model_label, self.noswap_threshold, sigmoid(self.noswap_threshold)))
            if self.allow_swap:
                self.game_phase = GamePhase.MAY_SWAP
            else:
                self.game_phase = GamePhase.AFTER_SWAP
            return
        if self.game_phase == GamePhase.MAY_SWAP:
            best_move_index = max(range(len(self.valid_moves)),
                                  key=lambda x: win_logits[x])
            best_move_logits = float(win_logits[best_move_index])
            if best_move_logits > self.noswap_threshold:
                self._play_move(best_move_index, (model_label, best_move_logits, sigmoid(best_move_logits)))
                self.swapped = 0
            else:
                self.swapped = 1
            self.game_phase = GamePhase.AFTER_SWAP
            return
        if self.game_phase == GamePhase.AFTER_SWAP:
            best_move_index = max(range(len(self.valid_moves)),
                                  key=lambda x: win_logits[x])
            best_move_logits = float(win_logits[best_move_index])
            self._play_move(best_move_index, (model_label, best_move_logits, sigmoid(best_move_logits)))
            if self.board.winner is not None:
                self.game_phase = GamePhase.FINISHED
            return
        if self.game_phase == GamePhase.FINISHED:
            return

        raise Exception("Unrecognized game phase.")


def make_games(modelA, modelB, games_required, num_initial_moves, batch_size=3, allow_swap=True):
    game_makers = [GameMaker(board_size, num_initial_moves, allow_swap) for _ in range(batch_size)]
    games = []
    start_time = time.time()

    if num_initial_moves % 2 == 0:
        models = [(modelA, "A"), (modelB, "B")]
    else:
        models = [(modelB, "B"), (modelA, "A")]

    while game_makers:

        for model, label in models:
            num_positions_required = sum(g.num_positions_required() for g in game_makers)
            positions = dict((input_names[k],
                              np.zeros([num_positions_required,
                                        board_size + 1,
                                        board_size + 1,
                                        2],
                                       dtype="float32"))
                             for k in itertools.product((0, 1), (False, True)))

            position_counter = 0

            for g in game_makers:
                g.fill_positions(positions,
                                 np.s_[position_counter: position_counter + g.num_positions_required()])
                position_counter += g.num_positions_required()

            win_logits = model.predict(positions)

            position_counter = 0
            for g in game_makers:
                g.update(win_logits[position_counter: position_counter + g.num_positions_required()], label)
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


def add_training_data(moves, winner, num_initial_moves, positions, winners, slice_):
    temp_positions = dict(((player_perspective, flipped), np.copy(initial_position))
                          for player_perspective, flipped in itertools.product((0, 1), (False, True)))

    for i, (move, annotation) in enumerate(moves):
        # update temporary positions
        a, b = move
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a1, b1 = (a, b) if player_perspective == 0 else (b, a)
            a2, b2 = (board_size - a1, board_size - b1) if flipped else (a1 + 1, b1 + 1)
            temp_positions[(player_perspective, flipped)][a2, b2, (i + player_perspective) % 2] = 1

        # update training data
        j = i - num_initial_moves
        if j >= 0:
            for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                positions[input_names[(player_perspective, flipped)]][slice_][j] = \
                    temp_positions[((player_perspective + i) % 2, flipped)]
            winners[j] = winner == i % 2


def make_training_data(model, games_required, num_initial_moves, save_filename=None):
    games = make_games(model, model, games_required, num_initial_moves, allow_swap=False)
    total_moves = sum(len(moves[num_initial_moves:]) for moves, winner, swapped in games)

    positions = dict((input_names[k],
                      np.zeros([total_moves,
                                board_size + 1,
                                board_size + 1,
                                2],
                               dtype="float32"))
                     for k in itertools.product((0, 1), (False, True)))

    winners = np.zeros(total_moves, dtype="float32")

    total_moves_counter = 0

    while games:
        if len(games) % 1000 == 0:
            print(len(games), "more games to process.")
        moves, winner, swapped = games.pop()
        counter_diff = len(moves[num_initial_moves:])
        slice_ = np.s_[total_moves_counter: total_moves_counter + counter_diff]
        add_training_data(moves, winner, num_initial_moves,
                          positions,
                          winners[slice_],
                          slice_)
        # total_moves_counter += len(moves)
        total_moves_counter += counter_diff

    if save_filename is not None:
        np.savez("../data/%s" % save_filename,
                 winners=winners,
                 **dict((input_names[k], v) for k, v in positions.items()))

    return positions, winners

# for i in range(total_moves * 2):
#     print(positions_array[i, :, :, 0])
#     print(positions_array[i, :, :, 1])
#     print("------------------------------")

# np.savez("training_data5.npz",
#          positions=positions_array,
#          winners=winners_array)
