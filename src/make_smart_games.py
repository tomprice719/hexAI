# Generates games by choosing moves with maximum win probability according to a trained model

import numpy as np
from utils import input_names, board_size, initial_position
import itertools
from board_utils import Board, Player
from random import randint
from enum import Enum
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class GamePhase(Enum):
    BEFORE_SWAP = 1
    MAY_SWAP = 2
    AFTER_SWAP = 3
    FINISHED = 4


def transform_coordinates(point, player_perspective, flip):
    a, b = point
    a1, b1 = (a, b) if player_perspective == 0 else (b, a)
    return (board_size - a1, board_size - b1) if flip is True else (a1 + 1, b1 + 1)


class GameMaker:

    def __init__(self, board_size, num_initial_moves, allow_swap):
        self.board = Board(board_size)
        self.current_positions = dict()
        self.allow_swap = allow_swap
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            self.current_positions[(player_perspective, flipped)] = np.copy(initial_position)
        self.num_initial_moves = num_initial_moves
        self.refresh()

    def _get_position(self, player_perspective, flipped):
        return self.current_positions[(player_perspective, flipped)]

    def num_positions_required(self):
        if self.game_phase == GamePhase.FINISHED:
            return 0
        else:
            return len(self.valid_moves)

    def finished(self):
        return self.board.winner is not None

    def game(self):
        return self.moves_played, self.board.winner.value, self.swapped

    def refresh(self):
        self.board.refresh()
        self.current_player = 0
        self.moves_played = []
        self.valid_moves = list(self.board.all_points)
        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            np.copyto(self._get_position(player_perspective, flipped), initial_position)
        for i in range(self.num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))
        self.game_phase = GamePhase.BEFORE_SWAP
        self.swapped = None

    def fill_positions(self, hypothetical_positions, slice_):
        if self.game_phase == GamePhase.FINISHED:
            return
        else:
            for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                hypothetical_positions[input_names[((self.current_player + player_perspective) % 2, flipped)]][slice_] = \
                    self._get_position(player_perspective, flipped)

            for i, move in enumerate(self.valid_moves):
                for player_perspective, flipped in itertools.product((0, 1), (False, True)):
                    a, b = transform_coordinates(move, player_perspective, flipped)
                    hypothetical_positions[input_names[((self.current_player + player_perspective) % 2, flipped)]] \
                        [slice_][i, a, b, (self.current_player + player_perspective) % 2] = 1

    def _play_move(self, move_index, annotation=None):
        """Plays a move on the board, where the move is specified by its index in valid_moves"""
        move = self.valid_moves[move_index]

        for player_perspective, flipped in itertools.product((0, 1), (False, True)):
            a, b = transform_coordinates(move, player_perspective, flipped)
            self._get_position(player_perspective, flipped)[a, b, (self.current_player + player_perspective) % 2] = 1

        self.board.update(Player(self.current_player), move)
        self.current_player = 1 - self.current_player
        self.valid_moves[move_index] = self.valid_moves[-1]
        del self.valid_moves[-1]
        self.moves_played.append((move, annotation))

    def update(self, win_logits, model_label):
        if self.game_phase == GamePhase.BEFORE_SWAP:
            medium_move_index = min(range(len(self.valid_moves)),
                                    key=lambda x: abs(win_logits[x]))
            medium_move_logits = float(win_logits[medium_move_index])
            self._play_move(medium_move_index, (model_label, medium_move_logits, sigmoid(medium_move_logits)))
            if self.allow_swap:
                self.game_phase = GamePhase.MAY_SWAP
            else:
                self.game_phase = GamePhase.AFTER_SWAP
            return
        if self.game_phase == GamePhase.MAY_SWAP:
            best_move_index = max(range(len(self.valid_moves)),
                                  key=lambda x: win_logits[x])
            best_move_logits = float(win_logits[best_move_index])
            if best_move_logits > 0.5:
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

        assert False


def make_games(model_a, model_b, games_required, num_initial_moves, batch_size=3, allow_swap=True):
    game_makers = [GameMaker(board_size, num_initial_moves, allow_swap) for _ in range(batch_size)]
    games = []

    if num_initial_moves % 2 == 0:
        models = [(model_a, "A"), (model_b, "B")]
    else:
        models = [(model_b, "B"), (model_a, "A")]

    while game_makers:

        for model, label in models:
            num_positions_required = sum(g.num_positions_required() for g in game_makers)
            hypothetical_positions = dict((input_names[k],
                              np.zeros([num_positions_required,
                                        board_size + 1,
                                        board_size + 1,
                                        2],
                                       dtype="float32"))
                             for k in itertools.product((0, 1), (False, True)))

            position_counter = 0

            for g in game_makers:
                g.fill_positions(hypothetical_positions,
                                 np.s_[position_counter: position_counter + g.num_positions_required()])
                position_counter += g.num_positions_required()

            win_logits = model.predict(hypothetical_positions)

            position_counter = 0
            for g in game_makers:
                num_positions_required = g.num_positions_required()
                g.update(win_logits[position_counter: position_counter + num_positions_required], label)
                position_counter += num_positions_required

        finished_game_makers = [g for g in game_makers if g.finished()]
        game_makers = [g for g in game_makers if not g.finished()]
        games += [g.game() for g in finished_game_makers]
        new_games_required = games_required - len(games) - len(game_makers)
        assert new_games_required >= 0
        finished_game_makers = finished_game_makers[:new_games_required]
        for g in finished_game_makers:
            g.refresh()
        game_makers += finished_game_makers

    return games
