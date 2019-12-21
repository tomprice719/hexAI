"""
Use trained models to generate games,
by choosing whatever move leads to the board position with the highest win probability.

Games are represented as triples (moves, winner, swapped).
The winner is represented with an element of the Player enum.
See https://en.wikipedia.org/wiki/Pie_rule for an explanation of "swapped."

Moves are represented as list of (point, annotation) tuples.
The annotation gives some additional data about the move that could be useful for debugging.

The models will typically be deterministic, so to keep the games diverse, the first few moves are randomized.
The next move after that is chosen to make the winning probability as close as possible to 50 percent.
After this, the pie rule is applied, and then the game continues normally.

Note that the first player and second player do not necessarily correspond to red and blue respectively;
if the players swap in accordance with the pie rule, it will be the opposite.
"""
import numpy as np
from .config import board_size
from .board_utils import Player, opposite_player
from random import randint
from enum import Enum
import math
from .model_input import ArrayBoard, new_model_input, fill_model_input, update_model_input


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class GamePhase(Enum):
    BEFORE_SWAP = 1
    MAY_SWAP = 2
    AFTER_SWAP = 3
    FINISHED = 4


class GameMaker:

    def __init__(self, num_initial_moves, allow_swap):
        self.board = ArrayBoard(board_size)
        self.allow_swap = allow_swap
        self.current_player = Player.RED
        self.moves_played = []
        self.valid_moves = list(self.board.all_points)
        self.game_phase = GamePhase.BEFORE_SWAP
        self.swapped = None
        for i in range(num_initial_moves):
            self._play_move(randint(0, len(self.valid_moves) - 1))

    def num_positions_required(self):
        if self.game_phase == GamePhase.FINISHED:
            return 0
        else:
            return len(self.valid_moves)

    def finished(self):
        return self.board.winner is not None

    def game(self):
        return self.moves_played, self.board.winner, self.swapped

    def fill_model_input(self, model_input, slice_):
        if self.game_phase == GamePhase.FINISHED:
            return
        else:
            fill_model_input(model_input,
                             self.board.array_position,
                             self.current_player,
                             slice_)
            update_model_input(model_input,
                               self.valid_moves,
                               self.current_player,
                               self.current_player,
                               slice_
                               )

    def _play_move(self, move_index, annotation=None):
        """Plays a move on the board, where the move is specified by its index in valid_moves"""
        move = self.valid_moves[move_index]

        self.board.update(Player(self.current_player), move)
        self.current_player = opposite_player(self.current_player)
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
            if best_move_logits > 0:
                self._play_move(best_move_index, (model_label, best_move_logits, sigmoid(best_move_logits)))
                self.swapped = False
            else:
                self.swapped = True
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


def make_games(model_a, model_b, num_games, num_initial_moves, batch_size=3, allow_swap=True):
    """
    Create a list of games, with moves chosen by trained models.
    Several games can be created simultaneously to make more efficient use of parallel processing.
    See module docstring for more information.

    Parameters:
        model_a: The model used by the first player
        model_b: the model used by the second player
        num_games: the number of games to create in total
        num_initial_moves: the number of moves to play randomly at the beginning of the game.
        batch_size: the number of games to create simultaneously.
        allow_swap: set to True to use pie rule, False otherwise.

    Returns a list of games.
    """
    game_makers = [GameMaker(num_initial_moves, allow_swap) for _ in range(batch_size)]
    games = []

    if num_initial_moves % 2 == 0:
        models = [(model_a, "A"), (model_b, "B")]
    else:
        models = [(model_b, "B"), (model_a, "A")]

    while game_makers:

        for model, label in models:
            model_input = new_model_input(sum(g.num_positions_required() for g in game_makers))

            position_counter = 0

            for g in game_makers:
                g.fill_model_input(model_input,
                                   np.s_[position_counter: position_counter + g.num_positions_required()])
                position_counter += g.num_positions_required()

            win_logits = model.predict(model_input)

            position_counter = 0
            for g in game_makers:
                num_positions_required = g.num_positions_required()
                g.update(win_logits[position_counter: position_counter + num_positions_required], label)
                position_counter += num_positions_required

        games += [g.game() for g in game_makers if g.finished()]
        game_makers = [g for g in game_makers if not g.finished()]
        new_games_required = num_games - len(games) - len(game_makers)
        assert new_games_required >= 0
        game_makers += [GameMaker(num_initial_moves, allow_swap)
                        for _ in range(min(new_games_required, batch_size - len(game_makers)))]

    assert (len(games) == num_games)

    return games
