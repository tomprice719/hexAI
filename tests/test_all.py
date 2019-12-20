import unittest
from hexAI.model import get_main_model, create_model, RandomModel
from hexAI.main import compare_models
from hexAI.main import make_initial_training_data


class TestAll(unittest.TestCase):

    def test_main_model(self):
        good_model = get_main_model()
        bad_model = RandomModel()
        self.assertEqual(compare_models(good_model, bad_model, 10), 1)
        self.assertEqual(compare_models(bad_model, good_model, 10), 0)

    def test_new_model(self):
        model_input, winners, move_numbers = make_initial_training_data(5000)
        good_model = create_model(depth=5, breadth=20)
        good_model.fit(
            model_input,
            winners,
            batch_size=64,
            epochs=1,
            shuffle=True,
            validation_split=0.1
        )
        bad_model = RandomModel()
        self.assertEqual(compare_models(good_model, bad_model, 10), 1)
        self.assertEqual(compare_models(bad_model, good_model, 10), 0)
