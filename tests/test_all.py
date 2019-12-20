import unittest
from hexAI.model import get_main_model, get_new_model
from hexAI.main import compare_models


class TestAll(unittest.TestCase):
    def test_saved_model(self):
        good_model = get_main_model()
        bad_model = get_new_model()
        self.assertEqual(compare_models(good_model, bad_model, 10), 1)
        self.assertEqual(compare_models(bad_model, good_model, 10), 0)
