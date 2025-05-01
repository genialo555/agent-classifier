import unittest
import importlib.util
import pathlib
import sys

# Dynamically load the normalisation script (filename begins with digits)
SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / '01_collect_normalize.py'

spec = importlib.util.spec_from_file_location('collect_normalize', SCRIPT_PATH)
collect_normalize = importlib.util.module_from_spec(spec)
sys.modules['collect_normalize'] = collect_normalize
spec.loader.exec_module(collect_normalize)


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(collect_normalize.normalize_text('Énergie'), 'energie')
        self.assertEqual(collect_normalize.normalize_text('Café'), 'cafe')
        self.assertEqual(collect_normalize.normalize_text('Déjà Vu'), 'deja vu')


if __name__ == '__main__':
    unittest.main() 