import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from run_five import aggregate


class FiveRunAggregationTest(unittest.TestCase):
    def test_mean_and_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, value in enumerate((0.80, 0.81, 0.82, 0.83, 0.84)):
                path = root / ('run{}.json'.format(index))
                path.write_text(json.dumps({
                    'status': 'complete',
                    'best_epoch': index,
                    'best_test': {'hits1': value, 'hits10': value + 0.1},
                    'protocol': {'selection_protocol': 'test_best'},
                }), encoding='utf-8')
                paths.append(path)
            output = root / 'summary.json'
            report = aggregate(paths, output, {'profile': 'paper'})
            self.assertTrue(output.exists())
            self.assertAlmostEqual(report['best_test_hits1']['mean'], 0.82)
            self.assertAlmostEqual(
                report['best_test_hits1']['sample_std'], math.sqrt(0.001 / 4)
            )
            self.assertEqual(report['standard_deviation'], 'sample (N-1 denominator)')


if __name__ == '__main__':
    unittest.main()
