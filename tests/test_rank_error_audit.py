import unittest

import numpy as np

from scripts.audit_rank_errors import exact_rank_profile, transition_report


class RankErrorAuditTest(unittest.TestCase):
    def test_exact_rank_profile_uses_squared_l2_and_full_candidates(self):
        candidates = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0],
                                 [6.0], [7.0], [8.0], [9.0]], dtype=np.float32)
        queries = np.asarray([[0.1], [9.1]], dtype=np.float32)
        target = np.asarray([0, 0], dtype=np.int64)
        report, private = exact_rank_profile(queries, candidates, target, max_rank=10)
        self.assertEqual(private['ranks'].tolist(), [1, 10])
        self.assertEqual(report['hits1'], 0.5)
        self.assertEqual(report['hits10'], 1.0)
        self.assertEqual(report['rank_bins']['rank_1'], 1)
        self.assertEqual(report['rank_bins']['rank_6_10'], 1)

    def test_transition_report_counts_rescued_and_lost_queries(self):
        before = {
            'ranks': np.asarray([1, 11, 3]),
            'hits1': np.asarray([True, False, False]),
            'hits10': np.asarray([True, False, True]),
        }
        after = {
            'ranks': np.asarray([2, 5, 11]),
            'hits1': np.asarray([False, False, False]),
            'hits10': np.asarray([True, True, False]),
        }
        report = transition_report(
            before,
            after,
            np.asarray([100, 101, 102]),
            np.asarray([200, 201, 202]),
        )
        self.assertEqual(report['hits10']['rescued_count'], 1)
        self.assertEqual(report['hits10']['lost_count'], 1)
        self.assertEqual(report['hits10']['rescued_source_entity_ids'], [101])
        self.assertEqual(report['hits1']['lost_target_entity_ids'], [200])

    def test_rejects_topk_smaller_than_hits10(self):
        with self.assertRaisesRegex(ValueError, 'at least 10'):
            exact_rank_profile(
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((10, 2), dtype=np.float32),
                np.asarray([0]),
                max_rank=9,
            )


if __name__ == '__main__':
    unittest.main()
