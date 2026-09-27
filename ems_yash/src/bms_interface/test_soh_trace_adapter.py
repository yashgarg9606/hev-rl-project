"""Synthetic trace tests; run with python -m unittest from ems_yash."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .soh_trace_adapter import SOHTraceAdapter


class SOHTraceAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace_path = Path(directory.name) / "trace.npz"
        self.arrays = {
            "time_seconds": np.array([0.0, 1.0, 3.0]),
            "soh_predicted": np.array([0.95, 0.93, 0.91]),
            "soh_true": np.array([0.96, 0.94, 0.92]),
        }

    def adapter(self, **overrides):
        np.savez(self.trace_path, **(self.arrays | overrides))
        return SOHTraceAdapter(self.trace_path)

    def test_right_hold_boundaries_endpoints_and_backward_queries(self):
        adapter = self.adapter()
        # Intentionally interleave times and reference queries. A forward-only
        # cursor would return a future prediction for the backward queries.
        queries = [(0.0, 0), (1.0, 1), (3.0, 2), (1.5, 1),
                   (-1.0, 0), (100.0, 2), (0.5, 0), (2.99, 1)]
        for time, index in queries:
            with self.subTest(time=time):
                self.assertEqual(adapter.get_soh_at_time(time), self.arrays["soh_predicted"][index])
                self.assertEqual(adapter.current_index, index)
                self.assertEqual(adapter.get_true_soh_at_time(100.0), 0.92)
                self.assertEqual(adapter.get_true_soh_at_time(time), self.arrays["soh_true"][index])

    def test_reset_preserves_arbitrary_time_access(self):
        adapter = self.adapter()
        self.assertEqual(adapter.get_soh_at_time(3.0), 0.91)
        adapter.reset()
        self.assertEqual(adapter.current_index, 0)
        self.assertEqual(adapter.get_soh_at_time(1.0), 0.93)
        self.assertEqual(adapter.get_soh_at_time(0.5), 0.95)

    def test_single_sample_trace(self):
        adapter = self.adapter(time_seconds=[5.0], soh_predicted=[0.95], soh_true=[0.96])
        for time in (-1.0, 5.0, 100.0):
            self.assertEqual(adapter.get_soh_at_time(time), 0.95)
            self.assertEqual(adapter.get_true_soh_at_time(time), 0.96)

    def test_missing_file_and_required_array(self):
        with self.assertRaises(FileNotFoundError):
            SOHTraceAdapter(self.trace_path)
        for missing in self.arrays:
            with self.subTest(missing=missing):
                np.savez(self.trace_path, **{k: v for k, v in self.arrays.items() if k != missing})
                with self.assertRaisesRegex(ValueError, "missing required arrays"):
                    SOHTraceAdapter(self.trace_path)

    def test_empty_and_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.adapter(time_seconds=[], soh_predicted=[], soh_true=[])
        for name in self.arrays:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "same length"):
                self.adapter(**{name: [0.9, 0.8]})

    def test_bad_dimensions(self):
        for name in self.arrays:
            for values in (0.9, [[0.9, 0.8, 0.7]]):
                with self.subTest(name=name, values=values), self.assertRaisesRegex(ValueError, "1-D"):
                    self.adapter(**{name: values})

    def test_non_numeric_complex_and_nonfinite_values(self):
        for name in self.arrays:
            for values, message in ((["a", "b", "c"], "real numeric"),
                                    ([1j, 2j, 3j], "real numeric"),
                                    ([0.9, np.nan, 0.7], "finite"),
                                    ([0.9, np.inf, 0.7], "finite")):
                with self.subTest(name=name, values=values), self.assertRaisesRegex(ValueError, message):
                    self.adapter(**{name: values})

    def test_duplicate_and_decreasing_timestamps(self):
        for times in ([0.0, 1.0, 1.0], [0.0, 3.0, 1.0], np.array([3, 2, 1], dtype=np.uint64)):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "strictly increasing"):
                self.adapter(time_seconds=times)

    def test_nonfinite_query_times(self):
        adapter = self.adapter()
        for method in (adapter.get_soh_at_time, adapter.get_true_soh_at_time):
            for time in (np.nan, np.inf, -np.inf):
                with self.subTest(method=method.__name__, time=time), self.assertRaisesRegex(ValueError, "finite"):
                    method(time)


if __name__ == "__main__":
    unittest.main()
