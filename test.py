"""
person 3 - comprehensive test suite for the brfss regression project.

goes beyond the part 1 / part 2 test files:
  - numpy verification across many shapes, sizes and column subsets
  - streaming properties (one-pass == batch, chunked == one-shot)
  - edge cases: bad inputs, missing files, empty data, singular systems,
    boundary sample sizes, invalid field codes, require_all on/off
  - real-data verification when the BRFSS zips are present (BRFSS_DATA_DIR)

run:  python -m unittest test_person3 -v
"""

import math
import os
import tempfile
import types
import unittest
import zipfile

import numpy as np

import data_processing as dp
import regression_part2 as rp
from test_data_processing import make_record, write_fake_zips, find_real_data_dir


# ---------------------------------------------------------------------------
# reference implementation: plain numpy lstsq on materialized arrays
# ---------------------------------------------------------------------------

def numpy_reference(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xi = np.column_stack((np.ones(len(X)), X))
    beta = np.linalg.lstsq(Xi, y, rcond=None)[0]
    res = y - Xi @ beta
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    n, k1 = Xi.shape
    adj = 1 - (1 - r2) * (n - 1) / (n - k1)
    return beta, adj


def stats_from(X, y, k):
    return rp.accumulate_stats(zip(np.asarray(X, dtype=float).tolist(),
                                   np.asarray(y, dtype=float).tolist()), k)


# ---------------------------------------------------------------------------
# 1. numpy verification across shapes and sizes
# ---------------------------------------------------------------------------

class TestNumpyVerification(unittest.TestCase):

    def test_many_shapes_match_numpy(self):
        rng = np.random.default_rng(42)
        for n, k in [(50, 1), (100, 2), (500, 3), (1000, 5), (37, 3)]:
            with self.subTest(n=n, k=k):
                X = rng.normal(size=(n, k)) * rng.integers(1, 5, size=k)
                y = X @ rng.normal(size=k) + 2.5 + rng.normal(size=n)
                stats = stats_from(X, y, k)
                beta = rp.compute_beta(stats)
                ref_beta, ref_adj = numpy_reference(X, y)
                np.testing.assert_allclose(beta, ref_beta, rtol=1e-8, atol=1e-10)
                self.assertAlmostEqual(rp.compute_adj_r2(stats, beta), ref_adj, places=8)

    def test_large_scale_streaming_matches_numpy(self):
        # 200k rows fed as a generator - the one-pass sums must match batch numpy
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200_000, 3))
        y = X @ [0.3, -1.2, 2.0] + 5 + rng.normal(size=200_000)
        stats = rp.accumulate_stats(((row, val) for row, val in zip(X.tolist(), y.tolist())), 3)
        beta = rp.compute_beta(stats)
        ref_beta, ref_adj = numpy_reference(X, y)
        np.testing.assert_allclose(beta, ref_beta, rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(rp.compute_adj_r2(stats, beta), ref_adj, places=7)
        self.assertEqual(stats["n"], 200_000)

    def test_chunked_updates_equal_one_shot(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(1000, 2))
        y = rng.normal(size=1000)
        whole = stats_from(X, y, 2)
        chunked = rp.init_stats(2)
        for i in range(0, 1000, 100):
            for row, val in zip(X[i:i + 100].tolist(), y[i:i + 100].tolist()):
                rp.update_stats(chunked, row, val)
        np.testing.assert_allclose(chunked["A"], whole["A"])
        np.testing.assert_allclose(chunked["z"], whole["z"])
        self.assertEqual(chunked["n"], whole["n"])

    def test_single_predictor_matches_closed_form(self):
        # slope = cov(x, y) / var(x), intercept = ybar - slope * xbar
        rng = np.random.default_rng(11)
        x = rng.normal(10, 3, size=400)
        y = 1.5 * x + rng.normal(size=400)
        stats = stats_from(x.reshape(-1, 1), y, 1)
        beta = rp.compute_beta(stats)
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        np.testing.assert_allclose(beta, [y.mean() - slope * x.mean(), slope], rtol=1e-10)

    def test_column_order_permutation(self):
        # permuting columns must permute beta the same way and leave adj R^2 unchanged
        rng = np.random.default_rng(5)
        X = rng.normal(size=(800, 3))
        y = X @ [1.0, 2.0, 3.0] + rng.normal(size=800)
        s1 = stats_from(X, y, 3)
        s2 = stats_from(X[:, [2, 0, 1]], y, 3)
        b1, b2 = rp.compute_beta(s1), rp.compute_beta(s2)
        np.testing.assert_allclose(b2, [b1[0], b1[3], b1[1], b1[2]], rtol=1e-10)
        self.assertAlmostEqual(rp.compute_adj_r2(s1, b1), rp.compute_adj_r2(s2, b2), places=10)


# ---------------------------------------------------------------------------
# 2. end-to-end on fake BRFSS zips: every column subset, both require_all modes
# ---------------------------------------------------------------------------

def build_fake_dataset(folder, seed=9, per_year=300):
    """returns (X_by_columns, y) of the records that SHOULD survive filtering."""
    rng = np.random.default_rng(seed)
    recs = {y_: [] for y_ in dp.YEARS}
    rows, ys = [], []
    for i in range(per_year * 4):
        year = dp.YEARS[i % 4]
        e = int(rng.integers(1, 7))          # education 1..6 (valid)
        inc = int(rng.integers(1, 9))       # income 1..8 (valid)
        bmi = int(rng.integers(1500, 4500))  # 15.00 .. 44.99
        h = int(rng.integers(1, 6))          # genhlth 1..5
        recs[year].append(make_record(year, genhlth=str(h), education=str(e),
                                      income=f"{inc:02d}", bmi=str(bmi)))
        rows.append({"education": float(e), "income": float(inc), "bmi": bmi * 0.01})
        ys.append(float(h))
    write_fake_zips(folder, recs)
    return rows, ys


class TestSubsetsEndToEnd(unittest.TestCase):

    def test_every_subset_matches_numpy(self):
        subsets = [["education"], ["income"], ["bmi"],
                   ["education", "income"], ["education", "bmi"], ["income", "bmi"],
                   ["education", "income", "bmi"], ["bmi", "education"]]  # order matters
        with tempfile.TemporaryDirectory() as d:
            rows, ys = build_fake_dataset(d)
            for cols in subsets:
                with self.subTest(columns=cols):
                    beta, adj = rp.regression_health(d, cols)
                    X = [[r[c] for c in cols] for r in rows]
                    ref_beta, ref_adj = numpy_reference(X, ys)
                    self.assertEqual(beta.shape, (len(cols) + 1,))
                    np.testing.assert_allclose(beta, ref_beta, rtol=1e-8)
                    self.assertAlmostEqual(adj, ref_adj, places=8)

    def test_require_all_changes_sample(self):
        # one record per year has valid bmi but missing income:
        # included with columns=['bmi'], require_all=False; excluded with True
        recs, clean = {y_: [] for y_ in dp.YEARS}, {y_: [] for y_ in dp.YEARS}
        for year in dp.YEARS:
            for _ in range(20):
                recs[year].append(make_record(year, genhlth="3", education="4",
                                              income="05", bmi="2500"))
                clean[year].append(recs[year][-1])
            recs[year].append(make_record(year, genhlth="3", education="4",
                                          income="  ", bmi="2500"))  # income -> 9
        with tempfile.TemporaryDirectory() as d:
            write_fake_zips(d, recs)
            n_loose = sum(dp.count_by_year(d, ["bmi"]).values())
            n_strict = sum(dp.count_by_year(d, ["bmi"], require_all=True).values())
        self.assertEqual(n_loose, 84)
        self.assertEqual(n_strict, 80)


# ---------------------------------------------------------------------------
# 3. edge cases: inputs and files
# ---------------------------------------------------------------------------

class TestInputEdgeCases(unittest.TestCase):

    def test_columns_wrong_types(self):
        with self.assertRaises(TypeError):
            dp.validate_columns("bmi")            # a string is not a list of columns
        with self.assertRaises(TypeError):
            dp.validate_columns(None)
        with self.assertRaises(ValueError):
            dp.validate_columns([])               # empty not allowed
        with self.assertRaises(ValueError):
            dp.validate_columns(["height"])       # unknown predictor
        with self.assertRaises(ValueError):
            dp.validate_columns(["bmi", "bmi"])   # duplicates
        self.assertEqual(dp.validate_columns(("bmi",)), ["bmi"])  # tuple ok

    def test_missing_directory_and_files(self):
        with self.assertRaises(FileNotFoundError):
            dp.find_zip_files("/no/such/directory")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                dp.find_zip_files(d)              # empty dir -> all four missing
            for year in dp.YEARS:
                self.assertIn(dp.zip_name(year), str(cm.exception))
        with tempfile.TemporaryDirectory() as d:
            # only 2011 present -> error names a missing one
            write_fake_zips(d, {11: [make_record(11)]})
            with self.assertRaises(FileNotFoundError) as cm:
                dp.find_zip_files(d)
            self.assertIn("LLCP2012ASC.ZIP", str(cm.exception))

    def test_case_insensitive_zip_names(self):
        with tempfile.TemporaryDirectory() as d:
            for year in dp.YEARS:
                with zipfile.ZipFile(os.path.join(d, f"llcp20{year}asc.zip"), "w") as zf:
                    zf.writestr(f"LLCP20{year}.ASC", make_record(year))
            self.assertEqual(set(dp.find_zip_files(d)), set(dp.YEARS))


# ---------------------------------------------------------------------------
# 4. edge cases: regression math failure modes
# ---------------------------------------------------------------------------

class TestMathEdgeCases(unittest.TestCase):

    def test_empty_dataset_fails_loudly(self):
        stats = rp.init_stats(3)
        with self.assertRaises((ValueError, np.linalg.LinAlgError)):
            rp.compute_beta(stats)
            rp.compute_adj_r2(stats, np.zeros(4))

    def test_boundary_sample_sizes(self):
        rng = np.random.default_rng(0)
        k = 3
        # n = k + 1 -> adj R^2 undefined (division by n - k - 1 = 0): must raise
        stats = stats_from(rng.normal(size=(k + 1, k)), rng.normal(size=k + 1), k)
        with self.assertRaises(ValueError):
            rp.compute_adj_r2(stats, rp.compute_beta(stats))
        # n = k + 2 -> smallest legal sample, must work and match numpy
        X, y = rng.normal(size=(k + 2, k)), rng.normal(size=k + 2)
        stats = stats_from(X, y, k)
        beta = rp.compute_beta(stats)
        _, ref_adj = numpy_reference(X, y)
        self.assertAlmostEqual(rp.compute_adj_r2(stats, beta), ref_adj, places=8)

    def test_perfect_collinearity_is_singular(self):
        x = np.arange(1.0, 30.0)
        X = np.column_stack((x, 2 * x))          # second predictor = 2 * first
        stats = stats_from(X, 0.5 * x + 1, 2)
        with self.assertRaises(np.linalg.LinAlgError):
            rp.compute_beta(stats)

    def test_constant_y_documents_nan(self):
        # zero variance in y -> SST = 0 -> adj R^2 is nan, beta still solves.
        # documents the behavior instead of crashing somewhere downstream.
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 2))
        stats = stats_from(X, np.full(100, 3.0), 2)
        beta = rp.compute_beta(stats)
        np.testing.assert_allclose(beta, [3.0, 0.0, 0.0], atol=1e-8)
        self.assertTrue(math.isnan(rp.compute_adj_r2(stats, beta)))


# ---------------------------------------------------------------------------
# 5. edge cases: field values and record filtering
# ---------------------------------------------------------------------------

class TestFieldEdgeCases(unittest.TestCase):

    def parse(self, year=13, **kw):
        return dp.parse_record(make_record(year, **kw), year,
                               ["education", "income", "bmi"])

    def test_genhlth_codes(self):
        self.assertIsNotNone(self.parse(genhlth="6"))   # rare but kept
        self.assertIsNone(self.parse(genhlth="7"))      # don't know -> dropped
        self.assertIsNone(self.parse(genhlth="9"))      # refused -> dropped
        self.assertIsNone(self.parse(genhlth=" "))      # blank -> dropped

    def test_education_income_codes(self):
        self.assertIsNone(self.parse(education=" "))    # blank -> 9 -> dropped
        self.assertIsNone(self.parse(income="  "))      # blank -> 9 -> dropped
        self.assertIsNone(self.parse(income="09"))      # refused code 9 -> dropped
        self.assertIsNone(self.parse(income="77"))      # don't know -> dropped
        self.assertIsNotNone(self.parse(income="08", education="6"))

    def test_bmi_boundaries(self):
        self.assertIsNone(self.parse(bmi="9999"))       # missing code -> 0 -> dropped
        self.assertIsNone(self.parse(bmi="9900"))       # 99.00 fails the < 99 rule
        self.assertIsNotNone(self.parse(bmi="9899"))    # 98.99 is kept
        self.assertIsNone(self.parse(bmi="0000"))       # 0 fails the > 0 rule
        got = self.parse(bmi="1234")
        self.assertAlmostEqual(got[0][2], 12.34)

    def test_truncated_and_garbled_lines_skipped(self):
        self.assertIsNone(dp.parse_record("2\n", 13, ["education"]))
        self.assertIsNone(dp.parse_record("x" * 3000 + "\n", 13, ["education"]))


# ---------------------------------------------------------------------------
# 6. streaming design guarantees
# ---------------------------------------------------------------------------

class TestStreamingDesign(unittest.TestCase):

    def test_iterator_is_lazy_generator(self):
        gen = dp.iter_health_records.__wrapped__ if hasattr(dp.iter_health_records, "__wrapped__") else None
        # iter_health_records must not touch the filesystem until iterated:
        lazy = dp.iter_health_records("/no/such/directory", ["bmi"])
        self.assertIsInstance(lazy, types.GeneratorType)
        with self.assertRaises(FileNotFoundError):
            next(lazy)

    def test_zips_stay_zipped_and_no_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            build_fake_dataset(d, per_year=5)
            before = sorted(os.listdir(d))
            rp.regression_health(d, ["education", "income", "bmi"])
            after = sorted(os.listdir(d))
        self.assertEqual(before, after)                    # nothing extracted/added
        self.assertTrue(all(name.endswith(".ZIP") for name in after))


# ---------------------------------------------------------------------------
# 7. real-data verification (runs only when the BRFSS zips are available)
# ---------------------------------------------------------------------------

REAL_DIR = find_real_data_dir()


@unittest.skipUnless(REAL_DIR, "BRFSS data zips not found - skipping real-data verification")
class TestRealData(unittest.TestCase):

    def test_full_model_matches_numpy_on_real_data(self):
        cols = ["education", "income", "bmi"]
        beta, adj = rp.regression_health(REAL_DIR, cols)
        # materialize once, only for verification
        X, y = [], []
        for predictors, health in dp.iter_health_records(REAL_DIR, cols):
            X.append(predictors)
            y.append(health)
        self.assertGreater(len(y), 1_000_000)   # sanity: really the full dataset
        ref_beta, ref_adj = numpy_reference(np.array(X), np.array(y))
        np.testing.assert_allclose(beta, ref_beta, rtol=1e-6)
        self.assertAlmostEqual(adj, ref_adj, places=6)

    def test_counts_by_year_are_plausible(self):
        counts = dp.count_by_year(REAL_DIR, ["education", "income", "bmi"])
        self.assertEqual(set(counts), {2011, 2012, 2013, 2014})
        for year, n in counts.items():
            self.assertGreater(n, 100_000, f"{year} count suspiciously low: {n}")


if __name__ == "__main__":
    unittest.main()
