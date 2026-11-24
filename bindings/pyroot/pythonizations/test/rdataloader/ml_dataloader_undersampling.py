import math
from random import randrange, uniform

import numpy as np
import pytest
import ROOT
from conftest import big_data, from_files, squares, vectors, weighted


class TestDataLoaderRandomUndersampling:
    # default constants
    n_train_batch = 4
    n_val_batch = 3
    train_remainder = 1

    @pytest.fixture
    def df_major(self):
        """The majority class, 20 entries: b1 = 0, 2, ..., 38 and b2 = b1 * b1."""
        return from_files(squares(20, "(int) 2 * rdfentry_"))

    @pytest.fixture
    def df_minor(self):
        """The minority class, 5 entries: b1 = 1, 3, ..., 9 and b2 = b1 * b1."""
        return from_files(squares(5, "(int) 2 * rdfentry_ + 1"))

    @pytest.fixture
    def weighted_df_major(self):
        """The majority class, 20 entries: b1 = 0, 2, ..., 38, b2 = b1 * b1, b3 = b1 * 10 and b4 = b3 * 10."""
        return from_files(weighted(20, "(int) 2 * rdfentry_"))

    @pytest.fixture
    def weighted_df_minor(self):
        """The minority class, 5 entries: b1 = 1, 3, ..., 9, b2 = b1 * b1, b3 = b1 * 10 and b4 = b3 * 10."""
        return from_files(weighted(5, "(int) 2 * rdfentry_ + 1"))

    def test01_each_element_is_generated_unshuffled(self, df_major, df_minor):
        major_entries_before = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
        minor_entries_before = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
        results_x_val = [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]
        results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 1.0, 9.0, 25.0]
        results_y_val = [576.0, 676.0, 784.0, 900.0, 49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(train_iter)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        num_major_train = sum(np.array(flat_x_train) % 2 == 0)
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = sum(np.array(flat_x_val) % 2 == 0)
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        # check if there are no duplicate entries (replacement=False)
        assert len(set(flat_x_train)) == len(flat_x_train)
        assert len(set(flat_x_val)) == len(flat_x_val)
        assert len(set(flat_y_train)) == len(flat_y_train)
        assert len(set(flat_y_val)) == len(flat_y_val)

        # check if correct sampling_ratio (0.5 = minor/major)
        assert num_major_train == 6
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2

        major_entries_after = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
        minor_entries_after = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

        # check if the dataframes are correctly reset
        assert np.array_equal(major_entries_before, major_entries_after)
        assert np.array_equal(minor_entries_before, minor_entries_after)

    def test01_each_element_is_generated_unshuffled_replacement(self, df_major, df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.2,
            replacement=True,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            0.0,
            2.0,
            4.0,
            1.0,
            3.0,
            5.0,
        ]
        results_x_val = [24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 24.0, 26.0, 7.0, 9.0]
        results_y_train = [
            0.0,
            4.0,
            16.0,
            36.0,
            64.0,
            100.0,
            144.0,
            196.0,
            256.0,
            324.0,
            400.0,
            484.0,
            0.0,
            4.0,
            16.0,
            1.0,
            9.0,
            25.0,
        ]
        results_y_val = [576.0, 676.0, 784.0, 900.0, 1024.0, 1156.0, 1296.0, 1444.0, 576.0, 676.0, 49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(6):
            x, y = next(val_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(9):
            x, y = next(train_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        num_major_train = sum(np.array(flat_x_train) % 2 == 0)
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = sum(np.array(flat_x_val) % 2 == 0)
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        # check if there are duplicate entries (replacement=True)
        assert len(set(flat_x_train)) < len(flat_x_train)
        assert len(set(flat_x_val)) < len(flat_x_val)
        assert len(set(flat_y_train)) < len(flat_y_train)
        assert len(set(flat_y_val)) < len(flat_y_val)

        # check if correct sampling_ratio (0.2 = minor/major)
        assert num_major_train == 15
        assert num_minor_train == 3
        assert num_major_val == 10
        assert num_minor_val == 2

    def test02_each_element_is_generated_shuffled(self, df_major, df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(train_iter)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert len(flat_x_train) == 9
        assert len(flat_x_val) == 6
        assert len(flat_y_train) == 9
        assert len(flat_y_val) == 6

    def test04_dropping_remainder(self, df_major, df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        collected_x = []
        collected_y = []

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x.append(x)
            collected_y.append(y)

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x.append(x)
            collected_y.append(y)

        assert len(collected_x) == 7
        assert len(collected_y) == 7

    def test05_more_than_one_file(self, df_minor):
        df_major = from_files(squares(20, "(int) 2 * rdfentry_"), squares(5, "(int) 2 * (rdfentry_ + 20)"))

        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
        results_x_val = [30.0, 32.0, 34.0, 36.0, 7.0, 9.0]
        results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 1.0, 9.0, 25.0]
        results_y_val = [900.0, 1024.0, 1156.0, 1296.0, 49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        iter_train = iter(gen_train.as_numpy())
        iter_val = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y = next(iter_train)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for _ in range(self.n_val_batch):
            x, y = next(iter_val)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        x, y = next(iter_train)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

    def test06_multiple_target_columns(self, weighted_df_major, weighted_df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df_minor, weighted_df_major],
            batch_size=2,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
        results_x_val = [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]
        results_y_train = [
            0.0,
            0.0,
            4.0,
            200.0,
            16.0,
            400.0,
            36.0,
            600.0,
            64.0,
            800.0,
            100.0,
            1000.0,
            1.0,
            100.0,
            9.0,
            300.0,
            25.0,
            500.0,
        ]
        results_y_val = [576.0, 2400.0, 676.0, 2600.0, 784.0, 2800.0, 900.0, 3000.0, 49.0, 700.0, 81.0, 900.0]
        results_z_train = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 10.0, 30.0, 50.0]
        results_z_val = [240.0, 260.0, 280.0, 300.0, 70.0, 90.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []
        collected_z_train = []
        collected_z_val = []

        iter_train = iter(gen_train.as_numpy())
        iter_val = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y, z = next(iter_train)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

        x, y, z = next(iter_train)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 2)
        assert z.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())
        collected_z_train.append(z.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
        flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
        flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val
        assert results_z_train == flat_z_train
        assert results_z_val == flat_z_val

        num_major_train = sum(np.array(flat_x_train) % 2 == 0)
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = sum(np.array(flat_x_val) % 2 == 0)
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        assert num_major_train == 6
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2

    def test07_multiple_input_columns(self):
        df_major = from_files(
            ROOT
            .RDataFrame(20)
            .Define("b1", "(int) 2 * rdfentry_")
            .Define("b2", "(int) b1 * b1")
            .Define("b3", "(double) b1 * 10")
        )
        df_minor = from_files(
            ROOT
            .RDataFrame(5)
            .Define("b1", "(int) 2 * rdfentry_ + 1")
            .Define("b2", "(int) b1 * b1")
            .Define("b3", "(double) b1 * 10")
        )

        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [
            0.0,
            0.0,
            2.0,
            20.0,
            4.0,
            40.0,
            6.0,
            60.0,
            8.0,
            80.0,
            10.0,
            100.0,
            1.0,
            10.0,
            3.0,
            30.0,
            5.0,
            50.0,
        ]
        results_x_val = [24.0, 240.0, 26.0, 260.0, 28.0, 280.0, 30.0, 300.0, 7.0, 70.0, 9.0, 90.0]
        results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 1.0, 9.0, 25.0]
        results_y_val = [576.0, 676.0, 784.0, 900.0, 49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        iter_train = iter(gen_train.as_numpy())
        iter_val = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y = next(iter_train)
            assert x.shape == (2, 2)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for _ in range(self.n_val_batch):
            x, y = next(iter_val)
            assert x.shape == (2, 2)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        x, y = next(iter_train)
        assert x.shape == (self.train_remainder, 2)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

    def test08_filtered(self, df_major, df_minor):
        df_major_filter = df_major.Filter("b1 % 8 != 0", "name")

        major_filter_entries_before = df_major_filter.AsNumpy(["rdfentry_"])["rdfentry_"]
        minor_entries_before = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major_filter, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [2.0, 4.0, 6.0, 10.0, 12.0, 14.0, 1.0, 3.0, 5.0]
        results_x_val = [26.0, 28.0, 30.0, 34.0, 7.0, 9.0]
        results_y_train = [4.0, 16.0, 36.0, 100.0, 144.0, 196.0, 1.0, 9.0, 25.0]
        results_y_val = [676.0, 784.0, 900.0, 1156.0, 49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (2, 1)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        x, y = next(train_iter)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        num_major_train = np.sum((np.asarray(flat_x_train) % 2 == 0) & (np.asarray(flat_x_train) % 8 != 0))
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = np.sum((np.asarray(flat_x_val) % 2 == 0) & (np.asarray(flat_x_val) % 8 != 0))
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        assert num_major_train == 6
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2

        major_filter_entries_after = df_major_filter.AsNumpy(["rdfentry_"])["rdfentry_"]
        minor_entries_after = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

        # check if the dataframes are correctly reset
        assert np.array_equal(major_filter_entries_before, major_filter_entries_after)
        assert np.array_equal(minor_entries_before, minor_entries_after)

    def test10_two_epochs_shuffled(self, df_major, df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        both_epochs_collected_x_val = []
        both_epochs_collected_y_val = []

        for _ in range(2):
            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train.as_numpy())
            iter_val = iter(gen_validation.as_numpy())

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                assert x.shape == (2, 1)
                assert y.shape == (2, 1)
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                assert x.shape == (2, 1)
                assert y.shape == (2, 1)
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            assert x.shape == (self.train_remainder, 1)
            assert y.shape == (self.train_remainder, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            assert len(flat_x_train) == 9
            assert len(flat_x_val) == 6
            assert len(flat_y_train) == 9
            assert len(flat_y_val) == 6

            both_epochs_collected_x_val.append(collected_x_val)
            both_epochs_collected_y_val.append(collected_y_val)

        assert both_epochs_collected_x_val[0] == both_epochs_collected_x_val[1]
        assert both_epochs_collected_y_val[0] == both_epochs_collected_y_val[1]

    def test11_number_of_training_and_validation_batches_remainder(self, df_major, df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        number_of_training_batches = 0
        number_of_validation_batches = 0

        for _ in gen_train.as_numpy():
            number_of_training_batches += 1

        for _ in gen_validation.as_numpy():
            number_of_validation_batches += 1

        assert gen_train.num_batches == number_of_training_batches
        assert gen_validation.num_batches == number_of_validation_batches
        assert gen_train.last_batch_no_of_rows == 1
        assert gen_validation.last_batch_no_of_rows == 0

    def test12_PyTorch(self, weighted_df_major, weighted_df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df_major, weighted_df_minor],
            batch_size=2,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
        results_x_val = [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]
        results_y_train = [
            0.0,
            0.0,
            4.0,
            200.0,
            16.0,
            400.0,
            36.0,
            600.0,
            64.0,
            800.0,
            100.0,
            1000.0,
            1.0,
            100.0,
            9.0,
            300.0,
            25.0,
            500.0,
        ]
        results_y_val = [576.0, 2400.0, 676.0, 2600.0, 784.0, 2800.0, 900.0, 3000.0, 49.0, 700.0, 81.0, 900.0]
        results_z_train = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 10.0, 30.0, 50.0]
        results_z_val = [240.0, 260.0, 280.0, 300.0, 70.0, 90.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []
        collected_z_train = []
        collected_z_val = []

        iter_train = iter(gen_train.as_torch(device="cpu"))
        iter_val = iter(gen_validation.as_torch())

        for _ in range(self.n_train_batch):
            x, y, z = next(iter_train)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

        x, y, z = next(iter_train)
        assert x.shape == (self.train_remainder, 1)
        assert y.shape == (self.train_remainder, 2)
        assert z.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())
        collected_z_train.append(z.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
        flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
        flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val
        assert results_z_train == flat_z_train
        assert results_z_val == flat_z_val

        num_major_train = sum(np.array(flat_x_train) % 2 == 0)
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = sum(np.array(flat_x_val) % 2 == 0)
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        assert num_major_train == 6
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2

    def test13_TensorFlow(self, weighted_df_major, weighted_df_minor):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df_major, weighted_df_minor],
            batch_size=2,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 0.0]
        results_x_val = [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]
        results_y_train = [
            0.0,
            0.0,
            4.0,
            200.0,
            16.0,
            400.0,
            36.0,
            600.0,
            64.0,
            800.0,
            100.0,
            1000.0,
            1.0,
            100.0,
            9.0,
            300.0,
            25.0,
            500.0,
            0.0,
            0.0,
        ]
        results_y_val = [576.0, 2400.0, 676.0, 2600.0, 784.0, 2800.0, 900.0, 3000.0, 49.0, 700.0, 81.0, 900.0]
        results_z_train = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 10.0, 30.0, 50.0, 0.0]
        results_z_val = [240.0, 260.0, 280.0, 300.0, 70.0, 90.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []
        collected_z_train = []
        collected_z_val = []

        iter_train = iter(gen_train.as_tensorflow())
        iter_val = iter(gen_validation.as_tensorflow())

        for _ in range(self.n_train_batch):
            x, y, z = next(iter_train)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_train.append(x.numpy().tolist())
            collected_y_train.append(y.numpy().tolist())
            collected_z_train.append(z.numpy().tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (2, 1)
            assert y.shape == (2, 2)
            assert z.shape == (2, 1)
            collected_x_val.append(x.numpy().tolist())
            collected_y_val.append(y.numpy().tolist())
            collected_z_val.append(z.numpy().tolist())

        x, y, z = next(iter_train)
        assert x.shape == (2, 1)
        assert y.shape == (2, 2)
        assert z.shape == (2, 1)
        collected_x_train.append(x.numpy().tolist())
        collected_y_train.append(y.numpy().tolist())
        collected_z_train.append(z.numpy().tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
        flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
        flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val
        assert results_z_train == flat_z_train
        assert results_z_val == flat_z_val

        num_major_train = sum(np.array(flat_x_train) % 2 == 0)
        num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
        num_major_val = sum(np.array(flat_x_val) % 2 == 0)
        num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

        assert num_major_train == 7  # num_major_train = 6 + 1 padded 0 from the last batch
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2

    def test14_big_data_replacement_false(self):
        entries_in_rdf_major = randrange(10000, 30000)
        entries_in_rdf_minor = randrange(8000, 9999)
        batch_size = randrange(100, 501)
        min_allowed_sampling_ratio = entries_in_rdf_minor / entries_in_rdf_major
        sampling_ratio = math.ceil(uniform(min_allowed_sampling_ratio, 2) * 100) / 100

        error_message = f"\n Batch size: {batch_size}\
            Number of major entries: {entries_in_rdf_major} \
            Number of minor entries: {entries_in_rdf_minor}"

        def test(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            df1 = from_files(big_data(num_of_entries_major, "(int) 2 * rdfentry_"))
            df2 = from_files(big_data(num_of_entries_minor, "(int) 2 * rdfentry_ + 1"))

            dl = ROOT.Experimental.ML.RDataLoader(
                [df1, df2],
                batch_size=size_of_batch,
                target=["b3", "b5"],
                weights="b1",
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=sampling_ratio,
                replacement=False,
            )

            gen_train, gen_validation = dl.train_test_split(0.3)

            collected_z_train = []
            collected_z_val = []

            train_remainder = gen_train.last_batch_no_of_rows
            val_remainder = gen_validation.last_batch_no_of_rows

            n_train_batches = gen_train.num_batches - 1 if train_remainder else gen_train.num_batches
            n_val_batches = gen_validation.num_batches - 1 if val_remainder else gen_validation.num_batches

            iter_train = iter(gen_train.as_numpy())
            iter_val = iter(gen_validation.as_numpy())

            for i in range(n_train_batches):
                x, y, z = next(iter_train)

                assert x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}"
                assert y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}"
                assert z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}"
                collected_z_train.append(z.tolist())

            if train_remainder:
                x, y, z = next(iter_train)
                assert x.shape == (train_remainder, 2), error_message
                assert y.shape == (train_remainder, 2), error_message
                assert z.shape == (train_remainder, 1), error_message
                collected_z_train.append(z.tolist())

            for _ in range(n_val_batches):
                x, y, z = next(iter_val)

                assert x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}"
                assert y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}"
                assert z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}"
                collected_z_val.append(z.tolist())

            if val_remainder:
                x, y, z = next(iter_val)
                assert x.shape == (val_remainder, 2), error_message
                assert y.shape == (val_remainder, 2), error_message
                assert z.shape == (val_remainder, 1), error_message
                collected_z_val.append(z.tolist())

            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            num_major_train = sum(np.array(flat_z_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_z_train) % 2 != 0)
            num_major_val = sum(np.array(flat_z_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_z_val) % 2 != 0)

            # check if there are no duplicate entries (replacement=False)
            assert len(set(flat_z_train)) == len(flat_z_train)
            assert len(set(flat_z_val)) == len(flat_z_val)

            # check if the sampling stategy is correct
            assert round((num_minor_train / num_major_train), 2) == sampling_ratio
            assert round((num_minor_val / num_major_val), 2) == sampling_ratio

        def test_raises(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            df1 = from_files(big_data(num_of_entries_major, "(int) 2 * rdfentry_"))
            df2 = from_files(big_data(num_of_entries_minor, "(int) 2 * rdfentry_ + 1"))

            with pytest.raises(
                Exception,
                match=r"The sampling_ratio is too low: not enough entries in the majority class to sample from.",
            ):
                ROOT.Experimental.ML.RDataLoader(
                    [df1, df2],
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b1",
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="undersampling",
                    sampling_ratio=sampling_ratio,
                    replacement=False,
                ).train_test_split(0.3)

        # test the functionality with a proper sampling ratio
        test(batch_size, entries_in_rdf_major, entries_in_rdf_minor, sampling_ratio)

        bad_sampling_ratio = round(max(min_allowed_sampling_ratio - 0.01, 0.01), 2)
        # test that an error is raised when the sampling ratio is too low
        test_raises(batch_size, entries_in_rdf_major, entries_in_rdf_minor, bad_sampling_ratio)

    def test14_big_data_replacement_true(self):
        entries_in_rdf_major = randrange(10000, 30000)
        entries_in_rdf_minor = randrange(8000, 9999)
        batch_size = randrange(100, 501)

        # max samling strategy to guarantee duplicate sampled entires
        max_sampling_ratio = entries_in_rdf_minor / entries_in_rdf_major
        sampling_ratio = round(uniform(0.01, max_sampling_ratio - 0.01), 2)

        error_message = f"\n Batch size: {batch_size}\
            Number of major entries: {entries_in_rdf_major} \
            Number of minor entries: {entries_in_rdf_minor}"

        def test(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            df1 = from_files(big_data(num_of_entries_major, "(int) 2 * rdfentry_"))
            df2 = from_files(big_data(num_of_entries_minor, "(int) 2 * rdfentry_ + 1"))

            dl = ROOT.Experimental.ML.RDataLoader(
                [df1, df2],
                batch_size=size_of_batch,
                target=["b3", "b5"],
                weights="b1",
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=sampling_ratio,
                replacement=True,
            )

            gen_train, gen_validation = dl.train_test_split(0.3)

            collected_z_train = []
            collected_z_val = []

            train_remainder = gen_train.last_batch_no_of_rows
            val_remainder = gen_validation.last_batch_no_of_rows

            n_train_batches = gen_train.num_batches - 1 if train_remainder else gen_train.num_batches
            n_val_batches = gen_validation.num_batches - 1 if val_remainder else gen_validation.num_batches

            iter_train = iter(gen_train.as_numpy())
            iter_val = iter(gen_validation.as_numpy())

            for i in range(n_train_batches):
                x, y, z = next(iter_train)

                assert x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}"
                assert y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}"
                assert z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}"
                collected_z_train.append(z.tolist())

            if train_remainder:
                x, y, z = next(iter_train)
                assert x.shape == (train_remainder, 2), error_message
                assert y.shape == (train_remainder, 2), error_message
                assert z.shape == (train_remainder, 1), error_message
                collected_z_train.append(z.tolist())

            for _ in range(n_val_batches):
                x, y, z = next(iter_val)

                assert x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}"
                assert y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}"
                assert z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}"
                collected_z_val.append(z.tolist())

            if val_remainder:
                x, y, z = next(iter_val)
                assert x.shape == (val_remainder, 2), error_message
                assert y.shape == (val_remainder, 2), error_message
                assert z.shape == (val_remainder, 1), error_message
                collected_z_val.append(z.tolist())

            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            num_major_train = sum(np.array(flat_z_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_z_train) % 2 != 0)
            num_major_val = sum(np.array(flat_z_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_z_val) % 2 != 0)

            # check if there are duplicate entries (replacement=True)
            assert len(set(flat_z_train)) < len(flat_z_train)
            assert len(set(flat_z_val)) < len(flat_z_val)

            # check if the sampling stategy is correct
            assert round((num_minor_train / num_major_train), 2) == sampling_ratio
            assert round((num_minor_val / num_major_val), 2) == sampling_ratio

        test(batch_size, entries_in_rdf_major, entries_in_rdf_minor, sampling_ratio)

    def test15_two_runs_set_seed(self, df_major, df_minor):
        both_runs_collected_x_val = []
        both_runs_collected_y_val = []

        for _ in range(2):
            dl = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            gen_train, gen_validation = dl.train_test_split(0.4)

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train.as_numpy())
            iter_val = iter(gen_validation.as_numpy())

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                assert x.shape == (2, 1)
                assert y.shape == (2, 1)
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                assert x.shape == (2, 1)
                assert y.shape == (2, 1)
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            assert x.shape == (self.train_remainder, 1)
            assert y.shape == (self.train_remainder, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            assert len(flat_x_train) == 9
            assert len(flat_x_val) == 6
            assert len(flat_y_train) == 9
            assert len(flat_y_val) == 6

            both_runs_collected_x_val.append(collected_x_val)
            both_runs_collected_y_val.append(collected_y_val)
        assert both_runs_collected_x_val[0] == both_runs_collected_x_val[1]
        assert both_runs_collected_y_val[0] == both_runs_collected_y_val[1]

    def test16_vector_padding(self):
        df_major = from_files(vectors(20))
        df_minor = from_files(vectors(5, "(int) rdfentry_ + 20"))
        max_vec_sizes = {"v1": 3, "v2": 2}

        dl = ROOT.Experimental.ML.RDataLoader(
            [df_major, df_minor],
            batch_size=2,
            target="b1",
            max_vec_sizes=max_vec_sizes,
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
            sampling_type="undersampling",
            sampling_ratio=0.5,
            replacement=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        results_x_train = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            10.0,
            0.0,
            100.0,
            1000.0,
            2.0,
            20.0,
            0.0,
            200.0,
            2000.0,
            3.0,
            30.0,
            0.0,
            300.0,
            3000.0,
            4.0,
            40.0,
            0.0,
            400.0,
            4000.0,
            5.0,
            50.0,
            0.0,
            500.0,
            5000.0,
            20.0,
            200.0,
            0.0,
            2000.0,
            20000.0,
            21.0,
            210.0,
            0.0,
            2100.0,
            21000.0,
            22.0,
            220.0,
            0.0,
            2200.0,
            22000.0,
        ]
        results_y_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 21.0, 22.0]
        results_x_val = [
            12.0,
            120.0,
            0.0,
            1200.0,
            12000.0,
            13.0,
            130.0,
            0.0,
            1300.0,
            13000.0,
            14.0,
            140.0,
            0.0,
            1400.0,
            14000.0,
            15.0,
            150.0,
            0.0,
            1500.0,
            15000.0,
            23.0,
            230.0,
            0.0,
            2300.0,
            23000.0,
            24.0,
            240.0,
            0.0,
            2400.0,
            24000.0,
        ]
        results_y_val = [12.0, 13.0, 14.0, 15.0, 23.0, 24.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (2, 5)
            assert y.shape == (2, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (2, 5)
            assert y.shape == (2, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(train_iter)
        assert x.shape == (self.train_remainder, 5)
        assert y.shape == (self.train_remainder, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        num_major_train = sum(np.array(flat_y_train) < 20)
        num_minor_train = sum(np.array(flat_y_train) >= 20)
        num_major_val = sum(np.array(flat_y_val) < 20)
        num_minor_val = sum(np.array(flat_y_val) >= 20)

        assert num_major_train == 6
        assert num_minor_train == 3
        assert num_major_val == 4
        assert num_minor_val == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
