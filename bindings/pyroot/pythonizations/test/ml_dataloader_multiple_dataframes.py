from random import randrange

import numpy as np
import pytest
import ROOT
from conftest import big_data, from_files, squares, vectors, weighted


class TestDataLoaderMultipleDataframes:
    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    @pytest.fixture
    def df1(self):
        """5 entries: b1 = 0, ..., 4 and b2 = b1 * b1."""
        return from_files(squares(5))

    @pytest.fixture
    def df2(self):
        """5 entries: b1 = 5, ..., 9 and b2 = b1 * b1."""
        return from_files(squares(5, "(int) rdfentry_ + 5"))

    @pytest.fixture
    def weighted_df1(self):
        """5 entries: b1 = 0, ..., 4, b2 = b1 * b1, b3 = b1 * 10 and b4 = b3 * 10."""
        return from_files(weighted(5))

    @pytest.fixture
    def weighted_df2(self):
        """5 entries: b1 = 5, ..., 9, b2 = b1 * b1, b3 = b1 * 10 and b4 = b3 * 10."""
        return from_files(weighted(5, "(int) rdfentry_ + 5"))

    def test01_each_element_is_generated_unshuffled(self, df1, df2, load_eager):
        df1_entries_before = df1.AsNumpy(["rdfentry_"])["rdfentry_"]
        df2_entries_before = df2.AsNumpy(["rdfentry_"])["rdfentry_"]

        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.0]
            results_y_val = [9.0, 16.0, 64.0, 81.0]
        else:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(val_iter)
        assert x.shape == (self.val_remainder, 1)
        assert y.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        df1_entries_after = df1.AsNumpy(["rdfentry_"])["rdfentry_"]
        df2_entries_after = df2.AsNumpy(["rdfentry_"])["rdfentry_"]

        # check if the dataframes are correctly reset
        assert np.array_equal(df1_entries_before, df1_entries_after)
        assert np.array_equal(df2_entries_before, df2_entries_after)

    def test02_each_element_is_generated_shuffled(self, df1, df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=True,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        x, y = next(val_iter)
        assert x.shape == (self.val_remainder, 1)
        assert y.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())

        flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
        flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
        flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
        flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

        assert len(flat_x_train) == 6
        assert len(flat_x_val) == 4
        assert len(flat_y_train) == 6
        assert len(flat_y_val) == 4

    def test04_dropping_remainder(self, df1, df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=True,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        collected_x = []
        collected_y = []

        for x, y in gen_train.as_numpy():
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x.append(x)
            collected_y.append(y)

        for x, y in gen_validation.as_numpy():
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x.append(x)
            collected_y.append(y)

        assert len(collected_x) == 3
        assert len(collected_y) == 3

    def test05_more_than_one_file(self, load_eager):
        df1 = from_files(squares(5), squares(5, "(int) rdfentry_ + 5"))
        df2 = from_files(squares(5, "(int) rdfentry_ + 10"))

        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 121.0, 144.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0, 169.0, 196.0]
        else:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            results_x_val = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
            results_y_val = [81.0, 100.0, 121.0, 144.0, 169.0, 196.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        for x, y in gen_train.as_numpy():
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for x, y in gen_validation.as_numpy():
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

    def test06_multiple_target_columns(self, weighted_df1, weighted_df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df1, weighted_df2],
            batch_size=3,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0]
        else:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 9.0, 300.0, 16.0, 400.0, 25.0, 500.0]
            results_y_val = [36.0, 600.0, 49.0, 700.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
            results_z_val = [60.0, 70.0, 80.0, 90.0]

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
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

        x, y, z = next(iter_val)
        assert x.shape == (self.val_remainder, 1)
        assert y.shape == (self.val_remainder, 2)
        assert z.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())
        collected_z_val.append(z.tolist())

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

    def test07_multiple_input_columns(self, load_eager):
        df1 = from_files(
            ROOT
            .RDataFrame(5)
            .Define("b1", "(int) rdfentry_")
            .Define("b2", "(int) b1 * b1")
            .Define("b3", "(double) b1 * 10")
        )
        df2 = from_files(
            ROOT
            .RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 5")
            .Define("b2", "(int) b1 * b1")
            .Define("b3", "(double) b1 * 10")
        )

        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0]
            results_x_val = [3.0, 30.0, 4.0, 40.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.0]
            results_y_val = [9.0, 16.0, 64.0, 81.0]
        else:
            results_x_train = [
                0.0,
                0.0,
                1.0,
                10.0,
                2.0,
                20.0,
                3.0,
                30.0,
                4.0,
                40.0,
                5.0,
                50.0,
            ]
            results_x_val = [6.0, 60.0, 7.0, 70.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        iter_train = iter(gen_train.as_numpy())
        iter_val = iter(gen_validation.as_numpy())

        for _ in range(self.n_train_batch):
            x, y = next(iter_train)
            assert x.shape == (3, 2)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        for _ in range(self.n_val_batch):
            x, y = next(iter_val)
            assert x.shape == (3, 2)
            assert y.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        x, y = next(iter_val)
        assert x.shape == (self.val_remainder, 2)
        assert y.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

    def test08_filtered(self, df1, df2, load_eager):
        dff1 = df1.Filter("b1 % 2 == 0", "name")
        dff2 = df2.Filter("b1 % 2 != 0", "name")

        dff1_entries_before = dff1.AsNumpy(["rdfentry_"])["rdfentry_"]
        dff2_entries_before = dff2.AsNumpy(["rdfentry_"])["rdfentry_"]

        dl = ROOT.Experimental.ML.RDataLoader(
            [dff1, dff2],
            batch_size=3 if load_eager else 4,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 2.0, 5.0]
            results_x_val = [4.0, 9.0]
            results_y_train = [0.0, 4.0, 25.0]
            results_y_val = [16.0, 81.0]
        else:
            # dff1 keeps the even entries 0, 2, 4 and dff2 the odd entries 5, 7, 9. Unshuffled, the first
            # 4 of these 6 entries (0, 2, 4, 5) go to training and the last 2 (7, 9) to validation
            results_x_train = [0.0, 2.0, 4.0, 5.0]
            results_x_val = [7.0, 9.0]
            results_y_train = [0.0, 4.0, 16.0, 25.0]
            results_y_val = [49.0, 81.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        x, y = next(train_iter)
        assert x.shape == (len(results_x_train), 1)
        assert y.shape == (len(results_y_train), 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        x, y = next(val_iter)
        assert x.shape == (2, 1)
        assert y.shape == (2, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

        dff1_entries_after = dff1.AsNumpy(["rdfentry_"])["rdfentry_"]
        dff2_entries_after = dff2.AsNumpy(["rdfentry_"])["rdfentry_"]

        # check if the dataframes are correctly reset
        assert np.array_equal(dff1_entries_before, dff1_entries_after)
        assert np.array_equal(dff2_entries_before, dff2_entries_after)

    def test09_filtered_last_chunk(self, load_eager):
        df1 = from_files(ROOT.RDataFrame(10).Define("b1", "(int) rdfentry_").Define("b2", "(UShort_t) b1 * b1"))
        df2 = from_files(ROOT.RDataFrame(10).Define("b1", "(int) rdfentry_ + 10").Define("b2", "(UShort_t) b1 * b1"))

        dff1 = df1.Filter("b1 % 2 == 0", "name")
        dff2 = df2.Filter("b1 % 2 == 0", "name")

        gen_train = ROOT.Experimental.ML.RDataLoader(
            [dff1, dff2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        if load_eager:
            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0]
        else:
            results_x_train = [0.0, 2.0, 4.0, 10.0, 12.0, 14.0, 6.0, 8.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 100.0, 144.0, 196.0, 36.0, 64.0, 256.0, 324.0]

        collected_x_train = []
        collected_y_train = []

        train_iter = iter(gen_train.as_numpy())

        for _ in range(3):
            x, y = next(train_iter)
            assert x.shape == (3, 1)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(train_iter)
        assert x.shape == (1, 1)
        assert y.shape == (1, 1)
        collected_x_train.append(x.tolist())
        collected_y_train.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_y_train == flat_y_train

    def test10_two_epochs_shuffled(self, df1, df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
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
                assert x.shape == (3, 1)
                assert y.shape == (3, 1)
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                assert x.shape == (3, 1)
                assert y.shape == (3, 1)
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_val)
            assert x.shape == (self.val_remainder, 1)
            assert y.shape == (self.val_remainder, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            assert len(flat_x_train) == 6
            assert len(flat_x_val) == 4
            assert len(flat_y_train) == 6
            assert len(flat_y_val) == 4

            both_epochs_collected_x_val.append(collected_x_val)
            both_epochs_collected_y_val.append(collected_y_val)

        assert both_epochs_collected_x_val[0] == both_epochs_collected_x_val[1]
        assert both_epochs_collected_y_val[0] == both_epochs_collected_y_val[1]

    def test11_number_of_training_and_validation_batches_remainder(self, df1, df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
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
        assert gen_train.last_batch_no_of_rows == 0
        assert gen_validation.last_batch_no_of_rows == 1

    def test12_PyTorch(self, weighted_df1, weighted_df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df1, weighted_df2],
            batch_size=3,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0]
        else:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 9.0, 300.0, 16.0, 400.0, 25.0, 500.0]
            results_y_val = [36.0, 600.0, 49.0, 700.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
            results_z_val = [60.0, 70.0, 80.0, 90.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []
        collected_z_train = []
        collected_z_val = []

        iter_train = iter(gen_train.as_torch())
        iter_val = iter(gen_validation.as_torch(device="cpu"))

        for _ in range(self.n_train_batch):
            x, y, z = next(iter_train)
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

        x, y, z = next(iter_val)
        assert x.shape == (self.val_remainder, 1)
        assert y.shape == (self.val_remainder, 2)
        assert z.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())
        collected_z_val.append(z.tolist())

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

    def test13_TensorFlow(self, weighted_df1, weighted_df2, load_eager):
        dl = ROOT.Experimental.ML.RDataLoader(
            [weighted_df1, weighted_df2],
            batch_size=3,
            target=["b2", "b4"],
            weights="b3",
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0, 0.0, 0.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0, 0.0, 0.0, 0.0, 0.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0, 0.0, 0.0]
        else:
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0, 0.0, 0.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 9.0, 300.0, 16.0, 400.0, 25.0, 500.0]
            results_y_val = [36.0, 600.0, 49.0, 700.0, 64.0, 800.0, 81.0, 900.0, 0.0, 0.0, 0.0, 0.0]
            results_z_train = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
            results_z_val = [60.0, 70.0, 80.0, 90.0, 0.0, 0.0]

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
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_train.append(x.numpy().tolist())
            collected_y_train.append(y.numpy().tolist())
            collected_z_train.append(z.numpy().tolist())

        for _ in range(self.n_val_batch):
            x, y, z = next(iter_val)
            assert x.shape == (3, 1)
            assert y.shape == (3, 2)
            assert z.shape == (3, 1)
            collected_x_val.append(x.numpy().tolist())
            collected_y_val.append(y.numpy().tolist())
            collected_z_val.append(z.numpy().tolist())

        x, y, z = next(iter_val)
        assert x.shape == (3, 1)
        assert y.shape == (3, 2)
        assert z.shape == (3, 1)
        collected_x_val.append(x.numpy().tolist())
        collected_y_val.append(y.numpy().tolist())
        collected_z_val.append(z.numpy().tolist())

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

    def test14_big_data(self, load_eager):
        entries_in_rdf = randrange(10000, 30000)
        batch_size = randrange(100, 501)
        batches_in_memory = randrange(5, 20)

        error_message = f"\n batch_size={batch_size} batches_in_memory={batches_in_memory} entries={entries_in_rdf}"

        def test(size_of_batch, batches_in_memory, num_of_entries):
            df1 = from_files(big_data(num_of_entries))
            df2 = from_files(big_data(num_of_entries))

            dl = ROOT.Experimental.ML.RDataLoader(
                [df1, df2],
                batch_size=size_of_batch,
                batches_in_memory=batches_in_memory,
                target=["b3", "b5"],
                weights="b2",
                shuffle=False,
                drop_remainder=False,
                load_eager=load_eager,
            )

            gen_train, gen_validation = dl.train_test_split(0.3)

            collect_x = []

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

                assert np.all(x[:, 0] * (-1) == x[:, 1]), error_message + f" row: {i}"
                assert np.all(x[:, 0] + 10192 == y[:, 0]), error_message + f" row: {i}"
                # assert np.all(x[:,0]*(-1)-10192==y[:,1]), error_message
                assert np.all(x[:, 0] * 2 == z[:, 0]), error_message + f" row: {i}"

                collect_x.extend(list(x[:, 0]))

            if train_remainder:
                x, y, z = next(iter_train)
                assert x.shape == (train_remainder, 2), error_message
                assert y.shape == (train_remainder, 2), error_message
                assert z.shape == (train_remainder, 1), error_message
                collect_x.extend(list(x[:, 0]))

            for _ in range(n_val_batches):
                x, y, z = next(iter_val)

                assert x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}"
                assert y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}"
                assert z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}"

                assert np.all(x[:, 0] * (-1) == x[:, 1]), error_message
                assert np.all(x[:, 0] + 10192 == y[:, 0]), error_message
                # assert np.all(x[:,0]*(-1)-10192==y[:,1]), error_message
                assert np.all(x[:, 0] * 2 == z[:, 0]), error_message

                collect_x.extend(list(x[:, 0]))

            if val_remainder:
                x, y, z = next(iter_val)
                assert x.shape == (val_remainder, 2), error_message
                assert y.shape == (val_remainder, 2), error_message
                assert z.shape == (val_remainder, 1), error_message
                collect_x.extend(list(x[:, 0]))

            assert set(collect_x) == set(i for i in range(num_of_entries)), (
                f"collected length: {len(set(collect_x))}\
                                 generated length {len(set(i for i in range(num_of_entries)))}"
            )

        test(batch_size, batches_in_memory, entries_in_rdf)

    def test15_two_runs_set_seed(self, df1, df2, load_eager):
        both_runs_collected_x_val = []
        both_runs_collected_y_val = []

        for _ in range(2):
            dl = ROOT.Experimental.ML.RDataLoader(
                [df1, df2],
                batch_size=3,
                target="b2",
                shuffle=True,
                drop_remainder=False,
                set_seed=42,
                load_eager=load_eager,
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
                assert x.shape == (3, 1)
                assert y.shape == (3, 1)
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                assert x.shape == (3, 1)
                assert y.shape == (3, 1)
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_val)
            assert x.shape == (self.val_remainder, 1)
            assert y.shape == (self.val_remainder, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            assert len(flat_x_train) == 6
            assert len(flat_x_val) == 4
            assert len(flat_y_train) == 6
            assert len(flat_y_val) == 4

            both_runs_collected_x_val.append(collected_x_val)
            both_runs_collected_y_val.append(collected_y_val)
        assert both_runs_collected_x_val[0] == both_runs_collected_x_val[1]
        assert both_runs_collected_y_val[0] == both_runs_collected_y_val[1]

    def test16_vector_padding(self, load_eager):
        df1 = from_files(vectors(5))
        df2 = from_files(vectors(5, "(int) rdfentry_ + 5"))
        max_vec_sizes = {"v1": 3, "v2": 2}

        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b1",
            max_vec_sizes=max_vec_sizes,
            shuffle=False,
            drop_remainder=False,
            load_eager=load_eager,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        if load_eager:
            results_x_train = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                10.0,
                0,
                100.0,
                1000.0,
                2.0,
                20.0,
                0,
                200.0,
                2000.0,
                5.0,
                50.0,
                0,
                500.0,
                5000.0,
                6.0,
                60.0,
                0.0,
                600.0,
                6000.0,
                7.0,
                70.0,
                0.0,
                700.0,
                7000.0,
            ]
            results_y_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [
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
                8.0,
                80.0,
                0.0,
                800.0,
                8000.0,
                9.0,
                90.0,
                0.0,
                900.0,
                9000.0,
            ]
            results_y_val = [3.0, 4.0, 8.0, 9.0]
        else:
            results_x_train = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                10.0,
                0,
                100.0,
                1000.0,
                2.0,
                20.0,
                0,
                200.0,
                2000.0,
                3.0,
                30.0,
                0,
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
            ]
            results_y_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [
                6.0,
                60.0,
                0.0,
                600.0,
                6000.0,
                7.0,
                70.0,
                0.0,
                700.0,
                7000.0,
                8.0,
                80.0,
                0.0,
                800.0,
                8000.0,
                9.0,
                90.0,
                0.0,
                900.0,
                9000.0,
            ]
            results_y_val = [6.0, 7.0, 8.0, 9.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        train_iter = iter(gen_train.as_numpy())
        val_iter = iter(gen_validation.as_numpy())

        for _ in range(self.n_val_batch):
            x, y = next(val_iter)
            assert x.shape == (3, 5)
            assert y.shape == (3, 1)
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        for _ in range(self.n_train_batch):
            x, y = next(train_iter)
            assert x.shape == (3, 5)
            assert y.shape == (3, 1)
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

        x, y = next(val_iter)
        assert x.shape == (self.val_remainder, 5)
        assert y.shape == (self.val_remainder, 1)
        collected_x_val.append(x.tolist())
        collected_y_val.append(y.tolist())

        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

        assert results_x_train == flat_x_train
        assert results_x_val == flat_x_val
        assert results_y_train == flat_y_train
        assert results_y_val == flat_y_val

    def test17_shuffled_split_varies_with_seed(self, df1, df2):
        dl1 = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=True,
            drop_remainder=False,
            set_seed=42,
        )

        dl2 = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=3,
            target="b2",
            shuffle=True,
            drop_remainder=False,
            set_seed=43,
        )

        _, gen_val1 = dl1.train_test_split(0.4)
        _, gen_val2 = dl2.train_test_split(0.4)

        val_1_collected = sorted([v for x, y in gen_val1.as_numpy() for v in x.flatten().tolist()])
        val_2_collected = sorted([v for x, y in gen_val2.as_numpy() for v in x.flatten().tolist()])

        assert val_1_collected != val_2_collected

    def test17_dataframes_without_cluster_metadata(self):
        # the dataframes are not read from a file, so their entries are counted with Count()
        dl = ROOT.Experimental.ML.RDataLoader(
            [squares(5), squares(5, "(int) rdfentry_ + 5")],
            batch_size=3,
            target="b2",
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
        )
        gen_train, gen_validation = dl.train_test_split(0.4)

        x_train = [v for x, _ in gen_train.as_numpy() for v in x.flatten().tolist()]
        x_val = [v for x, _ in gen_validation.as_numpy() for v in x.flatten().tolist()]
        assert x_train == [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
        assert x_val == [3.0, 4.0, 8.0, 9.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
