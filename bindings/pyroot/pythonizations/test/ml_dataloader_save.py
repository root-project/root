import numpy as np
import pytest
import ROOT
from conftest import from_files


class TestDataLoaderSave:
    tree_name = "mytree"

    out_name = "dataloader_save_output.root"
    out_name2 = "dataloader_save_output2.root"

    @pytest.fixture
    def df(self):
        """20 entries: b1 = 0, ..., 19 and b2 = b1 + 1."""
        return from_files(self.define_rdf())

    # Helpers
    def define_rdf(self, num_of_entries=20, offset=0):
        return ROOT.RDataFrame(num_of_entries).Define("b1", f"(int) rdfentry_ + {offset}").Define("b2", "b1 + 1")

    def read_back(self, file_name, column):
        """Read back one column of a saved file: a 1D array, or a 2D array for a vector column."""
        values = ROOT.RDataFrame(self.tree_name, file_name).AsNumpy([column])[column]
        # a vector column comes back as one RVec per entry, all of the same padded size
        return np.stack([np.asarray(v) for v in values]) if values.dtype == object else values

    def saved_columns(self, file_name):
        return {str(c) for c in ROOT.RDataFrame(self.tree_name, file_name).GetColumnNames()}

    def test01_unshuffled(self, df):
        df = df.Define("w", "(double) 0.5")

        dl = ROOT.Experimental.ML.RDataLoader(
            df,
            batch_size=5,
            batches_in_memory=2,
            columns=["b1", "b2", "w"],
            target="b2",
            weights="w",
            shuffle=False,
            drop_remainder=False,
        )

        dl.save(self.tree_name, self.out_name)

        # check all features, target and weights
        assert self.saved_columns(self.out_name) == {"b1", "b2", "w"}
        np.testing.assert_array_equal(self.read_back(self.out_name, "b1"), np.arange(20))
        np.testing.assert_array_equal(self.read_back(self.out_name, "b2"), np.arange(1, 21))
        np.testing.assert_array_equal(self.read_back(self.out_name, "w"), np.full(20, 0.5))

    def test02_shuffled(self, df):
        dl = ROOT.Experimental.ML.RDataLoader(
            df,
            batch_size=4,
            batches_in_memory=2,
            columns=["b1"],
            shuffle=True,
            set_seed=42,
            drop_remainder=False,
        )

        gen_train, gen_validation = dl.train_test_split(0.4)

        gen_train.save(self.tree_name, self.out_name)
        gen_validation.save(self.tree_name, self.out_name2)

        train_b1 = self.read_back(self.out_name, "b1")
        val_b1 = self.read_back(self.out_name2, "b1")

        # the two splits cover the dataset exactly once
        all_b1 = np.concatenate([train_b1, val_b1])
        np.testing.assert_array_equal(np.sort(all_b1), np.arange(20))

        # and the entries are not written in their original order
        assert (train_b1 != np.sort(train_b1)).any()

    def test03_vector_columns(self, df):
        df = df.Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")

        dl = ROOT.Experimental.ML.RDataLoader(
            df,
            batch_size=5,
            batches_in_memory=2,
            columns=["b1", "v1"],
            max_vec_sizes={"v1": 3},
            vec_padding=42.0,
            shuffle=False,
            drop_remainder=False,
        )

        dl.save(self.tree_name, self.out_name)

        # v1 is saved as a vector column, not exploded into v1_0...
        assert self.saved_columns(self.out_name) == {"b1", "v1"}

        v1 = self.read_back(self.out_name, "v1")
        assert v1.shape == (20, 3)
        # v1 contains 2 elements padded up to max_vec_sizes["v1"] (3) with vec_padding (42)
        np.testing.assert_array_equal(v1[3], [3.0, 30.0, 42.0])

    def test04_ttree_and_rntuple_agree(self):
        def save_as(file_name, output_format):
            df = from_files(self.define_rdf()).Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")

            dl = ROOT.Experimental.ML.RDataLoader(
                df,
                batch_size=5,
                batches_in_memory=2,
                columns=["b1", "v1"],
                max_vec_sizes={"v1": 3},
                shuffle=False,
                drop_remainder=False,
            )

            dl.save(self.tree_name, file_name, output_format=output_format)

        save_as(self.out_name, "ttree")
        save_as(self.out_name2, "rntuple")

        for column in ["b1", "v1"]:
            np.testing.assert_array_equal(
                self.read_back(self.out_name, column),
                self.read_back(self.out_name2, column),
            )

    def test05_filtered(self, df):
        df = df.Filter("b1 % 2 == 0", "even")

        dl = ROOT.Experimental.ML.RDataLoader(
            df,
            batch_size=2,
            batches_in_memory=2,
            columns=["b1"],
            shuffle=False,
            drop_remainder=False,
        )

        dl.save(self.tree_name, self.out_name)

        saved_b1 = self.read_back(self.out_name, "b1")
        np.testing.assert_array_equal(np.sort(saved_b1), np.arange(0, 20, 2))

    def test06_multiple_dataframes(self):
        df1 = from_files(self.define_rdf(num_of_entries=10))
        df2 = from_files(self.define_rdf(num_of_entries=10, offset=100))

        dl = ROOT.Experimental.ML.RDataLoader(
            [df1, df2],
            batch_size=5,
            batches_in_memory=2,
            columns=["b1"],
            shuffle=False,
            drop_remainder=False,
        )

        dl.save(self.tree_name, self.out_name)

        saved_b1 = self.read_back(self.out_name, "b1")
        expected = np.concatenate([np.arange(10), np.arange(100, 110)])
        np.testing.assert_array_equal(np.sort(saved_b1), expected)

    def test07_eager_loading(self, df):
        dl = ROOT.Experimental.ML.RDataLoader(
            df,
            batch_size=5,
            columns=["b1", "b2"],
            shuffle=False,
            drop_remainder=False,
            load_eager=True,
        )

        dl.save(self.tree_name, self.out_name)

        np.testing.assert_array_equal(self.read_back(self.out_name, "b1"), np.arange(20))
        np.testing.assert_array_equal(self.read_back(self.out_name, "b2"), np.arange(1, 21))

    def test08_round_trip(self, df):
        df = df.Define("w", "(double) 0.5").Define("v1", "ROOT::VecOps::RVec<int>{ b1, b1 * 10 }")

        loader_args = dict(
            batch_size=5,
            batches_in_memory=2,
            columns=["b1", "v1", "b2", "w"],
            max_vec_sizes={"v1": 3},
            vec_padding=42.0,
            target="b2",
            weights="w",
            shuffle=False,
            drop_remainder=False,
        )

        dl = ROOT.Experimental.ML.RDataLoader(df, **loader_args)
        dl.save(self.tree_name, self.out_name)

        # the saved file is fed straight back into a new RDataLoader with the same columns/target
        reloaded = ROOT.Experimental.ML.RDataLoader(ROOT.RDataFrame(self.tree_name, self.out_name), **loader_args)

        x, y, w = (np.concatenate(arrs) for arrs in zip(*reloaded.as_numpy()))
        np.testing.assert_array_equal(x[:, 0], np.arange(20))  # b1
        np.testing.assert_array_equal(x[:, 1], np.arange(20))  # v1[0] == b1
        np.testing.assert_array_equal(x[:, 2], np.arange(20) * 10)  # v1[1] == b1 * 10
        np.testing.assert_array_equal(x[:, 3], np.full(20, 42.0))  # v1[2], padding
        np.testing.assert_array_equal(y.flatten(), np.arange(1, 21))
        np.testing.assert_array_equal(w.flatten(), np.full(20, 0.5))


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
