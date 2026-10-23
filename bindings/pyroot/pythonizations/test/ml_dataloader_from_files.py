import numpy as np
import pytest
import ROOT


class TestDataLoaderFromFiles:
    file_name1 = "dataloader_from_files1.root"
    file_name2 = "dataloader_from_files2.root"
    tree_name = "mytree"

    # Helpers
    def create_file(self, file_name, num_of_entries=10, offset=0, output_format="ttree"):
        opts = ROOT.RDF.RSnapshotOptions()
        if output_format == "rntuple":
            opts.fOutputFormat = ROOT.RDF.ESnapshotOutputFormat.kRNTuple
        (
            ROOT
            .RDataFrame(num_of_entries)
            .Define("b1", f"(int) rdfentry_ + {offset}")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, file_name, ["b1", "b2"], opts)
        )

    def load_all(self, inputs, **kwargs):
        """Concatenate the features and the target of all the batches of an unshuffled loader."""
        dl = ROOT.Experimental.ML.RDataLoader(
            inputs, batch_size=3, batches_in_memory=2, target="b2", shuffle=False, drop_remainder=False, **kwargs
        )
        batches = list(dl.as_numpy())
        x = np.concatenate([x for x, _ in batches]).flatten()
        y = np.concatenate([y for _, y in batches]).flatten()
        return x, y

    def test01_single_path(self, load_eager):
        self.create_file(self.file_name1)

        x, y = self.load_all((self.tree_name, self.file_name1), load_eager=load_eager)
        np.testing.assert_array_equal(x, np.arange(10))
        np.testing.assert_array_equal(y, np.arange(10) ** 2)

    def test02_pathlike(self):
        from pathlib import Path

        self.create_file(self.file_name1)

        x, y = self.load_all((self.tree_name, Path(self.file_name1)))
        np.testing.assert_array_equal(x, np.arange(10))
        np.testing.assert_array_equal(y, np.arange(10) ** 2)

    def test03_list_of_paths(self):
        from pathlib import Path

        self.create_file(self.file_name1, num_of_entries=9)
        self.create_file(self.file_name2, num_of_entries=6, offset=9)

        # the files are read as a single dataset, in the given order
        x, y = self.load_all((self.tree_name, [self.file_name1, Path(self.file_name2)]))
        np.testing.assert_array_equal(x, np.arange(15))
        np.testing.assert_array_equal(y, np.arange(15) ** 2)

    def test04_glob(self):
        self.create_file(self.file_name1)
        self.create_file(self.file_name2, num_of_entries=5, offset=10)

        x, _ = self.load_all((self.tree_name, "dataloader_from_files*.root"))
        np.testing.assert_array_equal(np.sort(x), np.arange(15))

    def test05_rntuple(self):
        self.create_file(self.file_name1, output_format="rntuple")

        x, y = self.load_all((self.tree_name, self.file_name1))
        np.testing.assert_array_equal(x, np.arange(10))
        np.testing.assert_array_equal(y, np.arange(10) ** 2)

    def test06_train_test_split(self):
        self.create_file(self.file_name1)

        dl = ROOT.Experimental.ML.RDataLoader(
            (self.tree_name, self.file_name1),
            batch_size=2,
            columns=["b1"],
            shuffle=False,
            drop_remainder=False,
        )
        gen_train, gen_validation = dl.train_test_split(0.4)

        train_x = np.concatenate(list(gen_train.as_numpy())).flatten()
        val_x = np.concatenate(list(gen_validation.as_numpy())).flatten()
        np.testing.assert_array_equal(train_x, np.arange(6))
        np.testing.assert_array_equal(val_x, np.arange(6, 10))

    def test07_invalid_inputs(self):
        self.create_file(self.file_name1)
        df = ROOT.RDataFrame(self.tree_name, self.file_name1)

        # file paths need the name of the dataset to read
        with pytest.raises(TypeError, match="dataset_name, files"):
            self.load_all(self.file_name1)

        with pytest.raises(ValueError, match="at least one valid input is required"):
            self.load_all([])
        with pytest.raises(ValueError, match="at least one valid input is required"):
            self.load_all(df for df in [])

        # all the arguments but the inputs are keyword-only
        with pytest.raises(TypeError):
            ROOT.Experimental.ML.RDataLoader(df, 3)

    def test08_several_datasets(self):
        self.create_file(self.file_name1, num_of_entries=9)
        self.create_file(self.file_name2, num_of_entries=6, offset=9)

        # one RDataFrame per tuple, which can be mixed with RDataFrames
        df = ROOT.RDataFrame(self.tree_name, self.file_name2)
        for inputs in (
            [(self.tree_name, self.file_name1), (self.tree_name, self.file_name2)],
            [(self.tree_name, self.file_name1), df],
        ):
            x, y = self.load_all(inputs)
            np.testing.assert_array_equal(x, np.arange(15))
            np.testing.assert_array_equal(y, np.arange(15) ** 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
