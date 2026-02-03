import os
import unittest
from random import randrange, uniform

import numpy as np
import ROOT


class RBatchGeneratorMultipleFiles(unittest.TestCase):
    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    file_name3 = "vector_columns.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    # Helpers
    def define_rdf(self, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define("b2", "(double) b1*b1")

        return df

    def create_file(self, num_of_entries=10):
        self.define_rdf(num_of_entries).Snapshot(self.tree_name, self.file_name1)

    def create_5_entries_file(self):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 10")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, self.file_name2)
        )

    def create_vector_file(self, num_of_entries=10):
        (
            ROOT.RDataFrame(10)
            .Define("b1", "(int) rdfentry_")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name3)
        )

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            entries_before = df.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            entries_after = df.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframe is correctly reset
            self.assertTrue(np.array_equal(entries_before, entries_after))

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test02_each_element_is_generated_shuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=True,
                drop_remainder=False,
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            self.assertEqual(len(flat_x_train), 6)
            self.assertEqual(len(flat_x_val), 4)
            self.assertEqual(len(flat_y_train), 6)
            self.assertEqual(len(flat_y_val), 4)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test03_chunk_input_smaller_than_batch_size(self):
        """Checking for the situation when the batch can only be created after
        more than two chunks. If not, segmentation fault will arise"""
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=3,
                block_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            next(iter(gen_train))

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test04_dropping_remainder(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=True,
            )

            collected_x = []
            collected_y = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 3)
            self.assertEqual(len(collected_y), 3)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test05_more_than_one_file(self):
        self.create_file()
        self.create_5_entries_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, [self.file_name1, self.file_name2])

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0]
            results_x_val = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.0, 9.0, 16.0, 64.0]
            results_y_val = [81.0, 100.0, 121.0, 144.0, 169.0, 196.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test06_multiple_target_columns(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)
        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test07_multiple_input_columns(self):
        file_name = "multiple_input_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            results_x_train = [0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]
            results_x_val = [6.0, 60.0, 7.0, 70.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 2))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test08_filtered(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            dff = df.Filter("b1 % 2 == 0", "name")

            filter_entries_before = dff.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                dff,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            results_x_train = [0.0, 2.0, 4.0]
            results_x_val = [6.0, 8.0]
            results_y_train = [0.0, 4.0, 16.0]
            results_y_val = [36.0, 64.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            x, y = next(train_iter)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            filter_entries_after = dff.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframe is correctly reset
            self.assertTrue(np.array_equal(filter_entries_before, filter_entries_after))

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test09_filtered_last_chunk(self):
        file_name = "filtered_last_chunk.root"
        tree_name = "myTree"

        ROOT.RDataFrame(20).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Snapshot(
            tree_name, file_name
        )

        try:
            df = ROOT.RDataFrame(tree_name, file_name)

            dff = df.Filter("b1 % 2 == 0", "name")

            gen_train, _ = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                dff,
                batch_size=3,
                chunk_size=9,
                block_size=1,
                target="b2",
                validation_split=0,
                shuffle=False,
                drop_remainder=False,
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_y_train = []

            train_iter = iter(gen_train)

            for _ in range(3):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (1, 1))
            self.assertTrue(y.shape == (1, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_y_train, flat_y_train)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test10_two_epochs_shuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)

    def test11_number_of_training_and_validation_batches_remainder(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches, number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches, number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 0)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 1)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test12_PyTorch(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test13_TensorFlow(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=1,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.numpy().tolist())
                collected_y_train.append(y.numpy().tolist())
                collected_z_train.append(z.numpy().tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.numpy().tolist())
                collected_y_val.append(y.numpy().tolist())
                collected_z_val.append(z.numpy().tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 2))
            self.assertTrue(z.shape == (3, 1))
            collected_x_val.append(x.numpy().tolist())
            collected_y_val.append(y.numpy().tolist())
            collected_z_val.append(z.numpy().tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test14_big_data(self):
        file_name = "big_data.root"
        tree_name = "myTree"

        entries_in_rdf = randrange(10000, 30000)
        chunk_size = randrange(1000, 3001)
        batch_size = randrange(100, 501)

        error_message = f"\n Batch size: {batch_size} Chunk size: {chunk_size}\
            Number of entries: {entries_in_rdf}"

        def define_rdf(num_of_entries):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, size_of_chunk, num_of_entries):
            define_rdf(num_of_entries)

            try:
                df = ROOT.RDataFrame(tree_name, file_name)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    df,
                    batch_size=size_of_batch,
                    chunk_size=size_of_chunk,
                    block_size=1,
                    target=["b3", "b5"],
                    weights="b2",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                )

                collect_x = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message + f" row: {i}")
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message + f" row: {i}")
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message + f" row: {i}")

                    collect_x.extend(list(x[:, 0]))

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message)
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message)
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message)

                    collect_x.extend(list(x[:, 0]))

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                self.assertTrue(
                    set(collect_x) == set(i for i in range(num_of_entries)),
                    f"collected length: {len(set(collect_x))}\
                                 generated length {len(set(i for i in range(num_of_entries)))}",
                )

            except:
                self.teardown_file(file_name)
                raise

        test(batch_size, chunk_size, entries_in_rdf)

    def test15_two_runs_set_seed(self):
        self.create_file()

        try:
            both_runs_collected_x_val = []
            both_runs_collected_y_val = []

            df = ROOT.RDataFrame(self.tree_name, self.file_name1)
            for _ in range(2):
                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    df,
                    batch_size=3,
                    chunk_size=5,
                    block_size=2,
                    target="b2",
                    validation_split=0.4,
                    shuffle=True,
                    drop_remainder=False,
                    set_seed=42,
                )

                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_runs_collected_x_val.append(collected_x_val)
                both_runs_collected_y_val.append(collected_y_val)
            self.assertEqual(both_runs_collected_x_val[0], both_runs_collected_x_val[1])
            self.assertEqual(both_runs_collected_y_val[0], both_runs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)

    def test16_vector_padding(self):
        self.create_vector_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name3)
            max_vec_sizes = {"v1": 3, "v2": 2}

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                block_size=2,
                target="b1",
                validation_split=0.4,
                max_vec_sizes=max_vec_sizes,
                shuffle=False,
                drop_remainder=False,
            )

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
                0,
                400.0,
                4000.0,
                5.0,
                50.0,
                0,
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

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 5))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name3)

        except:
            self.teardown_file(self.file_name3)
            raise


class RBatchGeneratorEagerLoading(unittest.TestCase):
    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    file_name3 = "vector_columns.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    # Helpers
    def define_rdf(self, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define("b2", "(double) b1*b1")

        return df

    def create_file(self, num_of_entries=10):
        self.define_rdf(num_of_entries).Snapshot(self.tree_name, self.file_name1)

    def create_5_entries_file(self):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 10")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, self.file_name2)
        )

    def create_vector_file(self, num_of_entries=10):
        (
            ROOT.RDataFrame(10)
            .Define("b1", "(int) rdfentry_")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name3)
        )

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test02_each_element_is_generated_shuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df, batch_size=3, target="b2", validation_split=0.4, shuffle=True, drop_remainder=False, load_eager=True
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            self.assertEqual(len(flat_x_train), 6)
            self.assertEqual(len(flat_x_val), 4)
            self.assertEqual(len(flat_y_train), 6)
            self.assertEqual(len(flat_y_val), 4)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test04_dropping_remainder(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df, batch_size=3, target="b2", validation_split=0.4, shuffle=False, drop_remainder=True, load_eager=True
            )

            collected_x = []
            collected_y = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 3)
            self.assertEqual(len(collected_y), 3)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test05_more_than_one_file(self):
        self.create_file()
        self.create_5_entries_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, [self.file_name1, self.file_name2])

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            results_x_val = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
            results_y_val = [81.0, 100.0, 121.0, 144.0, 169.0, 196.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test06_multiple_target_columns(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)
        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test07_multiple_input_columns(self):
        file_name = "multiple_input_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]
            results_x_val = [6.0, 60.0, 7.0, 70.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 2))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test08_filtered(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            dff = df.Filter("b1 % 2 == 0", "name")

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                dff,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 2.0, 4.0]
            results_x_val = [6.0, 8.0]
            results_y_train = [0.0, 4.0, 16.0]
            results_y_val = [36.0, 64.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            x, y = next(train_iter)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test09_filtered_last_chunk(self):
        file_name = "filtered_last_chunk.root"
        tree_name = "myTree"

        ROOT.RDataFrame(20).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Snapshot(
            tree_name, file_name
        )

        try:
            df = ROOT.RDataFrame(tree_name, file_name)

            dff = df.Filter("b1 % 2 == 0", "name")

            gen_train, _ = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                dff, batch_size=3, target="b2", validation_split=0, shuffle=False, drop_remainder=False, load_eager=True
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_y_train = []

            train_iter = iter(gen_train)

            for _ in range(3):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (1, 1))
            self.assertTrue(y.shape == (1, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_y_train, flat_y_train)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test10_two_epochs_shuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)

    def test11_number_of_training_and_validation_batches_remainder(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches, number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches, number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 0)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 1)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test12_PyTorch(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                df,
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test13_TensorFlow(self):
        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10).Define("b1", "(Short_t) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Define(
            "b3", "(double) rdfentry_ * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
                df,
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.numpy().tolist())
                collected_y_train.append(y.numpy().tolist())
                collected_z_train.append(z.numpy().tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.numpy().tolist())
                collected_y_val.append(y.numpy().tolist())
                collected_z_val.append(z.numpy().tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 2))
            self.assertTrue(z.shape == (3, 1))
            collected_x_val.append(x.numpy().tolist())
            collected_y_val.append(y.numpy().tolist())
            collected_z_val.append(z.numpy().tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name)

        except:
            self.teardown_file(file_name)
            raise

    def test14_big_data(self):
        file_name = "big_data.root"
        tree_name = "myTree"

        entries_in_rdf = randrange(10000, 30000)
        chunk_size = randrange(1000, 3001)
        batch_size = randrange(100, 501)

        error_message = f"\n Batch size: {batch_size} Chunk size: {chunk_size}\
            Number of entries: {entries_in_rdf}"

        def define_rdf(num_of_entries):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, size_of_chunk, num_of_entries):
            define_rdf(num_of_entries)

            try:
                df = ROOT.RDataFrame(tree_name, file_name)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    df,
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b2",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                )

                collect_x = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message + f" row: {i}")
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message + f" row: {i}")
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message + f" row: {i}")

                    collect_x.extend(list(x[:, 0]))

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message)
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message)
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message)

                    collect_x.extend(list(x[:, 0]))

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                self.assertTrue(
                    set(collect_x) == set(i for i in range(num_of_entries)),
                    f"collected length: {len(set(collect_x))}\
                                 generated length {len(set(i for i in range(num_of_entries)))}",
                )

            except:
                self.teardown_file(file_name)
                raise

        test(batch_size, chunk_size, entries_in_rdf)

    def test15_two_runs_set_seed(self):
        self.create_file()

        try:
            both_runs_collected_x_val = []
            both_runs_collected_y_val = []

            df = ROOT.RDataFrame(self.tree_name, self.file_name1)
            for _ in range(2):
                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    df,
                    batch_size=3,
                    target="b2",
                    validation_split=0.4,
                    shuffle=True,
                    drop_remainder=False,
                    set_seed=42,
                    load_eager=True,
                )

                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_runs_collected_x_val.append(collected_x_val)
                both_runs_collected_y_val.append(collected_y_val)
            self.assertEqual(both_runs_collected_x_val[0], both_runs_collected_x_val[1])
            self.assertEqual(both_runs_collected_y_val[0], both_runs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)

    def test16_vector_padding(self):
        self.create_vector_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name3)
            max_vec_sizes = {"v1": 3, "v2": 2}

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b1",
                validation_split=0.4,
                max_vec_sizes=max_vec_sizes,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

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
                0,
                400.0,
                4000.0,
                5.0,
                50.0,
                0,
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

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 5))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name3)

        except:
            self.teardown_file(self.file_name3)
            raise


class RBatchGeneratorEagerLoadingMultipleDataframes(unittest.TestCase):
    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    file_name3 = "second_file.root"
    file_name4 = "vector_columns_1.root"
    file_name5 = "vector_columns_2.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    # Helpers
    def define_rdf1(self, num_of_entries=5):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define("b2", "(double) b1*b1")

        return df

    def define_rdf2(self, num_of_entries=5):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_ + 5").Define("b2", "(double) b1*b1")

        return df

    def create_file1(self, num_of_entries=5):
        self.define_rdf1(num_of_entries).Snapshot(self.tree_name, self.file_name1)

    def create_file2(self, num_of_entries=5):
        self.define_rdf2(num_of_entries).Snapshot(self.tree_name, self.file_name2)

    def create_5_entries_file(self):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 10")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, self.file_name3)
        )

    def create_vector_file1(self, num_of_entries=5):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name4)
        )

    def create_vector_file2(self, num_of_entries=5):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 5")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name5)
        )

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            df1_entries_before = df1.AsNumpy(["rdfentry_"])["rdfentry_"]
            df2_entries_before = df2.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.0]
            results_y_val = [9.0, 16.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            df1_entries_after = df1.AsNumpy(["rdfentry_"])["rdfentry_"]
            df2_entries_after = df2.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(df1_entries_before, df1_entries_after))
            self.assertTrue(np.array_equal(df2_entries_before, df2_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test02_each_element_is_generated_shuffled(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=True,
                drop_remainder=False,
                load_eager=True,
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
            flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

            self.assertEqual(len(flat_x_train), 6)
            self.assertEqual(len(flat_x_val), 4)
            self.assertEqual(len(flat_y_train), 6)
            self.assertEqual(len(flat_y_val), 4)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test04_dropping_remainder(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=True,
                load_eager=True,
            )

            collected_x = []
            collected_y = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 3)
            self.assertEqual(len(collected_y), 3)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test05_more_than_one_file(self):
        self.create_file1()
        self.create_file2()
        self.create_5_entries_file()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, [self.file_name1, self.file_name2])
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name3)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 121.0, 144.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0, 169.0, 196.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)
            raise

    def test06_multiple_target_columns(self):
        file_name1 = "multiple_target_columns_1.root"
        file_name2 = "multiple_target_columns_2.root"

        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_ + 5").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df1 = ROOT.RDataFrame("myTree", file_name1)
            df2 = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test07_multiple_input_columns(self):
        file_name1 = "multiple_target_columns_1.root"
        file_name2 = "multiple_target_columns_2.root"

        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name1)

        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_ + 5").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name2)

        try:
            df1 = ROOT.RDataFrame("myTree", file_name1)
            df2 = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0]
            results_x_val = [3.0, 30.0, 4.0, 40.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.0]
            results_y_val = [9.0, 16.0, 64.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (3, 2))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 2))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test08_filtered(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            dff1 = df1.Filter("b1 % 2 == 0", "name")
            dff2 = df2.Filter("b1 % 2 != 0", "name")

            dff1_entries_before = dff1.AsNumpy(["rdfentry_"])["rdfentry_"]
            dff2_entries_before = dff2.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [dff1, dff2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 2.0, 5.0]
            results_x_val = [4.0, 9.0]
            results_y_train = [0.0, 4.0, 25.0]
            results_y_val = [16.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            x, y = next(train_iter)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            dff1_entries_after = dff1.AsNumpy(["rdfentry_"])["rdfentry_"]
            dff2_entries_after = dff2.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(dff1_entries_before, dff1_entries_after))
            self.assertTrue(np.array_equal(dff2_entries_before, dff2_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test09_filtered_last_chunk(self):
        file_name1 = "filtered_last_chunk_1.root"
        file_name2 = "filtered_last_chunk_2.root"
        tree_name = "myTree"

        ROOT.RDataFrame(10).Define("b1", "(int) rdfentry_").Define("b2", "(UShort_t) b1 * b1").Snapshot(
            tree_name, file_name1
        )

        ROOT.RDataFrame(10).Define("b1", "(int) rdfentry_ + 10").Define("b2", "(UShort_t) b1 * b1").Snapshot(
            tree_name, file_name2
        )

        try:
            df1 = ROOT.RDataFrame(tree_name, file_name1)
            df2 = ROOT.RDataFrame(tree_name, file_name2)

            dff1 = df1.Filter("b1 % 2 == 0", "name")
            dff2 = df2.Filter("b1 % 2 == 0", "name")

            gen_train, _ = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [dff1, dff2],
                batch_size=3,
                target="b2",
                validation_split=0,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_y_train = []

            train_iter = iter(gen_train)

            for _ in range(3):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (1, 1))
            self.assertTrue(y.shape == (1, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_y_train, flat_y_train)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test10_two_epochs_shuffled(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test11_number_of_training_and_validation_batches_remainder(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches, number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches, number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 0)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 1)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test12_PyTorch(self):
        file_name1 = "multiple_target_columns_1.root"
        file_name2 = "multiple_target_columns_2.root"

        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_ + 5").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)

        try:
            df1 = ROOT.RDataFrame("myTree", file_name1)
            df2 = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                [df1, df2],
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 2))
            self.assertTrue(z.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            collected_z_val.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test13_TensorFlow(self):
        file_name1 = "multiple_target_columns_1.root"
        file_name2 = "multiple_target_columns_2.root"

        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) rdfentry_ + 5").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)

        try:
            df1 = ROOT.RDataFrame("myTree", file_name1)
            df2 = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
                [df1, df2],
                batch_size=3,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0, 0.0, 0.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0, 200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0, 800.0, 81.0, 900.0, 0.0, 0.0, 0.0, 0.0]
            results_z_train = [0.0, 10.0, 20.0, 50.0, 60.0, 70.0]
            results_z_val = [30.0, 40.0, 80.0, 90.0, 0.0, 0.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_train.append(x.numpy().tolist())
                collected_y_train.append(y.numpy().tolist())
                collected_z_train.append(z.numpy().tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 2))
                self.assertTrue(z.shape == (3, 1))
                collected_x_val.append(x.numpy().tolist())
                collected_y_val.append(y.numpy().tolist())
                collected_z_val.append(z.numpy().tolist())

            x, y, z = next(iter_val)
            self.assertTrue(x.shape == (3, 1))
            self.assertTrue(y.shape == (3, 2))
            self.assertTrue(z.shape == (3, 1))
            collected_x_val.append(x.numpy().tolist())
            collected_y_val.append(y.numpy().tolist())
            collected_z_val.append(z.numpy().tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test14_big_data(self):
        file_name1 = "big_data_1.root"
        file_name2 = "big_data_2.root"
        tree_name = "myTree"

        entries_in_rdf = randrange(10000, 30000)
        chunk_size = randrange(1000, 3001)
        batch_size = randrange(100, 501)

        error_message = f"\n Batch size: {batch_size} Chunk size: {chunk_size}\
            Number of entries: {entries_in_rdf}"

        def define_rdf(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, size_of_chunk, num_of_entries):
            define_rdf(num_of_entries, file_name1)
            define_rdf(num_of_entries, file_name2)

            try:
                df1 = ROOT.RDataFrame(tree_name, file_name1)
                df2 = ROOT.RDataFrame(tree_name, file_name2)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df1, df2],
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b2",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                )

                collect_x = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message + f" row: {i}")
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message + f" row: {i}")
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message + f" row: {i}")

                    collect_x.extend(list(x[:, 0]))

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(np.all(x[:, 0] * (-1) == x[:, 1]), error_message)
                    self.assertTrue(np.all(x[:, 0] + 10192 == y[:, 0]), error_message)
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(np.all(x[:, 0] * 2 == z[:, 0]), error_message)

                    collect_x.extend(list(x[:, 0]))

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                self.assertTrue(
                    set(collect_x) == set(i for i in range(num_of_entries)),
                    f"collected length: {len(set(collect_x))}\
                                 generated length {len(set(i for i in range(num_of_entries)))}",
                )

            except:
                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
                raise

        test(batch_size, chunk_size, entries_in_rdf)

    def test15_two_runs_set_seed(self):
        self.create_file1()
        self.create_file2()

        try:
            both_runs_collected_x_val = []
            both_runs_collected_y_val = []

            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)

            for _ in range(2):
                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df1, df2],
                    batch_size=3,
                    target="b2",
                    validation_split=0.4,
                    shuffle=True,
                    drop_remainder=False,
                    set_seed=42,
                    load_eager=True,
                )

                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (self.val_remainder, 1))
                self.assertTrue(y.shape == (self.val_remainder, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_runs_collected_x_val.append(collected_x_val)
                both_runs_collected_y_val.append(collected_y_val)
            self.assertEqual(both_runs_collected_x_val[0], both_runs_collected_x_val[1])
            self.assertEqual(both_runs_collected_y_val[0], both_runs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test16_vector_padding(self):
        self.create_vector_file1()
        self.create_vector_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name4)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name5)
            max_vec_sizes = {"v1": 3, "v2": 2}

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df1, df2],
                batch_size=3,
                target="b1",
                validation_split=0.4,
                max_vec_sizes=max_vec_sizes,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
            )

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

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 5))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 5))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)

        except:
            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)
            raise


class RBatchGeneratorRandomUndersampling(unittest.TestCase):
    file_name1 = "major.root"
    file_name2 = "minor.root"
    file_name3 = "second_file.root"
    file_name4 = "vector_columns_major.root"
    file_name5 = "vector_columns_minor.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 4
    n_val_batch = 3
    train_remainder = 1

    # Helpers
    def define_rdf_even(self, num_of_entries=20):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(double) b1*b1")

        return df

    def define_rdf_odd(self, num_of_entries=5):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(double) b1*b1")

        return df

    def create_file_major(self, num_of_entries=20):
        self.define_rdf_even(num_of_entries).Snapshot(self.tree_name, self.file_name1)

    def create_file_minor(self, num_of_entries=5):
        self.define_rdf_odd(num_of_entries).Snapshot(self.tree_name, self.file_name2)

    def create_5_entries_file(self):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) 2 * (rdfentry_ + 20)")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, self.file_name3)
        )

    def create_vector_file_major(self, num_of_entries=20):
        (
            ROOT.RDataFrame(20)
            .Define("b1", "(int) rdfentry_")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name4)
        )

    def create_vector_file_minor(self, num_of_entries=5):
        (
            ROOT.RDataFrame(5)
            .Define("b1", "(int) rdfentry_ + 20")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name5)
        )

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            major_entries_before = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_before = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
            results_x_val = [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 1.0, 9.0, 25.0]
            results_y_val = [576.0, 676.0, 784.0, 900.0, 49.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            # check if there are no duplicate entries (replacement=False)
            self.assertEqual(len(set(flat_x_train)), len(flat_x_train))
            self.assertEqual(len(set(flat_x_val)), len(flat_x_val))
            self.assertEqual(len(set(flat_y_train)), len(flat_y_train))
            self.assertEqual(len(set(flat_y_val)), len(flat_y_val))

            # check if correct sampling_ratio (0.5 = minor/major)
            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            major_entries_after = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_after = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(major_entries_before, major_entries_after))
            self.assertTrue(np.array_equal(minor_entries_before, minor_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test01_each_element_is_generated_unshuffled_replacement(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.2,
                replacement=True,
            )

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

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(6):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(9):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            # check if there are duplicate entries (replacement=True)
            self.assertLess(len(set(flat_x_train)), len(flat_x_train))
            self.assertLess(len(set(flat_x_val)), len(flat_x_val))
            self.assertLess(len(set(flat_y_train)), len(flat_y_train))
            self.assertLess(len(set(flat_y_val)), len(flat_y_val))

            # check if correct sampling_ratio (0.2 = minor/major)
            self.assertEqual(num_major_train, 15)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 10)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test02_each_element_is_generated_shuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(len(flat_x_train), 9)
            self.assertEqual(len(flat_x_val), 6)
            self.assertEqual(len(flat_y_train), 9)
            self.assertEqual(len(flat_y_val), 6)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test04_dropping_remainder(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            collected_x = []
            collected_y = []

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x.append(x)
                collected_y.append(y)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 7)
            self.assertEqual(len(collected_y), 7)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test05_more_than_one_file(self):
        self.create_file_major()
        self.create_file_minor()
        self.create_5_entries_file()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, [self.file_name1, self.file_name3])
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
            results_x_val = [30.0, 32.0, 34.0, 36.0, 7.0, 9.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 1.0, 9.0, 25.0]
            results_y_val = [900.0, 1024.0, 1156.0, 1296.0, 49.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)
            raise

    def test06_multiple_target_columns(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(20).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_major = ROOT.RDataFrame("myTree", file_name1)
            df_minor = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test07_multiple_input_columns(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(20).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name1)

        ROOT.RDataFrame(5).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name2)

        try:
            df_major = ROOT.RDataFrame("myTree", file_name1)
            df_minor = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (2, 2))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (2, 2))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 2))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test08_filtered(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            df_major_filter = df_major.Filter("b1 % 8 != 0", "name")

            major_filter_entries_before = df_major_filter.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_before = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major_filter, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            results_x_train = [2.0, 4.0, 6.0, 10.0, 12.0, 14.0, 1.0, 3.0, 5.0]
            results_x_val = [26.0, 28.0, 30.0, 34.0, 7.0, 9.0]
            results_y_train = [4.0, 16.0, 36.0, 100.0, 144.0, 196.0, 1.0, 9.0, 25.0]
            results_y_val = [676.0, 784.0, 900.0, 1156.0, 49.0, 81.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = np.sum((np.asarray(flat_x_train) % 2 == 0) & (np.asarray(flat_x_train) % 8 != 0))
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = np.sum((np.asarray(flat_x_val) % 2 == 0) & (np.asarray(flat_x_val) % 8 != 0))
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            major_filter_entries_after = df_major_filter.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_after = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(major_filter_entries_before, major_filter_entries_after))
            self.assertTrue(np.array_equal(minor_entries_before, minor_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test10_two_epochs_shuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )
            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_train)
                self.assertTrue(x.shape == (self.train_remainder, 1))
                self.assertTrue(y.shape == (self.train_remainder, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 9)
                self.assertEqual(len(flat_x_val), 6)
                self.assertEqual(len(flat_y_train), 9)
                self.assertEqual(len(flat_y_val), 6)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test11_number_of_training_and_validation_batches_remainder(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches, number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches, number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 1)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 0)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test12_PyTorch(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(20).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_minor = ROOT.RDataFrame("myTree", file_name1)
            df_major = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test13_TensorFlow(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(20).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(5).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_minor = ROOT.RDataFrame("myTree", file_name1)
            df_major = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

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

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test14_big_data_replacement_false(self):
        file_name1 = "big_data_major.root"
        file_name2 = "big_data_minor.root"
        tree_name = "myTree"

        entries_in_rdf_major = randrange(10000, 30000)
        entries_in_rdf_minor = randrange(8000, 9999)
        batch_size = randrange(100, 501)
        min_allowed_sampling_ratio = entries_in_rdf_minor / entries_in_rdf_major
        sampling_ratio = round(uniform(min_allowed_sampling_ratio, 2), 2)

        error_message = f"\n Batch size: {batch_size}\
            Number of major entries: {entries_in_rdf_major} \
            Number of minor entries: {entries_in_rdf_minor}"

        def define_rdf_major(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def define_rdf_minor(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            define_rdf_major(num_of_entries_major, file_name1)
            define_rdf_minor(num_of_entries_minor, file_name2)

            try:
                df1 = ROOT.RDataFrame(tree_name, file_name1)
                df2 = ROOT.RDataFrame(tree_name, file_name2)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df1, df2],
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b1",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="undersampling",
                    sampling_ratio=sampling_ratio,
                    replacement=False,
                )

                collected_z_train = []
                collected_z_val = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_train.append(z.tolist())

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collected_z_train.append(z.tolist())

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_val.append(z.tolist())

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collected_z_val.append(z.tolist())

                flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
                flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

                num_major_train = sum(np.array(flat_z_train) % 2 == 0)
                num_minor_train = sum(np.array(flat_z_train) % 2 != 0)
                num_major_val = sum(np.array(flat_z_val) % 2 == 0)
                num_minor_val = sum(np.array(flat_z_val) % 2 != 0)

                # check if there are no duplicate entries (replacement=False)
                self.assertEqual(len(set(flat_z_train)), len(flat_z_train))
                self.assertEqual(len(set(flat_z_val)), len(flat_z_val))

                # check if the sampling stategy is correct
                self.assertEqual(round((num_minor_train / num_major_train), 2), sampling_ratio)
                self.assertEqual(round((num_minor_val / num_major_val), 2), sampling_ratio)

                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
            except:
                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
                raise

        test(batch_size, entries_in_rdf_major, entries_in_rdf_minor, sampling_ratio)

    def test14_big_data_replacement_true(self):
        file_name1 = "big_data_major.root"
        file_name2 = "big_data_minor.root"
        tree_name = "myTree"

        entries_in_rdf_major = randrange(10000, 30000)
        entries_in_rdf_minor = randrange(8000, 9999)
        batch_size = randrange(100, 501)

        # max samling strategy to guarantee duplicate sampled entires
        max_sampling_ratio = entries_in_rdf_minor / entries_in_rdf_major
        sampling_ratio = round(uniform(0.01, max_sampling_ratio), 2)

        error_message = f"\n Batch size: {batch_size}\
            Number of major entries: {entries_in_rdf_major} \
            Number of minor entries: {entries_in_rdf_minor}"

        def define_rdf_major(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def define_rdf_minor(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            define_rdf_major(num_of_entries_major, file_name1)
            define_rdf_minor(num_of_entries_minor, file_name2)

            try:
                df1 = ROOT.RDataFrame(tree_name, file_name1)
                df2 = ROOT.RDataFrame(tree_name, file_name2)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df1, df2],
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b1",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="undersampling",
                    sampling_ratio=sampling_ratio,
                    replacement=True,
                )

                collected_z_train = []
                collected_z_val = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_train.append(z.tolist())

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collected_z_train.append(z.tolist())

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_val.append(z.tolist())

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collected_z_val.append(z.tolist())

                flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
                flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

                num_major_train = sum(np.array(flat_z_train) % 2 == 0)
                num_minor_train = sum(np.array(flat_z_train) % 2 != 0)
                num_major_val = sum(np.array(flat_z_val) % 2 == 0)
                num_minor_val = sum(np.array(flat_z_val) % 2 != 0)

                # check if there are duplicate entries (replacement=True)
                self.assertLess(len(set(flat_z_train)), len(flat_z_train))
                self.assertLess(len(set(flat_z_val)), len(flat_z_val))

                # check if the sampling stategy is correct
                self.assertEqual(round((num_minor_train / num_major_train), 2), sampling_ratio)
                self.assertEqual(round((num_minor_val / num_major_val), 2), sampling_ratio)

                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
            except:
                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
                raise

        test(batch_size, entries_in_rdf_major, entries_in_rdf_minor, sampling_ratio)

    def test15_two_runs_set_seed(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            both_runs_collected_x_val = []
            both_runs_collected_y_val = []

            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            for _ in range(2):
                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df_major, df_minor],
                    batch_size=2,
                    target="b2",
                    validation_split=0.4,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="undersampling",
                    sampling_ratio=0.5,
                    replacement=False,
                )

                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_train)
                self.assertTrue(x.shape == (self.train_remainder, 1))
                self.assertTrue(y.shape == (self.train_remainder, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

                flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 9)
                self.assertEqual(len(flat_x_val), 6)
                self.assertEqual(len(flat_y_train), 9)
                self.assertEqual(len(flat_y_val), 6)

                both_runs_collected_x_val.append(collected_x_val)
                both_runs_collected_y_val.append(collected_y_val)
            self.assertEqual(both_runs_collected_x_val[0], both_runs_collected_x_val[1])
            self.assertEqual(both_runs_collected_y_val[0], both_runs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test16_vector_padding(self):
        self.create_vector_file_major()
        self.create_vector_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name4)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name5)
            max_vec_sizes = {"v1": 3, "v2": 2}

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b1",
                validation_split=0.4,
                max_vec_sizes=max_vec_sizes,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="undersampling",
                sampling_ratio=0.5,
                replacement=False,
            )

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

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 5))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 5))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 5))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_y_train) < 20)
            num_minor_train = sum(np.array(flat_y_train) >= 20)
            num_major_val = sum(np.array(flat_y_val) < 20)
            num_minor_val = sum(np.array(flat_y_val) >= 20)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)

        except:
            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)
            raise


class RBatchGeneratorRandomOversampling(unittest.TestCase):
    file_name1 = "major.root"
    file_name2 = "minor.root"
    file_name3 = "second_file.root"
    file_name4 = "vector_columns_major.root"
    file_name5 = "vector_columns_minor.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 4
    n_val_batch = 3
    train_remainder = 1

    # Helpers
    def define_rdf_even(self, num_of_entries=20):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(double) b1*b1")

        return df

    def define_rdf_odd(self, num_of_entries=5):
        df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(double) b1*b1")

        return df

    def create_file_major(self, num_of_entries=10):
        self.define_rdf_even(num_of_entries).Snapshot(self.tree_name, self.file_name1)

    def create_file_minor(self, num_of_entries=3):
        self.define_rdf_odd(num_of_entries).Snapshot(self.tree_name, self.file_name2)

    def create_extra_entry_file(self):
        (
            ROOT.RDataFrame(1)
            .Define("b1", "(int) 2 * (rdfentry_ + 3) + 1")
            .Define("b2", "(double) b1 * b1")
            .Snapshot(self.tree_name, self.file_name3)
        )

    def create_vector_file_major(self, num_of_entries=10):
        (
            ROOT.RDataFrame(10)
            .Define("b1", "(int) rdfentry_")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name4)
        )

    def create_vector_file_minor(self, num_of_entries=3):
        (
            ROOT.RDataFrame(3)
            .Define("b1", "(int) rdfentry_ + 10")
            .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
            .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
            .Snapshot(self.tree_name, self.file_name5)
        )

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            major_entries_before = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_before = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [5.0, 5.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [1.0, 9.0, 1.0, 0.0, 4.0, 16.0, 36.0, 64.0, 100.0]
            results_y_val = [25.0, 25.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            # check if there are no duplicate entries (oversampling)
            self.assertLess(len(set(flat_x_train)), len(flat_x_train))
            self.assertLess(len(set(flat_x_val)), len(flat_x_val))
            self.assertLess(len(set(flat_y_train)), len(flat_y_train))
            self.assertLess(len(set(flat_y_val)), len(flat_y_val))

            # check if correct sampling_ratio (0.5 = minor/major)
            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            major_entries_after = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_entries_after = df_minor.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(major_entries_before, major_entries_after))
            self.assertTrue(np.array_equal(minor_entries_before, minor_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test02_each_element_is_generated_shuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            # check if there are no duplicate entries (oversampling)
            self.assertLess(len(set(flat_x_train)), len(flat_x_train))
            self.assertLess(len(set(flat_x_val)), len(flat_x_val))
            self.assertLess(len(set(flat_y_train)), len(flat_y_train))
            self.assertLess(len(set(flat_y_val)), len(flat_y_val))

            # check if correct sampling_ratio (0.5 = minor/major)
            self.assertEqual(len(flat_x_train), 9)
            self.assertEqual(len(flat_x_val), 6)
            self.assertEqual(len(flat_y_train), 9)
            self.assertEqual(len(flat_y_val), 6)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test04_dropping_remainder(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            collected_x = []
            collected_y = []

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x.append(x)
                collected_y.append(y)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 7)
            self.assertEqual(len(collected_y), 7)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test05_more_than_one_file(self):
        self.create_file_major()
        self.create_file_minor()
        self.create_extra_entry_file()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, [self.file_name2, self.file_name3])

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 5.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [7.0, 7.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [1.0, 9.0, 25.0, 0.0, 4.0, 16.0, 36.0, 64.0, 100.0]
            results_y_val = [49.0, 49.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            self.teardown_file(self.file_name3)
            raise

    def test06_multiple_target_columns(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(10).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(3).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_major = ROOT.RDataFrame("myTree", file_name1)
            df_minor = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [5.0, 5.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [
                1.0,
                100.0,
                9.0,
                300.0,
                1.0,
                100.0,
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
            ]
            results_y_val = [25.0, 500.0, 25.0, 500.0, 144.0, 1200.0, 196.0, 1400.0, 256.0, 1600.0, 324.0, 1800.0]
            results_z_train = [10.0, 30.0, 10.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
            results_z_val = [50.0, 50.0, 120.0, 140.0, 160.0, 180.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test07_multiple_input_columns(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(10).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name1)

        ROOT.RDataFrame(3).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Snapshot("myTree", file_name2)

        try:
            df_major = ROOT.RDataFrame("myTree", file_name1)
            df_minor = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )
            results_x_train = [
                1.0,
                10.0,
                3.0,
                30.0,
                1.0,
                10.0,
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
            ]
            results_x_val = [5.0, 50.0, 5.0, 50.0, 12.0, 120.0, 14.0, 140.0, 16.0, 160.0, 18.0, 180.0]
            results_y_train = [1.0, 9.0, 1.0, 0.0, 4.0, 16.0, 36.0, 64.0, 100.0]
            results_y_val = [25.0, 25.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(iter_train)
                self.assertTrue(x.shape == (2, 2))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(iter_val)
                self.assertTrue(x.shape == (2, 2))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 2))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test08_filtered(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor_duplicate = ROOT.RDataFrame(self.tree_name, [self.file_name2, self.file_name2, self.file_name2])

            df_minor_filter = df_minor_duplicate.Filter("rdfentry_ < 3", "name")

            major_entries_before = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_filter_entries_before = df_minor_filter.AsNumpy(["rdfentry_"])["rdfentry_"]

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor_filter],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [5.0, 5.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [1.0, 9.0, 1.0, 0.0, 4.0, 16.0, 36.0, 64.0, 100.0]
            results_y_val = [25.0, 25.0, 144.0, 196.0, 256.0, 324.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            major_entries_after = df_major.AsNumpy(["rdfentry_"])["rdfentry_"]
            minor_filter_entries_after = df_minor_filter.AsNumpy(["rdfentry_"])["rdfentry_"]

            # check if the dataframes are correctly reset
            self.assertTrue(np.array_equal(major_entries_before, major_entries_after))
            self.assertTrue(np.array_equal(minor_filter_entries_before, minor_filter_entries_after))

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test10_two_epochs_shuffled(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )
            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_train)
                self.assertTrue(x.shape == (self.train_remainder, 1))
                self.assertTrue(y.shape == (self.train_remainder, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

                flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
                flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
                flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
                flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

                self.assertEqual(len(flat_x_train), 9)
                self.assertEqual(len(flat_x_val), 6)
                self.assertEqual(len(flat_y_train), 9)
                self.assertEqual(len(flat_y_val), 6)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test11_number_of_training_and_validation_batches_remainder(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches, number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches, number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 1)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 0)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)
            raise

    def test12_PyTorch(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(10).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(3).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_minor = ROOT.RDataFrame("myTree", file_name1)
            df_major = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [5.0, 5.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [
                1.0,
                100.0,
                9.0,
                300.0,
                1.0,
                100.0,
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
            ]
            results_y_val = [25.0, 500.0, 25.0, 500.0, 144.0, 1200.0, 196.0, 1400.0, 256.0, 1600.0, 324.0, 1800.0]
            results_z_train = [10.0, 30.0, 10.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
            results_z_val = [50.0, 50.0, 120.0, 140.0, 160.0, 180.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test13_TensorFlow(self):
        file_name1 = "multiple_target_columns_major.root"
        file_name2 = "multiple_target_columns_minor.root"

        ROOT.RDataFrame(10).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name1)
        ROOT.RDataFrame(3).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1 * b1").Define(
            "b3", "(double) b1 * 10"
        ).Define("b4", "(double) b3 * 10").Snapshot("myTree", file_name2)
        try:
            df_minor = ROOT.RDataFrame("myTree", file_name1)
            df_major = ROOT.RDataFrame("myTree", file_name2)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                [df_minor, df_major],
                batch_size=2,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
            )

            results_x_train = [1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            results_x_val = [5.0, 5.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [
                1.0,
                100.0,
                9.0,
                300.0,
                1.0,
                100.0,
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
            ]
            results_y_val = [25.0, 500.0, 25.0, 500.0, 144.0, 1200.0, 196.0, 1400.0, 256.0, 1600.0, 324.0, 1800.0]
            results_z_train = [10.0, 30.0, 10.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
            results_z_val = [50.0, 50.0, 120.0, 140.0, 160.0, 180.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []
            collected_z_train = []
            collected_z_val = []

            iter_train = iter(gen_train)
            iter_val = iter(gen_validation)

            for _ in range(self.n_train_batch):
                x, y, z = next(iter_train)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
                collected_z_train.append(z.tolist())

            for _ in range(self.n_val_batch):
                x, y, z = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 2))
                self.assertTrue(z.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
                collected_z_val.append(z.tolist())

            x, y, z = next(iter_train)
            self.assertTrue(x.shape == (self.train_remainder, 1))
            self.assertTrue(y.shape == (self.train_remainder, 2))
            self.assertTrue(z.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
            flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)
            self.assertEqual(results_z_train, flat_z_train)
            self.assertEqual(results_z_val, flat_z_val)

            num_major_train = sum(np.array(flat_x_train) % 2 == 0)
            num_minor_train = sum(np.array(flat_x_train) % 2 != 0)
            num_major_val = sum(np.array(flat_x_val) % 2 == 0)
            num_minor_val = sum(np.array(flat_x_val) % 2 != 0)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(file_name1)
            self.teardown_file(file_name2)

        except:
            self.teardown_file(file_name1)
            self.teardown_file(file_name2)
            raise

    def test14_big_data_replacement_false(self):
        file_name1 = "big_data_major.root"
        file_name2 = "big_data_minor.root"
        tree_name = "myTree"

        entries_in_rdf_major = randrange(10000, 30000)
        entries_in_rdf_minor = randrange(8000, 9999)
        batch_size = randrange(100, 501)
        sampling_ratio = round(uniform(0.1, 2), 2)

        error_message = f"\n Batch size: {batch_size}\
            Number of major entries: {entries_in_rdf_major} \
            Number of minor entries: {entries_in_rdf_minor}"

        def define_rdf_major(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def define_rdf_minor(num_of_entries, file_name):
            ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define(
                "b2", "(double) rdfentry_ * 2"
            ).Define("b3", "(int) rdfentry_ + 10192").Define("b4", "(int) -rdfentry_").Define(
                "b5", "(double) -rdfentry_ - 10192"
            ).Snapshot(tree_name, file_name)

        def test(size_of_batch, num_of_entries_major, num_of_entries_minor, sampling_ratio):
            define_rdf_major(num_of_entries_major, file_name1)
            define_rdf_minor(num_of_entries_minor, file_name2)

            try:
                df1 = ROOT.RDataFrame(tree_name, file_name1)
                df2 = ROOT.RDataFrame(tree_name, file_name2)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df1, df2],
                    batch_size=size_of_batch,
                    target=["b3", "b5"],
                    weights="b1",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="oversampling",
                    sampling_ratio=sampling_ratio,
                )

                collected_z_train = []
                collected_z_val = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - 1 if train_remainder else gen_train.number_of_batches
                n_val_batches = (
                    gen_validation.number_of_batches - 1 if val_remainder else gen_validation.number_of_batches
                )

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_train.append(z.tolist())

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (train_remainder, 1), error_message)
                    collected_z_train.append(z.tolist())

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2), error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2), error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1), error_message + f" row: {i} z shape: {z.shape}")
                    collected_z_val.append(z.tolist())

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (val_remainder, 1), error_message)
                    collected_z_val.append(z.tolist())

                flat_z_train = [z for zl in collected_z_train for zs in zl for z in zs]
                flat_z_val = [z for zl in collected_z_val for zs in zl for z in zs]

                num_major_train = sum(np.array(flat_z_train) % 2 == 0)
                num_minor_train = sum(np.array(flat_z_train) % 2 != 0)
                num_major_val = sum(np.array(flat_z_val) % 2 == 0)
                num_minor_val = sum(np.array(flat_z_val) % 2 != 0)

                # check if there are no duplicate entries (replacement=False)
                self.assertLessEqual(len(set(flat_z_train)), len(flat_z_train))
                self.assertLessEqual(len(set(flat_z_val)), len(flat_z_val))

                # check if the sampling stategy is correct
                self.assertEqual(round((num_minor_train / num_major_train), 2), sampling_ratio)
                self.assertEqual(round((num_minor_val / num_major_val), 2), sampling_ratio)

                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
            except:
                self.teardown_file(file_name1)
                self.teardown_file(file_name2)
                raise

        test(batch_size, entries_in_rdf_major, entries_in_rdf_minor, sampling_ratio)

    def test15_two_runs_set_seed(self):
        self.create_file_major()
        self.create_file_minor()

        try:
            both_runs_collected_x_val = []
            both_runs_collected_y_val = []

            df_major = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name2)

            for _ in range(2):
                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    [df_major, df_minor],
                    batch_size=2,
                    target="b2",
                    validation_split=0.4,
                    shuffle=False,
                    drop_remainder=False,
                    load_eager=True,
                    sampling_type="oversampling",
                    sampling_ratio=0.5,
                )

                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(self.n_train_batch):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                for _ in range(self.n_val_batch):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (2, 1))
                    self.assertTrue(y.shape == (2, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_train)
                self.assertTrue(x.shape == (self.train_remainder, 1))
                self.assertTrue(y.shape == (self.train_remainder, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

                flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
                flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
                flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
                flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

                self.assertEqual(len(flat_x_train), 9)
                self.assertEqual(len(flat_x_val), 6)
                self.assertEqual(len(flat_y_train), 9)
                self.assertEqual(len(flat_y_val), 6)

                both_runs_collected_x_val.append(collected_x_val)
                both_runs_collected_y_val.append(collected_y_val)
            self.assertEqual(both_runs_collected_x_val[0], both_runs_collected_x_val[1])
            self.assertEqual(both_runs_collected_y_val[0], both_runs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)

    def test16_vector_padding(self):
        self.create_vector_file_major()
        self.create_vector_file_minor()

        try:
            df_major = ROOT.RDataFrame(self.tree_name, self.file_name4)
            df_minor = ROOT.RDataFrame(self.tree_name, self.file_name5)
            max_vec_sizes = {"v1": 3, "v2": 2}

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                [df_major, df_minor],
                batch_size=2,
                target="b1",
                validation_split=0.4,
                max_vec_sizes=max_vec_sizes,
                shuffle=False,
                drop_remainder=False,
                load_eager=True,
                sampling_type="oversampling",
                sampling_ratio=0.5,
                replacement=False,
            )

            results_x_train = [
                10.0,
                100.0,
                0.0,
                1000.0,
                10000.0,
                11.0,
                110.0,
                0.0,
                1100.0,
                11000.0,
                10.0,
                100.0,
                0.0,
                1000.0,
                10000.0,
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
            ]
            results_x_val = [
                12.0,
                120.0,
                0.0,
                1200.0,
                12000.0,
                12.0,
                120.0,
                0.0,
                1200.0,
                12000.0,
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
            results_y_train = [10.0, 11.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            results_y_val = [12.0, 12.0, 6.0, 7.0, 8.0, 9.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)

            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (2, 5))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (2, 5))
                self.assertTrue(y.shape == (2, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(train_iter)
            self.assertTrue(x.shape == (self.train_remainder, 5))
            self.assertTrue(y.shape == (self.train_remainder, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())

            flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            num_major_train = sum(np.array(flat_y_train) < 10)
            num_minor_train = sum(np.array(flat_y_train) >= 10)
            num_major_val = sum(np.array(flat_y_val) < 10)
            num_minor_val = sum(np.array(flat_y_val) >= 10)

            self.assertEqual(num_major_train, 6)
            self.assertEqual(num_minor_train, 3)
            self.assertEqual(num_major_val, 4)
            self.assertEqual(num_minor_val, 2)

            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)

        except:
            self.teardown_file(self.file_name4)
            self.teardown_file(self.file_name5)
            raise


if __name__ == "__main__":
    unittest.main()
