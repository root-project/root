import unittest
import os
import ROOT
import numpy as np
from random import randrange


class RBatchGeneratorMultipleFiles(unittest.TestCase):

    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    tree_name = "mytree"

    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    # Helpers
    def define_rdf(self, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries)\
            .Define("b1", "(int) rdfentry_")\
            .Define("b2", "(double) b1*b1")

        return df

    def create_file(self, num_of_entries=10):
        self.define_rdf(num_of_entries).Snapshot(
            self.tree_name, self.file_name1)

    def create_5_entries_file(self):
        df1 = ROOT.RDataFrame(5)\
            .Define("b1", "(int) rdfentry_ + 10")\
            .Define("b2", "(double) b1 * b1")\
            .Snapshot(self.tree_name, self.file_name2)

    def teardown_file(self, file):
        os.remove(file)

    def test01_each_element_is_generated_unshuffled(self):
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
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
                df,
                batch_size=3,
                chunk_size=5,
                target="b2",
                validation_split=0.4,
                shuffle=True,
                drop_remainder=False
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

            flat_x_train = {
                x for xl in collected_x_train for xs in xl for x in xs}
            flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
            flat_y_train = {
                y for yl in collected_y_train for ys in yl for y in ys}
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
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
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
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=True
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
            df = ROOT.RDataFrame(
                self.tree_name, [self.file_name1, self.file_name2])

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0, 13.0, 14.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0,
                               36.0, 49.0, 100.0, 121.0, 144.0]
            results_y_val = [9.0, 16.0, 64.0, 81.0, 169.0, 196.0]

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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
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

        ROOT.RDataFrame(10)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Define("b3", "(double) rdfentry_ * 10")\
            .Define("b4", "(double) b3 * 10")\
            .Snapshot("myTree", file_name)
        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0,
                               200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [
                z for zl in collected_z_train for zs in zl for z in zs]
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

        ROOT.RDataFrame(10)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Define("b3", "(double) rdfentry_ * 10")\
            .Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 0.0, 1.0, 10.0, 2.0,
                               20.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0]
            results_x_val = [3.0, 30.0, 4.0, 40.0, 8.0, 80.0, 9.0, 90.0]
            results_y_train = [0.0, 1.0, 4.0, 25.0, 36.0, 49.]
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
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
                chunk_size=5,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
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

        ROOT.RDataFrame(20)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Snapshot(tree_name, file_name)

        try:
            df = ROOT.RDataFrame(tree_name, file_name)

            dff = df.Filter("b1 % 2 == 0", "name")

            gen_train, _ = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                dff,
                batch_size=3,
                chunk_size=9,
                target="b2",
                validation_split=0,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 2.0, 4.0, 6.0,
                               8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            results_y_train = [0.0, 4.0, 16.0, 36.0,
                               64.0, 100.0, 144.0, 196.0, 256.0, 324.0]

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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]

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
                target="b2",
                validation_split=0.4,
                shuffle=True,
                drop_remainder=False
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

                flat_x_train = {
                    x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {
                    x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {
                    y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {
                    y for yl in collected_y_val for ys in yl for y in ys}

                self.assertEqual(len(flat_x_train), 6)
                self.assertEqual(len(flat_x_val), 4)
                self.assertEqual(len(flat_y_train), 6)
                self.assertEqual(len(flat_y_val), 4)

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(
                both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(
                both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
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
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            number_of_training_batches = 0
            number_of_validation_batches = 0

            for _ in gen_train:
                number_of_training_batches += 1

            for _ in gen_validation:
                number_of_validation_batches += 1

            self.assertEqual(gen_train.number_of_batches,
                             number_of_training_batches)
            self.assertEqual(gen_validation.number_of_batches,
                             number_of_validation_batches)
            self.assertEqual(gen_train.last_batch_no_of_rows, 0)
            self.assertEqual(gen_validation.last_batch_no_of_rows, 1)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

    def test12_PyTorch(self):
        import torch

        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Define("b3", "(double) rdfentry_ * 10")\
            .Define("b4", "(double) b3 * 10")\
            .Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
                df,
                batch_size=3,
                chunk_size=5,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0,
                               200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [
                z for zl in collected_z_train for zs in zl for z in zs]
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
        import tensorflow as tf

        file_name = "multiple_target_columns.root"

        ROOT.RDataFrame(10)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Define("b3", "(double) rdfentry_ * 10")\
            .Define("b4", "(double) b3 * 10")\
            .Snapshot("myTree", file_name)

        try:
            df = ROOT.RDataFrame("myTree", file_name)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateTFDatasets(
                df,
                batch_size=3,
                chunk_size=5,
                target=["b2", "b4"],
                weights="b3",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
            results_x_val = [3.0, 4.0, 8.0, 9.0, 0.0, 0.0]
            results_y_train = [0.0, 0.0, 1.0, 100.0, 4.0,
                               200.0, 25.0, 500.0, 36.0, 600.0, 49.0, 700.0]
            results_y_val = [9.0, 300.0, 16.0, 400.0, 64.0,
                             800.0, 81.0, 900.0, 0.0, 0.0, 0.0, 0.0]
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

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
            flat_z_train = [
                z for zl in collected_z_train for zs in zl for z in zs]
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
            ROOT.RDataFrame(num_of_entries)\
                .Define("b1", "(int) rdfentry_")\
                .Define("b2", "(double) rdfentry_ * 2")\
                .Define("b3", "(int) rdfentry_ + 10192")\
                .Define("b4", "(int) -rdfentry_")\
                .Define("b5", "(double) -rdfentry_ - 10192")\
                .Snapshot(tree_name, file_name)

        def test(size_of_batch, size_of_chunk, num_of_entries):
            define_rdf(num_of_entries)

            try:
                df = ROOT.RDataFrame(tree_name, file_name)

                gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                    df,
                    batch_size=size_of_batch,
                    chunk_size=size_of_chunk,
                    target=["b3", "b5"],
                    weights="b2",
                    validation_split=0.3,
                    shuffle=False,
                    drop_remainder=False
                )

                collect_x = []

                train_remainder = gen_train.last_batch_no_of_rows
                val_remainder = gen_validation.last_batch_no_of_rows

                n_train_batches = gen_train.number_of_batches - \
                    1 if train_remainder else gen_train.number_of_batches
                n_val_batches = gen_validation.number_of_batches - \
                    1 if val_remainder else gen_validation.number_of_batches

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for i in range(n_train_batches):
                    x, y, z = next(iter_train)

                    self.assertTrue(x.shape == (size_of_batch, 2),
                                    error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2),
                                    error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1),
                                    error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(
                        np.all(x[:, 0]*(-1) == x[:, 1]), error_message + f" row: {i}")
                    self.assertTrue(
                        np.all(x[:, 0]+10192 == y[:, 0]), error_message + f" row: {i}")
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(
                        np.all(x[:, 0]*2 == z[:, 0]), error_message + f" row: {i}")

                    collect_x.extend(list(x[:, 0]))

                if train_remainder:
                    x, y, z = next(iter_train)
                    self.assertTrue(x.shape == (
                        train_remainder, 2), error_message)
                    self.assertTrue(y.shape == (
                        train_remainder, 2), error_message)
                    self.assertTrue(z.shape == (
                        train_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                for _ in range(n_val_batches):
                    x, y, z = next(iter_val)

                    self.assertTrue(x.shape == (size_of_batch, 2),
                                    error_message + f" row: {i} x shape: {x.shape}")
                    self.assertTrue(y.shape == (size_of_batch, 2),
                                    error_message + f" row: {i} y shape: {y.shape}")
                    self.assertTrue(z.shape == (size_of_batch, 1),
                                    error_message + f" row: {i} z shape: {z.shape}")

                    self.assertTrue(
                        np.all(x[:, 0]*(-1) == x[:, 1]), error_message)
                    self.assertTrue(
                        np.all(x[:, 0]+10192 == y[:, 0]), error_message)
                    # self.assertTrue(np.all(x[:,0]*(-1)-10192==y[:,1]), error_message)
                    self.assertTrue(
                        np.all(x[:, 0]*2 == z[:, 0]), error_message)

                    collect_x.extend(list(x[:, 0]))

                if val_remainder:
                    x, y, z = next(iter_val)
                    self.assertTrue(x.shape == (
                        val_remainder, 2), error_message)
                    self.assertTrue(y.shape == (
                        val_remainder, 2), error_message)
                    self.assertTrue(z.shape == (
                        val_remainder, 1), error_message)
                    collect_x.extend(list(x[:, 0]))

                self.assertTrue(set(collect_x) == set(i for i in range(num_of_entries)), f"collected length: {len(set(collect_x))}\
                                 generated length {len(set(i for i in range(num_of_entries)))}")

            except:
                self.teardown_file(file_name)
                raise

        test(batch_size, chunk_size, entries_in_rdf)


if __name__ == '__main__':
    unittest.main()
