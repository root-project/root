import unittest
import os
import ROOT

class RBatchGeneratorMultipleFiles(unittest.TestCase):

    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    tree_name = "mytree"

    # Helpers
    def define_rdf(self, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries)\
            .Define("b1", "(int) rdfentry_")\
            .Define("b2", "(double) b1*b1")
        
        return df

    def create_file(self, num_of_entries=10):
        self.define_rdf(num_of_entries).Snapshot(self.tree_name, self.file_name1)
    
    def create_5_entries_file(self):
        df1 = ROOT.RDataFrame(5)\
            .Define("b1", "(int) rdfentry_ + 10")\
            .Define("b2", "(double) b1 * b1")\
            .Snapshot(self.tree_name, self.file_name2)

    def teardown_file(self, file):
        os.remove(file)


    def test01_each_element_is_generated_unshuffled(self):
        self.create_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        tree_name=self.tree_name,
        file_names=self.file_name1,
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )

        results_x_train = [2.0, 3.0, 4.0, 7.0, 8.0, 9.0]
        results_x_val = [0.0, 1.0, 5.0, 6.0]
        results_y_train = [4.0, 9.0, 16.0, 49.0, 64.0, 81.0]
        results_y_val = [0.0, 1.0, 25.0, 36.0]
        
        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        for x, y in gen_train:
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
        
        for x, y in gen_validation:
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

    def test02_each_element_is_generated_shuffled(self):
        df = self.define_rdf()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        rdataframe=df,
        target="b2",
        validation_split=0.3,
        shuffle=True,
        drop_remainder=False
        )
        
        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        for x, y in gen_train:
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
        
        for x, y in gen_validation:
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

        flat_x_train = {x for xl in collected_x_train for xs in xl for x in xs}
        flat_x_val = {x for xl in collected_x_val for xs in xl for x in xs}
        flat_y_train = {y for yl in collected_y_train for ys in yl for y in ys}
        flat_y_val = {y for yl in collected_y_val for ys in yl for y in ys}
        
        self.assertEqual(len(flat_x_train),6)
        self.assertEqual(len(flat_x_val),4)
        self.assertEqual(len(flat_y_train),6)
        self.assertEqual(len(flat_y_val),4)

    def test03_next_iteration(self):
        df = self.define_rdf()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        rdataframe=df,
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )

        results_x_train = [2.0, 3.0, 4.0, 7.0, 8.0, 9.0]
        results_x_val = [0.0, 1.0, 5.0, 6.0]
        results_y_train = [4.0, 9.0, 16.0, 49.0, 64.0, 81.0]
        results_y_val = [0.0, 1.0, 25.0, 36.0]
        
        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        print("Training")
        while True:
            try:
                x, y = next(gen_train)
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
            except StopIteration:
                break
        
        print("Validation")
        while True:
            try:
                x, y = next(gen_validation)
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
            except StopIteration:
                break


        flat_x_train = [x for xl in collected_x_train for xs in xl for x in xs]
        flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
        flat_y_train = [y for yl in collected_y_train for ys in yl for y in ys]
        flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]
        
        self.assertEqual(results_x_train, flat_x_train)
        self.assertEqual(results_x_val, flat_x_val)
        self.assertEqual(results_y_train, flat_y_train)
        self.assertEqual(results_y_val, flat_y_val)
    
    def test04_chunk_input_smaller_than_batch_size(self):
        df = self.define_rdf()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=3,
        rdataframe=df,
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )

        for x, y in gen_train:
            pass

        for x, y in gen_validation:
            pass
    
    def test05_dropping_remainder(self):
        df = self.define_rdf()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        rdataframe=df,
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=True
        )
        
        collected_x = []
        collected_y = []

        for x, y in gen_train:
            collected_x.append(x)
            collected_y.append(y)
        
        for x, y in gen_validation:
            collected_x.append(x)
            collected_y.append(y)
        
        print(len(collected_x))
        print(len(collected_y))
        
        self.assertEqual(len(collected_x), 3)
        self.assertEqual(len(collected_y), 3)
    
    def test06_more_than_one_file(self):
        self.create_file()
        self.create_5_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        tree_name=self.tree_name,
        file_names=[self.file_name1, self.file_name2],
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )

        results_x_train = [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0]
        results_x_val = [0.0, 1.0, 5.0, 6.0, 10.0, 11.0]
        results_y_train = [4.0, 9.0, 16.0, 49.0, 64.0, 81.0, 144.0, 169.0, 196.0]
        results_y_val = [0.0, 1.0, 25.0, 36.0, 100.0, 121.0]
        
        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []

        for x, y in gen_train:
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
        
        for x, y in gen_validation:
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
    
    def test07_multiple_target_columns(self):
        df = ROOT.RDataFrame(10)\
            .Define("b1", "(Short_t) rdfentry_")\
            .Define("b2", "(UShort_t) b1 * b1")\
            .Define("b3", "(double) rdfentry_ * 10")\
            .Define("b4", "(double) b3 * 10")
        
        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        batch_size=3,
        chunk_size=5,
        rdataframe=df,
        target=["b2","b4"],
        weights="b3",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )
        
        results_x_train = [2.0, 3.0, 4.0, 7.0, 8.0, 9.0]
        results_x_val = [0.0, 1.0, 5.0, 6.0]
        results_y_train = [4.0, 200.0, 9.0, 300.0, 16.0, 400.0, 49.0, 700.0, 64.0, 800.0, 81.0, 900.0]
        results_y_val = [0.0, 0.0, 1.0, 100.0, 25.0, 500.0, 36.0, 600.0]
        results_z_train = [20.0, 30.0, 40.0, 70.0, 80.0, 90.0]
        results_z_val = [0.0, 10.0, 50.0, 60.0]

        collected_x_train = []
        collected_x_val = []
        collected_y_train = []
        collected_y_val = []
        collected_z_train = []
        collected_z_val = []

        for x, y, z in gen_train:
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            collected_z_train.append(z.tolist())
        
        for x, y, z in gen_validation:
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


if __name__ == 'main':
    unittest.main()
