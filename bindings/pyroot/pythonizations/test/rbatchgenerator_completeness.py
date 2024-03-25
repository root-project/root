import unittest
import os
import ROOT

class RBatchGeneratorMultipleFiles(unittest.TestCase):

    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    tree_name = "mytree"

    # Helpers
    def create_10_entries_file(self):
        df2 = ROOT.RDataFrame(10)\
            .Define("b1", "(int) rdfentry_")\
            .Define("b2", "(double) b1*b1")\
            .Snapshot(self.tree_name, self.file_name1)
    
    # def create_20_entries_file(self):
    #     df1 = ROOT.RDataFrame(20)\
    #         .Define("b1", "(int) rdfentry_ + 10")\
    #         .Define("b2", "(double) rdfentry_ * rdfentry_")\
    #         .Snapshot(self.tree_name, self.file_name2)
        
        #print(df1.Describe())
        #print(df2.Describe())

        #df2.Display("",).Print()


    def teardown_file(self, file):
        os.remove(file)


    def test01_each_element_is_generated_unshuffled(self):
        self.create_10_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=3,
        chunk_size=5,
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
        self.create_10_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=3,
        chunk_size=5,
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

        self.teardown_file(self.file_name1)

    def test03_next_iteration(self):
        self.create_10_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=3,
        chunk_size=5,
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

        self.teardown_file(self.file_name1)
    
    def test04_chunk_input_smaller_than_batch_size(self):
        self.create_10_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=3,
        chunk_size=3,
        target="b2",
        validation_split=0.3,
        shuffle=False,
        drop_remainder=False
        )

        for x, y in gen_train:
            pass

        for x, y in gen_validation:
            pass

        self.teardown_file(self.file_name1)
    
    def test05_dropping_remainder(self):
        self.create_10_entries_file()

        gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=3,
        chunk_size=5,
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

        self.teardown_file(self.file_name1)


if __name__ == 'main':
    unittest.main()
