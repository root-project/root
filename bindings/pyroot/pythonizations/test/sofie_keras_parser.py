import unittest
import os
import shutil

from ROOT._pythonization._tmva._sofie._parser._keras.parser_test_function import generate_and_test_inference
from ROOT._pythonization._tmva._sofie._parser._keras.generate_keras_functional import generate_keras_functional
from ROOT._pythonization._tmva._sofie._parser._keras.generate_keras_sequential import generate_keras_sequential


def make_testname(test_case: str):
    test_case_name = test_case.replace("_", " ").removesuffix(".h5")
    return test_case_name

models = [
    "BatchNorm1D",
    "Conv2D_channels_first",
    "Conv2D_channels_last",
    "Conv2D_padding_same",
    "Conv2D_padding_valid",
    "Dense",
    "Flatten",
    # "GRU",
    "LeakyReLU",
    # "LSTM",
    "MaxPool2D_channels_first",
    "MaxPool2D_channels_last",
    "Permute",
    "Relu",
    "Reshape",
    "Selu",
    "Sigmoid",
    # "SimpleRNN",
    "Softmax",
    "Swish",
    "Tanh",
] + [f"Layer_Combination_{i}" for i in range(1, 4)]

class SOFIE_Keras_Parser(unittest.TestCase):
    
    def setUp(self):
        base_dir = self._testMethodName[5:]
        os.makedirs(base_dir + "/input_models")
        os.makedirs(base_dir + "/generated_header_files_dir") 
    
    def run_model_tests(self, model_type: str, generate_function, model_list):
        generate_function(f"{model_type}/input_models")
        for keras_model in model_list:
            keras_model_name = f"{model_type.capitalize()}_{keras_model}_test.h5"
            keras_model_path = f"{model_type}/input_models/" + keras_model_name
            with self.subTest(msg=make_testname(keras_model_name)):
                generate_and_test_inference(keras_model_path, f"{model_type}/generated_header_files_dir")
    
    def test_sequential(self):
        sequential_models = models
        self.run_model_tests("sequential", generate_keras_sequential, sequential_models)
        
    def test_functional(self):
        functional_models = models + ["Add", "Concat", "Multiply", "Subtract"]
        self.run_model_tests("functional", generate_keras_functional, functional_models)
    
    # def tearDown(self):
    #     base_dir = self._testMethodName[5:]
    #     shutil.rmtree(base_dir)
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("sequential")
        shutil.rmtree("functional")

if __name__ == "__main__":
    unittest.main()