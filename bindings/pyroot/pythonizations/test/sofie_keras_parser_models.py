import os
import shutil
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from parser_test_function import generate_and_test_inference

# Test the SOFIE Keras parser on full multi-layer models (sequential and
# functional), comparing the SOFIE inference results with the outputs of the
# original Keras models. In contrast to sofie_keras_parser.py, which tests the
# individual layers one by one, the models here are trained and use batch
# sizes larger than one, so the batch size stored in the saved model is also
# exercised.
#
# This is the Python translation of the former C++ googletest
# tmva/sofie/test/TestRModelParserKeras.C.

WORK_DIR = "keras_parser_models"

TOLERANCE = 1e-6


def generate_keras_models(dst_dir):
    from tensorflow.keras.layers import (
        Activation,
        Add,
        BatchNormalization,
        Concatenate,
        Conv2D,
        Dense,
        Input,
        LeakyReLU,
        Multiply,
        ReLU,
        Reshape,
        Subtract,
    )
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import SGD

    random_generator = np.random.RandomState(0)

    def train_and_save(model, x_train, y_train, name, batch_size):
        model.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=0.01))
        model.fit(x_train, y_train, verbose=0, epochs=10, batch_size=batch_size)
        model.save(f"{dst_dir}/{name}.keras")

    # Functional model
    input = Input(shape=(8,), batch_size=2)
    x = Dense(16)(input)
    x = Activation("relu")(x)
    x = Dense(32)(x)
    x = ReLU()(x)
    output = Dense(4, activation="selu")(x)
    model = Model(inputs=input, outputs=output)
    train_and_save(model, random_generator.rand(2, 8), random_generator.rand(2, 4), "KerasModelFunctional", 2)

    # Sequential model
    model = Sequential()
    model.add(Dense(8))
    model.add(ReLU())
    model.add(Dense(6))
    model.add(Activation("sigmoid"))
    train_and_save(model, random_generator.rand(4, 8), random_generator.rand(4, 6), "KerasModelSequential", 4)

    # Batch normalization model
    model = Sequential()
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Dense(2))
    train_and_save(model, random_generator.rand(4, 4), random_generator.rand(4, 2), "KerasModelBatchNorm", 2)

    # Conv2D model with valid padding
    model = Sequential()
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=(4, 4, 1), padding="valid"))
    train_and_save(
        model, random_generator.rand(1, 4, 4, 1), random_generator.rand(1, 2, 2, 8), "KerasModelConv2D_Valid", 2
    )

    # Conv2D model with same padding
    model = Sequential()
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=(4, 4, 1), padding="same"))
    train_and_save(
        model, random_generator.rand(1, 4, 4, 1), random_generator.rand(1, 4, 4, 8), "KerasModelConv2D_Same", 2
    )

    # Conv2D model with same padding and dilation
    model = Sequential()
    model.add(Conv2D(4, kernel_size=3, activation="relu", input_shape=(8, 8, 1), padding="same", dilation_rate=2))
    train_and_save(
        model, random_generator.rand(1, 8, 8, 1), random_generator.rand(1, 8, 8, 4), "KerasModelConv2D_SameDilated", 2
    )

    # Reshape model
    model = Sequential()
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=(4, 4, 1), padding="same"))
    model.add(Reshape((32, 4)))
    train_and_save(model, random_generator.rand(1, 4, 4, 1), random_generator.rand(1, 32, 4), "KerasModelReshape", 2)

    # Concatenate model
    input_1 = Input(shape=(2,))
    dense_1 = Dense(3)(input_1)
    input_2 = Input(shape=(2,))
    dense_2 = Dense(3)(input_2)
    concat = Concatenate(axis=1)([dense_1, dense_2])
    model = Model(inputs=[input_1, input_2], outputs=concat)
    train_and_save(
        model,
        [random_generator.rand(1, 2), random_generator.rand(1, 2)],
        random_generator.rand(1, 6),
        "KerasModelConcatenate",
        1,
    )

    # Binary operators model (Add, Subtract, Multiply)
    input_1 = Input(shape=(2,))
    input_2 = Input(shape=(2,))
    add = Add()([input_1, input_2])
    subtract = Subtract()([add, input_1])
    multiply = Multiply()([subtract, input_2])
    model = Model(inputs=[input_1, input_2], outputs=multiply)
    train_and_save(
        model,
        [random_generator.rand(2, 2), random_generator.rand(2, 2)],
        random_generator.rand(2, 2),
        "KerasModelBinaryOp",
        2,
    )

    # Model with different activations (tanh, LeakyReLU, softmax)
    input = Input(shape=(8,))
    x = Dense(16, activation="tanh")(input)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    output = Dense(4, activation="softmax")(x)
    model = Model(inputs=input, outputs=output)
    train_and_save(model, random_generator.rand(1, 8), random_generator.rand(1, 4), "KerasModelActivations", 1)

    # Swish activation model
    model = Sequential()
    model.add(Dense(64, activation="swish", input_shape=(8,)))
    model.add(Dense(32, activation="swish"))
    model.add(Dense(1, activation="swish"))
    train_and_save(model, random_generator.rand(1, 8), random_generator.rand(1, 1), "KerasModelSwish", 1)


class SOFIE_Keras_Parser_Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isdir(WORK_DIR):
            shutil.rmtree(WORK_DIR)
        os.makedirs(WORK_DIR)
        print("Generating Keras models for testing")
        generate_keras_models(WORK_DIR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WORK_DIR)

    def run_model_test(self, model_name, input_tensors, batch_size):
        generate_and_test_inference(
            f"{WORK_DIR}/{model_name}.keras",
            WORK_DIR,
            batch_size=batch_size,
            input_tensors=input_tensors,
            atol=TOLERANCE,
        )

    def test_sequential(self):
        # input is 8 x batch size that is fixed to be 4
        input_tensor = np.array(
            [
                0.12107884,
                0.89718615,
                0.89123899,
                0.32197549,
                0.17891638,
                0.83555135,
                0.98680066,
                0.14496809,
                0.07255503,
                0.55386989,
                0.6628149,
                0.29843291,
                0.71059786,
                0.44043452,
                0.13792047,
                0.93007397,
                0.16799397,
                0.75473803,
                0.43203355,
                0.68360968,
                0.83879351,
                0.0558927,
                0.57500447,
                0.49063431,
                0.63637339,
                0.94483464,
                0.11032887,
                0.22424818,
                0.50972592,
                0.04671024,
                0.39230661,
                0.80500943,
            ],
            dtype=np.float32,
        ).reshape(4, 8)
        self.run_model_test("KerasModelSequential", [input_tensor], batch_size=4)

    def test_functional(self):
        input_tensor = np.array(
            [
                0.60828574,
                0.50069386,
                0.75186709,
                0.14968806,
                0.7692464,
                0.77027585,
                0.75095316,
                0.96651197,
                0.38536308,
                0.95565917,
                0.62796356,
                0.13818375,
                0.65484891,
                0.89220363,
                0.23879365,
                0.00635323,
            ],
            dtype=np.float32,
        ).reshape(2, 8)
        self.run_model_test("KerasModelFunctional", [input_tensor], batch_size=2)

    def test_batch_norm(self):
        input_tensor = np.array(
            [0.22308163, 0.95274901, 0.44712538, 0.84640867, 0.69947928, 0.29743695, 0.81379782, 0.39650574],
            dtype=np.float32,
        ).reshape(2, 4)
        self.run_model_test("KerasModelBatchNorm", [input_tensor], batch_size=2)

    def test_conv2d_valid_padding(self):
        input_tensor = np.ones((1, 4, 4, 1), dtype=np.float32)
        self.run_model_test("KerasModelConv2D_Valid", [input_tensor], batch_size=1)

    def test_conv2d_same_padding(self):
        input_tensor = np.ones((1, 4, 4, 1), dtype=np.float32)
        self.run_model_test("KerasModelConv2D_Same", [input_tensor], batch_size=1)

    def test_conv2d_same_padding_dilated(self):
        input_tensor = np.ones((1, 8, 8, 1), dtype=np.float32)
        self.run_model_test("KerasModelConv2D_SameDilated", [input_tensor], batch_size=1)

    def test_reshape(self):
        input_tensor = np.ones((1, 4, 4, 1), dtype=np.float32)
        self.run_model_test("KerasModelReshape", [input_tensor], batch_size=1)

    def test_concatenate(self):
        input_tensors = [np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32)]
        self.run_model_test("KerasModelConcatenate", input_tensors, batch_size=1)

    def test_binary_op(self):
        # test with batch size = 2, input shapes are (2,2)
        input_tensors = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        ]
        self.run_model_test("KerasModelBinaryOp", input_tensors, batch_size=2)

    def test_activations(self):
        input_tensor = np.ones((1, 8), dtype=np.float32)
        self.run_model_test("KerasModelActivations", [input_tensor], batch_size=1)

    def test_swish(self):
        input_tensor = np.ones((1, 8), dtype=np.float32)
        self.run_model_test("KerasModelSwish", [input_tensor], batch_size=1)


if __name__ == "__main__":
    unittest.main()
