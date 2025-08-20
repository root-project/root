# functional_models.py
import numpy as np
from keras import models, layers, activations

def generate_keras_functional(dst_dir):
    # Helper training function
    def train_and_save(model, name):
        # Handle multiple inputs dynamically
        if isinstance(model.input_shape, list):
            x_train = [np.random.rand(32, *shape[1:]) for shape in model.input_shape]
        else:
            x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        # print(dst_dir)
        model.save(f"{dst_dir}/{name}.h5")
        # print(f"Saved {name}.h5")

    # # 1. Dropout (to test SOFIE's Identity operator)
    # inp = layers.Input(shape=(10,))
    # out = layers.Dropout(0.5)(inp)
    # model = models.Model(inputs=inp, outputs=out)
    # train_and_save(model, "Functional_Dropout_test")

    # 2. Binary Operators
    # Add
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Add()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Functional_Add_test")

    # Subtract
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Subtract()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Functional_Subtract_test")

    # Multiply
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Multiply()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Functional_Multiply_test")

    # 3. Concat
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Concatenate()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Functional_Concat_test")

    # 4. Reshape
    inp = layers.Input(shape=(4, 5))
    out = layers.Reshape((2, 10))(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Reshape_test")

    # 5. Flatten
    inp = layers.Input(shape=(4, 5))
    out = layers.Flatten()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Flatten_test")

    # 6. BatchNorm 1D
    inp = layers.Input(shape=(10,))
    out = layers.BatchNormalization()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_BatchNorm1D_test")

    # 7. Activation Functions
    for act in ['relu', 'selu', 'sigmoid', 'softmax', 'tanh']:
        inp = layers.Input(shape=(10,))
        out = layers.Activation(act)(inp)
        model = models.Model(inp, out)
        train_and_save(model, f"Functional_{act.capitalize()}_test")

    # LeakyReLU
    inp = layers.Input(shape=(10,))
    out = layers.LeakyReLU()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_LeakyReLU_test")

    # Swish
    inp = layers.Input(shape=(10,))
    out = layers.Activation(activations.swish)(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Swish_test")

    # 8. Permute
    inp = layers.Input(shape=(3, 4, 5))
    out = layers.Permute((2, 1, 3))(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Permute_test")

    # 9. Dense
    inp = layers.Input(shape=(10,))
    out = layers.Dense(5)(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Dense_test")

    # 10. Conv2D channels_last
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Conv2D_channels_last_test")

    # 10. Conv2D channels_first
    inp = layers.Input(shape=(3, 8, 8))
    out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_first')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Conv2D_channels_first_test")
    
    # Conv2D padding_same
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Conv2D_padding_same_test")
    
    # Conv2D padding_valid
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='valid', data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_Conv2D_padding_valid_test")

    # 11. MaxPooling2D channels_last
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_MaxPool2D_channels_last_test")

    # 11. MaxPooling2D channels_first
    inp = layers.Input(shape=(3, 8, 8))
    out = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Functional_MaxPool2D_channels_first_test")

    # # 12. RNN - SimpleRNN
    # inp = layers.Input(shape=(5, 3))
    # out = layers.SimpleRNN(4, return_sequences=True)(inp)
    # model = models.Model(inp, out)
    # train_and_save(model, "Functional_SimpleRNN_test")

    # # 12. RNN - LSTM
    # inp = layers.Input(shape=(5, 3))
    # out = layers.LSTM(4, return_sequences=True)(inp)
    # model = models.Model(inp, out)
    # train_and_save(model, "Functional_LSTM_test")

    # # 12. RNN - GRU
    # inp = layers.Input(shape=(5, 3))
    # out = layers.GRU(4, return_sequences=True)(inp)
    # model = models.Model(inp, out)
    # train_and_save(model, "Functional_GRU_test")
    
    # Layer Combination
    
    in1 = layers.Input(shape=(16,))
    in2 = layers.Input(shape=(16,))
    x1 = layers.Dense(32, activation="relu")(in1)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Dense(32, activation="sigmoid")(in2)
    merged = layers.Concatenate()([x1, x2])
    added = layers.Add()([merged, merged])
    out = layers.Dense(10, activation="softmax")(added)
    model1 = models.Model([in1, in2], out)
    train_and_save(model1, "Functional_Layer_Combination_1_test")
    
    
    inp1 = layers.Input(shape=(32, 32, 3))
    x1 = layers.Conv2D(8, (3,3), padding="same", data_format="channels_last", activation="relu")(inp1)
    x1 = layers.MaxPooling2D((2,2), data_format="channels_last")(x1)
    x1 = layers.Flatten()(x1)
    inp2 = layers.Input(shape=(3, 32, 32))
    x2 = layers.Conv2D(8, (5,5), padding="valid", data_format="channels_first")(inp2)
    x2 = layers.MaxPooling2D((2,2), data_format="channels_first")(x2)
    x2 = layers.Flatten()(x2)
    merged = layers.Concatenate()([x1, x2])
    out = layers.Dense(20, activation=activations.swish)(merged)
    model2 = models.Model([inp1, inp2], out)
    train_and_save(model2, "Functional_Layer_Combination_2_test")
    
    
    in1 = layers.Input(shape=(12,))
    in2 = layers.Input(shape=(12,))
    x1 = layers.Dense(24, activation="tanh")(in1)
    x1 = layers.Reshape((4, 6))(x1)
    x1 = layers.Permute((2,1))(x1)
    x2 = layers.Dense(24, activation="relu")(in2)
    x2 = layers.Reshape((6, 4))(x2)
    mul = layers.Multiply()([x1, x2])
    sub = layers.Subtract()([x1, x2])
    merged = layers.Concatenate()([mul, sub])
    flat = layers.Flatten()(merged)
    dense = layers.Dense(16)(flat)
    out = layers.LeakyReLU()(dense)
    model3 = models.Model([in1, in2], out)
    train_and_save(model3, "Functional_Layer_Combination_3_test")

