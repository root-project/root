def generate_keras_functional(dst_dir):

    import numpy as np
    from keras import layers, models
    from parser_test_function import is_channels_first_supported


    # Helper training function
    def train_and_save(model, name):
        # Handle multiple inputs dynamically
        if isinstance(model.input_shape, list):
            x_train = [np.random.rand(32, *shape[1:]) for shape in model.input_shape]
        else:
            x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.summary()
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Functional_{name}_test.keras")
        print("generated and saved functional model",name)


    # Activation Functions
    for act in ['relu', 'elu', 'leaky_relu', 'selu', 'sigmoid', 'softmax', 'swish', 'tanh']:
        inp = layers.Input(shape=(10,))
        out = layers.Activation(act)(inp)
        model = models.Model(inp, out)
        train_and_save(model, f"Activation_layer_{act.capitalize()}")
    # Along with these, Keras allows explicit delcaration of activation layers such as:
    # [ELU, ReLU, LeakyReLU, Softmax]

    # Add
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Add()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Add")

    # AveragePooling2D channels_first
    if (is_channels_first_supported()):
      inp = layers.Input(shape=(3, 8, 8))
      out = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(inp)
      model = models.Model(inp, out)
      train_and_save(model, "AveragePooling2D_channels_first")

    # AveragePooling2D channels_last
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "AveragePooling2D_channels_last")

    # BatchNorm
    inp = layers.Input(shape=(10, 3, 5))
    out = layers.BatchNormalization(axis=2)(inp)
    model = models.Model(inp, out)
    train_and_save(model, "BatchNorm")

    # Concat
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Concatenate()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Concat")

    # Conv2D channels_first
    if (is_channels_first_supported()):
      inp = layers.Input(shape=(3, 8, 8))
      out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_first', activation='relu')(inp)
      model = models.Model(inp, out)
      train_and_save(model, "Conv2D_channels_first")

    # Conv2D channels_last
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last', activation='leaky_relu')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Conv2D_channels_last")

    # Conv2D padding_same
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Conv2D_padding_same")

    # Conv2D padding_valid
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.Conv2D(4, (3, 3), padding='valid', data_format='channels_last', activation='elu')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Conv2D_padding_valid")

    # Dense
    inp = layers.Input(shape=(10,))
    out = layers.Dense(5, activation='tanh')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Dense")

    # ELU
    inp = layers.Input(shape=(10,))
    out = layers.ELU(alpha=0.5)(inp)
    model = models.Model(inp, out)
    train_and_save(model, "ELU")

    # Flatten
    inp = layers.Input(shape=(4, 5))
    out = layers.Flatten()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Flatten")

    # GlobalAveragePooling2D channels first
    if (is_channels_first_supported):
      inp = layers.Input(shape=(3, 4, 6))
      out = layers.GlobalAveragePooling2D(data_format='channels_first')(inp)
      model = models.Model(inp, out)
      train_and_save(model, "GlobalAveragePooling2D_channels_first")

    # GlobalAveragePooling2D channels last
    inp = layers.Input(shape=(4, 6, 3))
    out = layers.GlobalAveragePooling2D(data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "GlobalAveragePooling2D_channels_last")

    # LayerNorm
    inp = layers.Input(shape=(10, 3, 5))
    out = layers.LayerNormalization(axis=-1)(inp)
    model = models.Model(inp, out)
    train_and_save(model, "LayerNorm")

    # LeakyReLU
    inp = layers.Input(shape=(10,))
    out = layers.LeakyReLU()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "LeakyReLU")

    # MaxPooling2D channels_first
    if (is_channels_first_supported):
       inp = layers.Input(shape=(3, 8, 8))
       out = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(inp)
       model = models.Model(inp, out)
       train_and_save(model, "MaxPool2D_channels_first")

    # MaxPooling2D channels_last
    inp = layers.Input(shape=(8, 8, 3))
    out = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(inp)
    model = models.Model(inp, out)
    train_and_save(model, "MaxPool2D_channels_last")

    # Multiply
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Multiply()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Multiply")

    # Permute
    inp = layers.Input(shape=(3, 4, 5))
    out = layers.Permute((2, 1, 3))(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Permute")

    # ReLU
    inp = layers.Input(shape=(10,))
    out = layers.ReLU()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "ReLU")

    # Reshape
    inp = layers.Input(shape=(4, 5))
    out = layers.Reshape((2, 10))(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Reshape")

    # Softmax
    inp = layers.Input(shape=(10,))
    out = layers.Softmax()(inp)
    model = models.Model(inp, out)
    train_and_save(model, "Softmax")

    # Subtract
    in1 = layers.Input(shape=(8,))
    in2 = layers.Input(shape=(8,))
    out = layers.Subtract()([in1, in2])
    model = models.Model([in1, in2], out)
    train_and_save(model, "Subtract")

    # Layer Combination

    inp = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(8, (3,3), padding="same", activation="relu", data_format='channels_last')(inp)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Reshape((16, 16, 8))(x)
    x = layers.Permute((3, 1, 2))(x)
    x = layers.Flatten()(x)
    out = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inp, out)
    train_and_save(model, "Layer_Combination_1")

    inp = layers.Input(shape=(20,))
    x = layers.Dense(32, activation="tanh")(inp)
    x = layers.Dense(16)(x)
    x = layers.ELU()(x)
    x = layers.LayerNormalization()(x)
    out = layers.Dense(5, activation="sigmoid")(x)
    model = models.Model(inp, out)
    train_and_save(model, "Layer_Combination_2")

    inp1 = layers.Input(shape=(16,))
    inp2 = layers.Input(shape=(16,))
    d1 = layers.Dense(16, activation="relu")(inp1)
    d2 = layers.Dense(16, activation="selu")(inp2)
    add = layers.Add()([d1, d2])
    sub = layers.Subtract()([d1, d2])
    mul = layers.Multiply()([d1, d2])
    merged = layers.Concatenate()([add, sub, mul])
    merged = layers.LeakyReLU(alpha=0.1)(merged)
    out = layers.Dense(4, activation="softmax")(merged)
    model = models.Model([inp1, inp2], out)
    train_and_save(model, "Layer_Combination_3")
