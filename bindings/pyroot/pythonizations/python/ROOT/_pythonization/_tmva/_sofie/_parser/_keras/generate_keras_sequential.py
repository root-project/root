def generate_keras_sequential(dst_dir):
    
    from keras import models, layers
    import numpy as np
    
    # Helper training function
    def train_and_save(model, name):
        x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/Sequential_{name}_test.h5")

    # Binary Ops: Add, Subtract, Multiply are not typical in Sequential - skipping those
    # Concat (not applicable in Sequential without multi-input)
    
    # Activation Functions
    for act in ['relu', 'elu', 'leaky_relu', 'selu', 'sigmoid', 'softmax', 'swish', 'tanh']:
        model = models.Sequential([
            layers.Input(shape=(10,)),
            layers.Activation(act)
        ])
        train_and_save(model, f"Activation_layer_{act.capitalize()}")
    # Along with this, Keras also allows explicit delcaration of activation layers such as:
    # ELU, ReLU, LeakyReLU, Softmax
        
    # AveragePooling2D channels_first
    model = models.Sequential([
        layers.Input(shape=(3, 8, 8)),
        layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_first')
    ])
    train_and_save(model, "AveragePooling2D_channels_first")
    
    # AveragePooling2D channels_last
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),
        layers.AveragePooling2D(pool_size=(2, 2), data_format='channels_last')
    ])
    train_and_save(model, "AveragePooling2D_channels_last")

    # BatchNorm 
    model = models.Sequential([
        layers.Input(shape=(10, 3, 5)),
        layers.BatchNormalization(axis=2)
    ])
    train_and_save(model, "BatchNorm")
    
    # Conv2D channels_first
    model = models.Sequential([
        layers.Input(shape=(3, 8, 8)),
        layers.Conv2D(4, (3, 3), data_format='channels_first')
    ])
    train_and_save(model, "Conv2D_channels_first")
    
    # Conv2D channels_last
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),
        layers.Conv2D(4, (3, 3), data_format='channels_last', activation='tanh')
    ])
    train_and_save(model, "Conv2D_channels_last")
    
    # Conv2D padding_same
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),  
        layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last', activation='selu')
    ])
    train_and_save(model, "Conv2D_padding_same")
    
    # Conv2D padding_valid
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),  
        layers.Conv2D(4, (3, 3), padding='valid', data_format='channels_last', activation='swish')
    ])
    train_and_save(model, "Conv2D_padding_valid")
    
    # Dense
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(5, activation='sigmoid')
    ])
    train_and_save(model, "Dense")
    
    # ELU
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.ELU(alpha=0.5)
    ])
    train_and_save(model, "ELU")
    
    # Flatten
    model = models.Sequential([
        layers.Input(shape=(4, 5)),
        layers.Flatten()
    ])
    train_and_save(model, "Flatten")
    
    # GlobalAveragePooling2D channels first
    model = models.Sequential([
        layers.Input(shape=(3, 4, 6)), 
        layers.GlobalAveragePooling2D(data_format='channels_first')
    ])
    train_and_save(model, "GlobalAveragePooling2D_channels_first")
    
    # GlobalAveragePooling2D channels last
    model = models.Sequential([
        layers.Input(shape=(4, 6, 3)), 
        layers.GlobalAveragePooling2D(data_format='channels_last')
    ])
    train_and_save(model, "GlobalAveragePooling2D_channels_last")
    
    # LayerNorm
    model = models.Sequential([
        layers.Input(shape=(10, 3, 5)),
        layers.LayerNormalization(axis=-1)
    ])
    train_and_save(model, "LayerNorm")
    
    # LeakyReLU
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.LeakyReLU()
    ])
    train_and_save(model, "LeakyReLU")
    
    # MaxPooling2D channels_first
    model = models.Sequential([
        layers.Input(shape=(3, 8, 8)),
        layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
    ])
    train_and_save(model, "MaxPool2D_channels_first")
    
    # MaxPooling2D channels_last
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),
        layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')
    ])
    train_and_save(model, "MaxPool2D_channels_last")
    
    # Permute
    model = models.Sequential([
        layers.Input(shape=(3, 4, 5)),
        layers.Permute((2, 1, 3))
    ])
    train_and_save(model, "Permute")

    # Reshape
    model = models.Sequential([
        layers.Input(shape=(4, 5)),
        layers.Reshape((2, 10))
    ])
    train_and_save(model, "Reshape")
    
    # ReLU
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.ReLU()
    ])
    train_and_save(model, "ReLU")

    # Softmax
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Softmax()
    ])
    train_and_save(model, "Softmax")
    
    # Layer Combination
    
    modelA = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, (3,3), padding='same', activation='swish'),
        layers.AveragePooling2D((2,2), data_format='channels_last'),
        layers.GlobalAveragePooling2D(data_format='channels_last'),
        layers.Dense(10, activation='softmax'),
    ])
    train_and_save(modelA, "Layer_Combination_1")

    modelB = models.Sequential([
        layers.Input(shape=(3, 32, 32)),   
        layers.Conv2D(8, (3,3), padding='valid', data_format='channels_first', activation='relu'),
        layers.MaxPooling2D((2,2), data_format='channels_first'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),   
        layers.Reshape((16, 8)),                 
        layers.Permute((2, 1)),                  
        layers.Flatten(),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.1),             
        layers.Dense(10, activation='softmax'),
    ])
    train_and_save(modelB, "Layer_Combination_2")

    modelC = models.Sequential([
        layers.Input(shape=(4, 8, 2)),
        layers.Permute((2, 1, 3)),            
        layers.Reshape((8, 8, 1)),            
        layers.Conv2D(4, (3,3), padding='same', activation='relu'),
        layers.AveragePooling2D((2,2)),       
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(32, activation='elu'),   
        layers.Dense(8, activation='swish'),  
        layers.Dense(3, activation='softmax'),
    ])
    train_and_save(modelC, "Layer_Combination_3")