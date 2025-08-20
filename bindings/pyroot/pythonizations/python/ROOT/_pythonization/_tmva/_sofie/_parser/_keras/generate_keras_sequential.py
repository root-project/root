# sequential_models.py
import numpy as np
from keras import models, layers, activations

def generate_keras_sequential(dst_dir):
    # Helper training function
    def train_and_save(model, name):
        x_train = np.random.rand(32, *model.input_shape[1:])
        y_train = np.random.rand(32, *model.output_shape[1:])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        model.fit(x_train, y_train, epochs=1, verbose=0)
        model.save(f"{dst_dir}/{name}.h5")
        # print(f"Saved {name}.h5")

    # 1. Dropout
    # model = models.Sequential([
    #     layers.Input(shape=(10,)),
    #     layers.Dropout(0.5) # Dropout
    # ])
    # train_and_save(model, "Sequential_Dropout_test")

    # 2. Binary Ops: Add, Subtract, Multiply are not typical in Sequential — skipping here

    # 3. Concat (not applicable in Sequential without multi-input)

    # 4. Reshape
    model = models.Sequential([
        layers.Input(shape=(4, 5)),
        layers.Reshape((2, 10))
    ])
    train_and_save(model, "Sequential_Reshape_test")

    # 5. Flatten
    model = models.Sequential([
        layers.Input(shape=(4, 5)),
        layers.Flatten()
    ])
    train_and_save(model, "Sequential_Flatten_test")

    # 6. BatchNorm 1D
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.BatchNormalization()
    ])
    train_and_save(model, "Sequential_BatchNorm1D_test")

    # # 6. BatchNorm 2D
    # model = models.Sequential([
    #     layers.Input(shape=(8, 3)),
    #     layers.BatchNormalization()
    # ])
    # train_and_save(model, "Sequential_BatchNorm2D_test")

    # 7. Activation Functions
    for act in ['relu', 'selu', 'sigmoid', 'softmax', 'tanh']:
        model = models.Sequential([
            layers.Input(shape=(10,)),
            layers.Activation(act)
        ])
        train_and_save(model, f"Sequential_{act.capitalize()}_test")

    # LeakyReLU
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.LeakyReLU()
    ])
    train_and_save(model, "Sequential_LeakyReLU_test")

    # Swish
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Activation(activations.swish)
    ])
    train_and_save(model, "Sequential_Swish_test")

    # 8. Permute
    model = models.Sequential([
        layers.Input(shape=(3, 4, 5)),
        layers.Permute((2, 1, 3))
    ])
    train_and_save(model, "Sequential_Permute_test")

    # 9. Dense
    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(5)
    ])
    train_and_save(model, "Sequential_Dense_test")

    # 10. Conv2D channels_last
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),
        layers.Conv2D(4, (3, 3), data_format='channels_last')
    ])
    train_and_save(model, "Sequential_Conv2D_channels_last_test")

    # 10. Conv2D channels_first
    model = models.Sequential([
        layers.Input(shape=(3, 8, 8)),
        layers.Conv2D(4, (3, 3), data_format='channels_first')
    ])
    train_and_save(model, "Sequential_Conv2D_channels_first_test")
    
    # Conv2D padding_same
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),  
        layers.Conv2D(4, (3, 3), padding='same', data_format='channels_last')
    ])
    train_and_save(model, "Sequential_Conv2D_padding_same_test")
    
    # Conv2D padding_valid
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),  
        layers.Conv2D(4, (3, 3), padding='valid', data_format='channels_last')
    ])
    train_and_save(model, "Sequential_Conv2D_padding_valid_test")

    # 11. MaxPooling2D channels_last
    model = models.Sequential([
        layers.Input(shape=(8, 8, 3)),
        layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')
    ])
    train_and_save(model, "Sequential_MaxPool2D_channels_last_test")

    # 11. MaxPooling2D channels_first
    model = models.Sequential([
        layers.Input(shape=(3, 8, 8)),
        layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
    ])
    train_and_save(model, "Sequential_MaxPool2D_channels_first_test")

    # # 12. RNN - SimpleRNN
    # model = models.Sequential([
    #     layers.Input(shape=(5, 3)),
    #     layers.SimpleRNN(4, return_sequences=True)
    # ])
    # train_and_save(model, "Sequential_SimpleRNN_test")

    # # 12. RNN - LSTM
    # model = models.Sequential([
    #     layers.Input(shape=(5, 3)),
    #     layers.LSTM(4, return_sequences=True)
    # ])
    # train_and_save(model, "Sequential_LSTM_test")

    # # 12. RNN - GRU
    # model = models.Sequential([
    #     layers.Input(shape=(5, 3)),
    #     layers.GRU(4, return_sequences=True)
    # ])
    # train_and_save(model, "Sequential_GRU_test")
    
    # Layer combinations
    
    model = models.Sequential([
        layers.Input(shape=(20,)),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(16, activation="sigmoid"),
        layers.Dense(8, activation="softmax"),
    ])
    train_and_save(model, "Sequential_Layer_Combination_1_test")

    model2 = models.Sequential([
        layers.Input(shape=(28, 28, 3)),  
        layers.Conv2D(16, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (5,5), padding="valid"),
        layers.Flatten(),
        layers.Dense(32, activation="swish"),
        layers.Dense(10, activation="softmax"),
    ])
    train_and_save(model2, "Sequential_Layer_Combination_2_test")

    model3 = models.Sequential([
        layers.Input(shape=(3, 32, 32)),
        layers.Conv2D(8, (3,3), padding="same", data_format="channels_first"),
        layers.MaxPooling2D((2,2), data_format="channels_first"),
        layers.Flatten(),                      
        layers.Reshape((64, 32)),              
        layers.Permute((2,1)),                
        layers.Flatten(),
        layers.Dense(16),
        layers.LeakyReLU(),
    ])

    train_and_save(model3, "Sequential_Layer_Combination_3_test")
