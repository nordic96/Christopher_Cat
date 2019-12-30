from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def create_model(img_height, img_width, channel_size, num_class):
    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=(img_height, img_width, channel_size)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


class NeuralNetworkModel:
    def __init__(self, img_height, img_width, channel_size, num_class):
        self.model = create_model(img_height, img_width, channel_size, num_class)